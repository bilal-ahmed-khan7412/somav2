"""
SOMA V2 — DeliberativeAgent
============================
Handles complex tasks using a DAG-based PlanGraph + PlanExecutor.

Cache hierarchy (fastest → slowest):
  L1  Hot plan cache  — in-process dict keyed by normalised event hash; <0.1ms
  L2  Cold episode store — ChromaDB (semantic) or flat fallback;       ~10-50ms
  L3  LLM plan generation — called only on full cache miss

The planner prompt is domain-agnostic. Steps are executed by calling the LLM
with each step description as context, or by invoking a registered Tool when
the step's 'command' field matches a tool name in the ToolRegistry.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

from ..core.planner import PlanGraph, PlanNode, PlanExecutor
from ..core.tools import ToolRegistry

logger = logging.getLogger("SOMA_V2.DELIBERATIVE")

# ── event normalisation ───────────────────────────────────────────────────────
_STRIP_NUMBERS = re.compile(r"\b\d+\b")
_MULTI_SPACE   = re.compile(r"\s{2,}")


def _normalise(text: str) -> str:
    text = _STRIP_NUMBERS.sub("N", text)
    return _MULTI_SPACE.sub(" ", text).lower().strip()


def _plan_key(text: str) -> str:
    return hashlib.md5(_normalise(text).encode()).hexdigest()[:16]


# ── LLM plan generation ──────────────────────────────────────────────────────

_PLAN_PROMPT = """\
You are a general-purpose task planner for an autonomous multi-agent system.
Given the task below, produce a JSON execution plan.

Task: {event}
Agent Role: {role}
Urgency: {urgency}

Available Tools (agents may call these by name in the 'command' field):
{tools}

Relevant Past Episodes:
{memory}

Return ONLY valid JSON — no markdown fences, no extra text:
{{
  "steps": [
    {{
      "id": "s1",
      "description": "Short description of what this step does",
      "deps": [],
      "command": "tool_name",
      "params": {{"arg1": "val1"}},
      "alternative": null
    }}
  ]
}}

Rules:
- 2 to 6 steps.
- 'deps' is a list of step IDs that must complete before this step starts.
- 'command' is a registered tool name (from Available Tools) if this step calls a tool; otherwise null.
- 'params' is a dictionary of arguments to pass to the tool if command is not null; otherwise empty {{}}.
- 'alternative' is another step ID to attempt if this one fails; otherwise null.
- Keep steps concrete and action-oriented.
"""

_FALLBACK_STEPS = [
    {"id": "assess",  "description": "Assess the task: identify key goals and constraints.",
     "deps": [],         "command": None, "alternative": None},
    {"id": "plan",    "description": "Formulate a resolution plan with concrete actions.",
     "deps": ["assess"], "command": None, "alternative": None},
    {"id": "execute", "description": "Execute the chosen plan step by step.",
     "deps": ["plan"],   "command": None, "alternative": None},
    {"id": "verify",  "description": "Verify the outcome and summarise what was achieved.",
     "deps": ["execute"],"command": None, "alternative": None},
]

_HOT_CACHE_AGENT = "__plan_cache__"


# ── graph helpers ─────────────────────────────────────────────────────────────

def _steps_to_graph(steps: List[Dict]) -> PlanGraph:
    g = PlanGraph()
    for s in steps:
        g.add_node(PlanNode(
            node_id=s["id"],
            description=s["description"],
            deps=s.get("deps", []),
            command=s.get("command"),
            params=s.get("params", {}),
            alternative=s.get("alternative"),
            interrupt=s.get("interrupt", False),
        ))
    return g


def _graph_from_json(plan_json: str) -> Optional[PlanGraph]:
    try:
        data  = json.loads(plan_json)
        graph = _steps_to_graph(data["steps"])
        graph.topological_waves()   # validate — raises on cycle
        return graph
    except Exception as exc:
        logger.warning(f"DeliberativeAgent: cached plan invalid, discarding ({exc})")
        return None


async def _llm_plan(
    event: str,
    role: str,
    urgency: str,
    llm_callback: Callable,
    memory=None,
    tools: str = "None",
) -> PlanGraph:
    # Build memory context
    mem_text = "No relevant past episodes found."
    if memory:
        try:
            episodes = memory.recall_similar(event, n=3)
            if episodes:
                lines = []
                for ep in episodes:
                    meta    = ep.get("metadata", {})
                    status  = "SUCCESS" if meta.get("success") else "FAILED"
                    action  = meta.get("action", "?")
                    dist    = ep.get("distance")
                    sim_str = f"sim={1-dist:.2f}" if dist is not None else "sim=high"
                    lines.append(f"- [{status}] action={action} {sim_str}: {ep['event'][:120]}")
                mem_text = "\n".join(lines)
        except Exception as exc:
            logger.warning(f"DeliberativeAgent: memory recall failed: {exc}")

    prompt = _PLAN_PROMPT.format(
        event=event[:500], role=role, urgency=urgency,
        memory=mem_text, tools=tools,
    )
    try:
        raw = await llm_callback("deliberative_plan", prompt)
        # Extract first JSON block — strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        brace_match = re.search(r"\{[\s\S]*\}", cleaned)
        if brace_match:
            cleaned = brace_match.group(0)
        data  = json.loads(cleaned)
        graph = _steps_to_graph(data["steps"])
        graph.topological_waves()   # validate
        logger.info(f"DeliberativeAgent: LLM plan accepted ({len(graph)} steps)")
        return graph
    except Exception as exc:
        logger.warning(f"DeliberativeAgent: LLM plan parse failed ({exc}), using fallback")
        return _steps_to_graph(_FALLBACK_STEPS)


# ── agent ─────────────────────────────────────────────────────────────────────

class DeliberativeAgent:
    """
    Handles complex tasks via DAG planning.

    Plan resolution order:
      1. L1 hot cache   (in-memory, sub-ms)
      2. L2 cold store  (ChromaDB semantic search, ~10-50ms)
      3. L3 LLM         (full generation, network-bound)
    """

    def __init__(
        self,
        llm_callback: Optional[Callable[..., Coroutine]] = None,
        memory=None,
        cold_threshold: float = 0.40,
        tool_registry: Optional[ToolRegistry] = None,
        telemetry: Optional[Any] = None,
        agent_id: str = "unknown",
        **kwargs,          # absorb any legacy kwargs gracefully
    ):
        self.llm_callback   = llm_callback
        self.memory         = memory
        self.cold_threshold = cold_threshold
        self.tool_registry  = tool_registry
        self.telemetry      = telemetry
        self.agent_id       = agent_id
        self._executor = PlanExecutor(
            llm_callback=llm_callback,
            tool_registry=tool_registry,
            telemetry=telemetry,
            agent_id=agent_id,
        )
        self._cache_log: List[Dict[str, Any]] = []

    def get_suspended_node(self) -> Optional[Any]:
        return self._executor.suspended_node

    def approve(self):
        self._executor.approval_event.set()

    # ── cache helpers ─────────────────────────────────────────────────────────

    def _hot_get(self, event: str) -> Optional[PlanGraph]:
        if not self.memory:
            return None
        key       = _plan_key(event)
        plan_json = self.memory.recall_working(_HOT_CACHE_AGENT, key)
        if plan_json:
            graph = _graph_from_json(plan_json)
            if graph:
                logger.info(f"DeliberativeAgent: L1 hot HIT key={key[:8]}")
                return graph
        return None

    def _hot_put(self, event: str, plan_json: str) -> None:
        if not self.memory:
            return
        key = _plan_key(event)
        self.memory.remember(_HOT_CACHE_AGENT, key, plan_json, ttl=3600.0)

    def _cold_get(self, event: str) -> Optional[PlanGraph]:
        if not self.memory:
            return None
        try:
            episodes = self.memory.recall_similar(event, n=1, only_successes=True)
            if not episodes:
                return None
            ep        = episodes[0]
            dist      = ep.get("distance")
            plan_json = ep["metadata"].get("plan_json")
            if not plan_json:
                return None
            if dist is not None and dist >= self.cold_threshold:
                return None
            graph = _graph_from_json(plan_json)
            if graph:
                logger.info(
                    f"DeliberativeAgent: L2 cold HIT "
                    f"dist={'N/A' if dist is None else f'{dist:.4f}'}"
                )
                self._hot_put(event, plan_json)   # promote to L1
                return graph
        except Exception as exc:
            logger.debug(f"DeliberativeAgent: cold cache error: {exc}")
        return None

    # ── main handler ──────────────────────────────────────────────────────────

    async def handle(
        self,
        event: str,
        agent_role: str,
        confidence: float,
        urgency: str,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(f"DeliberativeAgent: planning for '{event[:80]}'")

        # L1 → L2 → L3
        graph        = self._hot_get(event)
        cached_match = graph is not None
        cache_level  = "L1-hot" if cached_match else None

        if not graph:
            graph = self._cold_get(event)
            if graph:
                cached_match = True
                cache_level  = "L2-cold"

        self._cache_log.append({
            "task_num": len(self._cache_log) + 1,
            "hit":      cached_match,
            "level":    cache_level,
            "event":    event[:60],
            "timestamp": time.time(),
        })

        if self.telemetry:
            self.telemetry.log_event("cache_query", {
                "hit":        cached_match,
                "level":      cache_level or "miss",
                "event_text": event[:60],
            })

        if not graph:
            tools_prompt = "None"
            if self.tool_registry:
                tools_prompt = self.tool_registry.get_prompt()
            if self.llm_callback:
                graph = await _llm_plan(
                    event, agent_role, urgency, self.llm_callback,
                    memory=self.memory, tools=tools_prompt,
                )
            else:
                graph = _steps_to_graph(_FALLBACK_STEPS)

        # Execute the plan
        result = await self._executor.execute(graph, context=event, task_id=task_id)

        # Evict broken cached plans so next call re-plans
        if cached_match and not result.success:
            key = _plan_key(event)
            if self.memory:
                self.memory.forget(_HOT_CACHE_AGENT, key)
                logger.warning(f"DeliberativeAgent: evicted broken L1 plan key={key[:8]}")

        logger.info(
            f"DeliberativeAgent: {result.steps_done}/{len(graph)} steps "
            f"success={result.success} cache={cache_level or 'miss'} "
            f"latency={result.total_latency_ms:.1f}ms"
        )

        done_outputs = [
            v["output"] for v in result.node_outcomes.values()
            if v["status"] == "done" and v["output"]
        ]
        rationale = done_outputs[-1] if done_outputs else result.summary
        action    = "resolve" if result.success else "escalate"

        # Serialise successful new plans for L1 + cold memory
        plan_json: Optional[str] = None
        if result.success and not cached_match:
            try:
                steps = [
                    {"id": n.node_id, "description": n.description,
                     "deps": n.deps, "alternative": n.alternative}
                    for n in graph.all_nodes()
                ]
                plan_json = json.dumps({"steps": steps})
                self._hot_put(event, plan_json)
            except Exception as exc:
                logger.warning(f"DeliberativeAgent: plan serialisation failed: {exc}")

        return {
            "action":       action,
            "confidence":   confidence,
            "urgency":      urgency,
            "rationale":    rationale,
            "steps":        result.steps_done,
            "plan_summary": result.summary,
            "plan_detail":  result.node_outcomes,
            "metadata": {
                "cached":      cached_match,
                "cache_level": cache_level,
                "plan_json":   plan_json,
            },
        }
