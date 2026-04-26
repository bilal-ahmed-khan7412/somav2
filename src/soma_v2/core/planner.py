"""
SOMA V2 — PlanGraph + PlanExecutor
====================================
DAG-based multi-step planner for DeliberativeAgent (complex tasks).

Architecture
------------
PlanGraph  : DAG of PlanNode objects. Nodes declare dependencies; the
             Executor resolves topological order and runs nodes whose
             deps are satisfied.
Executor   : Iterates the DAG in topological waves. On node failure,
             marks dependents SKIPPED and tries alternative paths if
             declared. Returns a PlanResult with all node outcomes.

Design notes
------------
- Pure-Python dataclasses + asyncio; no external graph library.
- Each node either:
    (a) calls the injected llm_callback with its description as a prompt, OR
    (b) invokes a registered Tool if node.command matches a tool name.
- Backtracking: if a node fails and has an `alternative` node, the
  Executor inserts the alternative and re-attempts that subgraph.
  Max one backtrack per node to prevent infinite loops.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .tools import ToolRegistry

logger = logging.getLogger("SOMA_V2.PLANNER")


# ── node status ──────────────────────────────────────────────────────────────

class NodeStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUSPENDED = "suspended"
    DONE      = "done"
    FAILED    = "failed"
    SKIPPED   = "skipped"


# ── plan node ────────────────────────────────────────────────────────────────

@dataclass
class PlanNode:
    """
    One step in the plan DAG.

    Parameters
    ----------
    node_id : str     — unique ID (e.g. s1, s2)
    description : str — natural language description of the step
    deps : List[str]  — node IDs that must complete before this one
    alternative : str — node ID to try if this node fails
    command : str     — registered Tool name to call (or None for LLM)
    interrupt : bool  — if True, executor pauses for human approval before running
    """
    node_id:     str
    description: str
    deps:        List[str]      = field(default_factory=list)
    alternative: Optional[str]  = None
    command:     Optional[str]  = None
    params:      Dict[str, Any] = field(default_factory=dict)
    interrupt:   bool           = False

    # runtime state — set by Executor
    status:     NodeStatus    = field(default=NodeStatus.PENDING, init=False)
    output:     Optional[str] = field(default=None,              init=False)
    latency_ms: float         = field(default=0.0,               init=False)


# ── plan graph ───────────────────────────────────────────────────────────────

class PlanGraph:
    """Directed acyclic graph of PlanNode objects."""

    def __init__(self) -> None:
        self._nodes: Dict[str, PlanNode] = {}

    def add_node(self, node: PlanNode) -> "PlanGraph":
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate node_id: {node.node_id}")
        self._nodes[node.node_id] = node
        return self

    def get(self, node_id: str) -> PlanNode:
        return self._nodes[node_id]

    def all_nodes(self) -> List[PlanNode]:
        return list(self._nodes.values())

    def topological_waves(self) -> List[List[str]]:
        """
        Groups nodes into dependency waves.
        Wave 0 = no deps. Wave N = all deps in waves 0..N-1.
        Raises ValueError on cycle detection.
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for node in self._nodes.values():
            for dep in node.deps:
                if dep not in self._nodes:
                    raise ValueError(
                        f"Node '{node.node_id}' depends on unknown '{dep}'"
                    )
                in_degree[node.node_id] += 1

        waves:     List[List[str]] = []
        remaining: set             = set(self._nodes.keys())

        while remaining:
            wave = [nid for nid in remaining if in_degree[nid] == 0]
            if not wave:
                raise ValueError(
                    f"Cycle detected in PlanGraph; remaining: {remaining}"
                )
            waves.append(wave)
            for nid in wave:
                remaining.remove(nid)
                for other in self._nodes.values():
                    if nid in other.deps:
                        in_degree[other.node_id] -= 1
        return waves

    def __len__(self) -> int:
        return len(self._nodes)


# ── plan result ──────────────────────────────────────────────────────────────

@dataclass
class PlanResult:
    success:          bool
    steps_done:       int
    steps_failed:     int
    steps_skipped:    int
    total_latency_ms: float
    node_outcomes:    Dict[str, Dict[str, Any]]
    summary:          str


# ── executor ─────────────────────────────────────────────────────────────────

class PlanExecutor:
    """
    Executes a PlanGraph wave-by-wave.

    For each node the executor:
      1. Checks if node.command matches a registered Tool → calls the tool.
      2. Otherwise calls llm_callback with the step description as a prompt.

    Nodes within a wave run concurrently (asyncio.gather).
    Failed nodes trigger backtracking via their `alternative` field.
    """

    def __init__(
        self,
        llm_callback:  Optional[Callable[..., Coroutine]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        telemetry:     Optional[Any] = None,
        agent_id:      str = "unknown",
        task_id:       Optional[str] = None,
    ) -> None:
        self.llm_callback  = llm_callback
        self.tool_registry = tool_registry
        self.telemetry     = telemetry
        self.agent_id      = agent_id
        self.task_id       = task_id
        self.approval_event     = asyncio.Event()
        self.suspended_node: Optional[PlanNode] = None

    # ── node execution ────────────────────────────────────────────────────────

    async def _run_node(self, node: PlanNode, context: str) -> None:
        """Execute a single plan node."""

        # Human-in-the-loop pause point
        if node.interrupt:
            logger.info(f"PlanNode '{node.node_id}': SUSPENDED — waiting for approval")
            node.status = NodeStatus.SUSPENDED
            self.suspended_node = node
            self.approval_event.clear()
            await self.approval_event.wait()
            self.suspended_node = None
            logger.info(f"PlanNode '{node.node_id}': RESUMED after approval")

        node.status = NodeStatus.RUNNING
        t0 = time.perf_counter()

        try:
            output = await self._execute_node_logic(node, context)
            node.output = output
            node.status = NodeStatus.DONE
        except Exception as exc:
            logger.warning(f"PlanNode '{node.node_id}' failed: {exc}")
            node.output = str(exc)
            node.status = NodeStatus.FAILED
        finally:
            node.latency_ms = (time.perf_counter() - t0) * 1000

    async def _execute_node_logic(self, node: PlanNode, context: str) -> str:
        """
        Core dispatch logic for a single node:
          1. Named tool     → call ToolRegistry
          2. LLM callback   → call LLM with step description
          3. Stub fallback  → return placeholder (no LLM configured)
        """
        # 1. Registered Tool
        if node.command and self.tool_registry:
            tool = self.tool_registry.get(node.command)
            if tool:
                logger.info(
                    f"PlanExecutor: tool '{tool.name}' for node '{node.node_id}'"
                )
                if self.telemetry:
                    self.telemetry.log_event("tool_call", {
                        "task_id":  self.task_id,
                        "agent_id": self.agent_id,
                        "node_id":  node.node_id,
                        "tool":     tool.name,
                        "params":   node.params
                    })
                # Call tool with node params if provided
                try:
                    kwargs = {**node.params}
                    # Automatically inject agent identity if tool expects it
                    import inspect
                    sig = inspect.signature(tool.func)
                    if "my_agent_id" in sig.parameters:
                        kwargs["my_agent_id"] = self.agent_id
                        
                    result = await tool.func(**kwargs)
                except TypeError as e:
                    # Fallback for tools that don't take arguments or have signature mismatches
                    logger.debug(f"Tool call fallback for {tool.name}: {e}")
                    result = await tool.func()
                return f"[TOOL: {tool.name}] {result}"

        # 2. LLM call
        if self.llm_callback:
            prompt = (
                f"Task context: {context}\n\n"
                f"Current step: {node.description}\n\n"
                f"Execute this step and describe the outcome in 1-2 sentences."
            )
            return await self.llm_callback("deliberative_step", prompt)

        # 3. Stub
        return f"[stub] completed: {node.description}"

    # ── graph execution ───────────────────────────────────────────────────────

    async def execute(
        self,
        graph: PlanGraph,
        context: str,
        task_id: Optional[str] = None,
    ) -> PlanResult:
        """Run all nodes in topological wave order with backtracking on failure."""
        if task_id:
            self.task_id = task_id

        t_total = time.perf_counter()
        backtracks_used: set = set()
        max_iterations       = len(graph) * 2 + 5
        iteration            = 0

        while iteration < max_iterations:
            iteration += 1
            try:
                waves = graph.topological_waves()
            except ValueError as exc:
                logger.error(f"PlanGraph topology error: {exc}")
                break

            all_done = all(
                n.status in (NodeStatus.DONE, NodeStatus.SKIPPED, NodeStatus.FAILED)
                for n in graph.all_nodes()
            )
            if all_done:
                break

            # Find the next wave with at least one PENDING / runnable node
            pending_wave = None
            for wave in waves:
                wave_ready = []
                for nid in wave:
                    node = graph.get(nid)
                    if node.status != NodeStatus.PENDING:
                        continue
                    deps_ok = all(
                        graph.get(d).status == NodeStatus.DONE for d in node.deps
                    )
                    deps_blocked = any(
                        graph.get(d).status in (NodeStatus.FAILED, NodeStatus.SKIPPED)
                        for d in node.deps
                    )
                    if deps_blocked:
                        node.status = NodeStatus.SKIPPED
                        logger.debug(f"Skipping '{nid}' — dep failed/skipped")
                    elif deps_ok:
                        wave_ready.append(node)
                if wave_ready:
                    pending_wave = wave_ready
                    break

            if pending_wave is None:
                break

            # Run wave concurrently
            await asyncio.gather(*[
                self._run_node(node, context) for node in pending_wave
            ])

            # Backtracking: insert alternative node for failed nodes
            for node in pending_wave:
                if (
                    node.status == NodeStatus.FAILED
                    and node.alternative
                    and node.node_id not in backtracks_used
                    and node.alternative not in {n.node_id for n in graph.all_nodes()}
                ):
                    backtracks_used.add(node.node_id)
                    alt = PlanNode(
                        node_id=node.alternative,
                        description=f"[retry] alternative for '{node.node_id}': {node.description}",
                        deps=node.deps,
                    )
                    graph.add_node(alt)
                    logger.info(f"Backtrack: inserted alt node '{alt.node_id}'")

        # Build result
        nodes      = graph.all_nodes()
        done_nodes = [n for n in nodes if n.status == NodeStatus.DONE]
        fail_nodes = [n for n in nodes if n.status == NodeStatus.FAILED]
        skip_nodes = [n for n in nodes if n.status == NodeStatus.SKIPPED]

        outcomes = {
            n.node_id: {
                "status":     n.status.value,
                "output":     n.output,
                "latency_ms": round(n.latency_ms, 2),
            }
            for n in nodes
        }

        success       = len(fail_nodes) == 0 and len(done_nodes) > 0
        summary_parts = [f"Completed {len(done_nodes)}/{len(nodes)} steps."]
        if fail_nodes:
            summary_parts.append(f"Failed: {[n.node_id for n in fail_nodes]}.")
        if backtracks_used:
            summary_parts.append(f"Backtracks: {list(backtracks_used)}.")

        return PlanResult(
            success=success,
            steps_done=len(done_nodes),
            steps_failed=len(fail_nodes),
            steps_skipped=len(skip_nodes),
            total_latency_ms=round((time.perf_counter() - t_total) * 1000, 2),
            node_outcomes=outcomes,
            summary=" ".join(summary_parts),
        )
