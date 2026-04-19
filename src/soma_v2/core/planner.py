"""
SOMA V2 — PlanGraph + Executor
================================
DAG-based multi-step planner for DeliberativeAgent (complex tasks).

Architecture
------------
PlanGraph  : DAG of PlanNode objects. Nodes declare dependencies; the
             Executor resolves a topological order and runs nodes whose
             deps are satisfied.
Executor   : Iterates the DAG in topological waves. On node failure,
             marks dependents SKIPPED and tries alternative paths if
             declared. Returns a PlanResult with all node outcomes.

Design notes
------------
- Pure-Python dataclasses + asyncio; no external graph library.
- Each node calls the injected llm_callback — Executor stays
  provider-agnostic.
- Backtracking: if a node fails and has an `alternative` node, the
  Executor swaps in the alternative and re-attempts the subgraph.
  Max one backtrack per node to prevent infinite loops.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Callable, Coroutine, Dict, List, Optional
from .tools import ToolRegistry

_DELEGATE_RE = re.compile(r'\[DELEGATE\]\s*(.+)', re.IGNORECASE)
_CMD_UNIT_RE = re.compile(r'\[CMD\]\s+\w+\s+([A-Z]\d+)', re.IGNORECASE)  # [CMD] VERB UNIT

logger = logging.getLogger("SOMA_V2.PLANNER")


# ── node status ──────────────────────────────────────────────────────────────

class NodeStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUSPENDED = "suspended"   # Waiting for human approval
    DONE      = "done"
    FAILED    = "failed"
    SKIPPED   = "skipped"
    RETRIED   = "retried"


# ── plan node ────────────────────────────────────────────────────────────────

@dataclass
class PlanNode:
    """
    One step in the plan DAG.

    Parameters
    ----------
    node_id : str
        Unique ID for this node (e.g. s1, s2).
    description : str
        Natural language description of the step.
    deps : List[str]
        IDs of nodes that must complete before this one starts.
    alternative : Optional[str]
        ID of a node to attempt if this one fails.
    command : Optional[str]
        Explicit structured command or tool name.
    interrupt : bool
        If True, the executor pauses before running this node.
    """
    node_id:     str
    description: str
    deps:        List[str] = field(default_factory=list)
    alternative: Optional[str] = None
    command:     Optional[str] = None
    interrupt:   bool = False

    # runtime state — set by Executor
    status:     NodeStatus      = field(default=NodeStatus.PENDING, init=False)
    output:     Optional[str]   = field(default=None,              init=False)
    latency_ms: float           = field(default=0.0,               init=False)
    retried:    bool            = field(default=False,             init=False)


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
        Returns nodes grouped into dependency waves.
        Wave 0 = no deps, wave N = deps all in waves 0..N-1.
        Raises ValueError on cycle detection.
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for node in self._nodes.values():
            for dep in node.deps:
                if dep not in self._nodes:
                    raise ValueError(f"Node '{node.node_id}' depends on unknown '{dep}'")
                in_degree[node.node_id] += 1

        waves:     List[List[str]] = []
        remaining: set             = set(self._nodes.keys())

        while remaining:
            wave = [
                nid for nid in remaining
                if in_degree[nid] == 0
            ]
            if not wave:
                raise ValueError(f"Cycle detected in PlanGraph; remaining: {remaining}")
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
    success:      bool
    steps_done:   int
    steps_failed: int
    steps_skipped: int
    total_latency_ms: float
    node_outcomes: Dict[str, Dict[str, Any]]
    summary:      str


# ── executor ─────────────────────────────────────────────────────────────────

class PlanExecutor:
    """
    Executes a PlanGraph wave-by-wave.

    Each node fires the llm_callback with its description as the prompt.
    Nodes within a wave run concurrently (asyncio.gather).
    On failure the Executor checks for an `alternative` node, inserts it
    into the graph, and re-evaluates the remaining waves.
    """

    def __init__(
        self,
        llm_callback: Optional[Callable[..., Coroutine]],
        task_type: str = "deliberative",
        actuator: Optional[Any] = None,
        delegate_fn: Optional[Callable[..., Coroutine]] = None,
        blackboard: Optional[Any] = None,
        agent_id: str = "unknown",
        negotiation_broker: Optional[Any] = None,
        claim_timeout_s: float = 2.0,
        delegate_timeout_s: float = 10.0,
        resource_pools: Optional[Dict[str, Any]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        telemetry: Optional[Any] = None,
        task_id: Optional[str] = None,
    ) -> None:
        self.llm_callback       = llm_callback
        self.task_type          = task_type
        self.actuator           = actuator
        self.delegate_fn        = delegate_fn
        self.blackboard         = blackboard
        self.agent_id           = agent_id
        self.negotiation_broker = negotiation_broker
        self.claim_timeout_s    = claim_timeout_s
        self.delegate_timeout_s = delegate_timeout_s
        self.resource_pools     = resource_pools or {}
        self.tool_registry      = tool_registry
        self.telemetry          = telemetry
        self.task_id            = task_id
        self.approval_event     = asyncio.Event()
        self.suspended_node: Optional[PlanNode] = None
        self._negotiated_load   = 0   # count of negotiated steps currently executing

        # Self-register with broker so other agents can propose steps to us
        if negotiation_broker is not None:
            negotiation_broker.register(agent_id, self)

    async def _execute_negotiated_step(self, step_desc: str, context: str) -> str:
        """
        Execute a single step on behalf of another agent (negotiation accept path).
        Increments _negotiated_load so the broker can back-pressure further proposals.
        """
        self._negotiated_load += 1
        try:
            node = PlanNode(node_id="__neg__", description=step_desc, deps=[])
            await self._run_node(node, context)
            return node.output or "[negotiated step completed]"
        finally:
            self._negotiated_load = max(0, self._negotiated_load - 1)

    async def _run_node(self, node: PlanNode, context: str) -> None:
        if node.interrupt:
            logger.info(f"PlanNode '{node.node_id}': SUSPENDED (waiting for human approval)")
            node.status = NodeStatus.SUSPENDED
            self.suspended_node = node
            self.approval_event.clear()
            await self.approval_event.wait()
            self.suspended_node = None
            logger.info(f"PlanNode '{node.node_id}': RESUMED after approval")

        node.status = NodeStatus.RUNNING
        t0 = time.perf_counter()
        try:
            # ── [DELEGATE] — route sub-task to another agent via A2A ─────────
            delegate_match = _DELEGATE_RE.search(node.description)
            if delegate_match and self.delegate_fn:
                sub_task = delegate_match.group(1).strip()
                logger.info(f"PlanNode '{node.node_id}': delegating sub-task '{sub_task[:60]}'")
                try:
                    result = await asyncio.wait_for(
                        self.delegate_fn(sub_task, urgency="high"),
                        timeout=self.delegate_timeout_s
                    )
                    node.output = f"[DELEGATED] {result}"
                except asyncio.TimeoutError:
                    raise RuntimeError(f"Delegation timed out after {self.delegate_timeout_s}s")
            elif self.llm_callback:
                prompt = (
                    f"Context: {context}\n\n"
                    f"Step: {node.description}\n\n"
                    f"Execute this step and describe the outcome in 1-2 sentences."
                )
                node.output = await self.llm_callback(self.task_type, prompt)
            else:
                node.output = f"[stub] completed: {node.description}"

            # ── Actuation Bridge: Tools, [CMD], or structured command ──────────
            if self.actuator:
                cmd_part = None
                if node.command:
                    cmd_part = node.command
                elif "[CMD]" in node.description:
                    cmd_part = node.description.split("[CMD]")[-1].strip()
                    cmd_part = cmd_part.split("[DELEGATE]")[0].strip()

                if cmd_part:
                    # 1. Check if it's a registered tool
                    if self.tool_registry:
                        tool = self.tool_registry.get(cmd_part)
                        if tool:
                            logger.info(f"PlanExecutor: executing tool '{tool.name}' for node {node.node_id}")
                            if self.telemetry:
                                self.telemetry.log_event("tool_call", {
                                    "task_id": self.task_id,
                                    "agent_id": self.agent_id,
                                    "node_id": node.node_id,
                                    "tool": tool.name,
                                    "cmd": cmd_part
                                })
                            try:
                                tool_result = await tool.func()
                                node.output = (node.output or "") + f" | [TOOL_RESULT: {tool_result}]"
                                node.status = NodeStatus.DONE
                                return
                            except Exception as tool_exc:
                                logger.error(f"Tool '{tool.name}' failed: {tool_exc}")
                                raise

                    # 2. Standard Actuator logic (Blackboard + [CMD])
                    # Claim the unit on the shared blackboard before actuating
                    unit_match  = _CMD_UNIT_RE.search(node.description)
                    unit_id     = unit_match.group(1) if unit_match else None
                    actual_unit = unit_id   # may differ when pool-claimed
                    claimed     = False
                    pool_claimed = False    # True if claimed via pool (released at task end)
                    if unit_id and self.blackboard:
                        pool = self.resource_pools.get(unit_id)
                        if pool is not None:
                            existing = self._task_pool_claims.get(pool.pool_id)
                            if existing is not None:
                                # Reuse the unit claimed earlier in this task (reentrant)
                                actual_unit = existing
                                claimed = await self.blackboard.claim(actual_unit, self.agent_id, node.node_id, timeout_s=self.claim_timeout_s)
                            else:
                                # First pool-eligible node: grab any free drone
                                actual_unit = await pool.claim_any(
                                    self.agent_id, self.blackboard, node.node_id,
                                    timeout_s=self.claim_timeout_s,
                                )
                                claimed = actual_unit is not None
                                if claimed:
                                    self._task_pool_claims[pool.pool_id] = actual_unit
                                    pool_claimed = True  # held until task end
                            if claimed and actual_unit != unit_id:
                                cmd_part = cmd_part.replace(unit_id, actual_unit, 1)
                        else:
                            claimed = await self.blackboard.claim(unit_id, self.agent_id, node.node_id, timeout_s=self.claim_timeout_s)

                        if not claimed:
                            # Try to negotiate: propose the step to whoever owns the unit
                            if self.negotiation_broker:
                                neg = await self.negotiation_broker.propose(
                                    from_agent=self.agent_id,
                                    step_desc=node.description,
                                    unit_id=unit_id,
                                    context=context,
                                )
                                if neg.accepted:
                                    node.output = (
                                        f"[NEGOTIATED→{neg.by_agent}] {neg.result}"
                                    )
                                    node.status = NodeStatus.DONE
                                    node.latency_ms = neg.latency_ms
                                    return
                            raise RuntimeError(
                                f"Resource conflict: unit '{unit_id}' unavailable "
                                f"and negotiation failed — node will backtrack"
                            )
                    try:
                        act_success = await self.actuator.execute_command(cmd_part)
                        if not act_success:
                            raise RuntimeError(f"Actuator failed: {cmd_part}")
                        node.output = (node.output or "") + f" | [ACTUATED: {cmd_part}]"
                    finally:
                        # Pool-claimed units are held for the full task — released in execute()
                        if self.blackboard and claimed and not pool_claimed:
                            await self.blackboard.release(actual_unit, self.agent_id, node.node_id)

            node.status = NodeStatus.DONE
        except Exception as exc:
            logger.warning(f"PlanNode '{node.node_id}' failed: {exc}")
            node.output = str(exc)
            node.status = NodeStatus.FAILED
        finally:
            node.latency_ms = (time.perf_counter() - t0) * 1000

    async def execute(self, graph: PlanGraph, context: str, task_id: Optional[str] = None) -> PlanResult:
        """Run all nodes in topological wave order with backtrack on failure."""
        t_total = time.perf_counter()
        backtracks_used: set = set()
        self._task_pool_claims: Dict[str, str] = {}  # pool_id -> actual_unit; held for task duration

        # We re-compute waves after any backtrack insertion
        max_iterations = len(graph) * 2 + 5
        iteration      = 0

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

            # Find the next wave with at least one PENDING node
            pending_wave = None
            for wave in waves:
                nodes_in_wave = [graph.get(nid) for nid in wave]
                if any(n.status == NodeStatus.PENDING for n in nodes_in_wave):
                    # Check all deps are DONE (not SKIPPED/FAILED)
                    wave_ready = []
                    for nid in wave:
                        node = graph.get(nid)
                        if node.status != NodeStatus.PENDING:
                            continue
                        deps_ok = all(
                            graph.get(d).status == NodeStatus.DONE
                            for d in node.deps
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
                break  # nothing runnable left

            # Run the wave concurrently
            await asyncio.gather(*[
                self._run_node(node, context)
                for node in pending_wave
            ])

            # Backtrack: for any failed node with an alternative, insert alt
            for node in pending_wave:
                if (
                    node.status == NodeStatus.FAILED
                    and node.alternative
                    and node.node_id not in backtracks_used
                    and node.alternative not in [n.node_id for n in graph.all_nodes()]
                ):
                    backtracks_used.add(node.node_id)
                    alt = PlanNode(
                        node_id=node.alternative,
                        description=f"[backtrack] retry of '{node.node_id}': {node.description}",
                        deps=node.deps,  # same deps, fresh attempt
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

        success = len(fail_nodes) == 0 and len(done_nodes) > 0
        summary_parts = [f"Completed {len(done_nodes)}/{len(nodes)} steps."]
        if fail_nodes:
            summary_parts.append(f"Failed: {[n.node_id for n in fail_nodes]}.")
        if backtracks_used:
            summary_parts.append(f"Backtracks: {list(backtracks_used)}.")

        # Release any units held at task-level via pool claiming
        for pool_id, actual_unit in self._task_pool_claims.items():
            pool = next((p for p in self.resource_pools.values() if p.pool_id == pool_id), None)
            if pool and self.blackboard:
                await pool.release(actual_unit, self.agent_id, "__task_end__", self.blackboard)
        self._task_pool_claims = {}

        return PlanResult(
            success=success,
            steps_done=len(done_nodes),
            steps_failed=len(fail_nodes),
            steps_skipped=len(skip_nodes),
            total_latency_ms=round((time.perf_counter() - t_total) * 1000, 2),
            node_outcomes=outcomes,
            summary=" ".join(summary_parts),
        )
