"""
SOMA V2 — AgentDirector
========================
Orchestrates a pool of named AgentSlot objects (each wrapping a V2Kernel).
Assigns incoming tasks via a fast local slot-selection algorithm.

Lifecycle of a task
-------------------
1. Director receives task via assign(event, ...).
2. Director picks the least-loaded available slot locally (no message round-trip).
3. Selected slot executes the task via its V2Kernel.handle().
4. Director returns TASK_RESULT to the caller.

Pool roles
----------
EMERGENCY / SUPERVISOR slots take priority for high/emergency urgency tasks.
PEER / ROUTINE slots reject tasks if current_load >= capacity.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from .a2a import A2ABus
from .blackboard import ResourceBlackboard
from .broker import NegotiationBroker
from .kernel import V2Kernel
from .tools import ToolRegistry
from ..memory.hierarchical import HierarchicalMemory

logger = logging.getLogger("SOMA_V2.DIRECTOR")

PRIORITY_ROLES    = {"EMERGENCY", "SUPERVISOR"}
MAX_DELEGATE_HOPS = 2


# ── agent slot ────────────────────────────────────────────────────────────────

@dataclass
class AgentSlot:
    """One agent in the pool. Wraps a V2Kernel instance."""
    slot_id:  str
    role:     str           # EMERGENCY | SUPERVISOR | PEER | ROUTINE
    capacity: int = 4       # max concurrent tasks
    kernel:   Optional[V2Kernel] = field(default=None, repr=False)
    memory:   Optional[HierarchicalMemory] = field(default=None, repr=False)

    _load: int = field(default=0, init=False, repr=False)

    @property
    def load(self) -> int:
        return self._load

    @property
    def available(self) -> bool:
        return self._load < self.capacity


# ── director ─────────────────────────────────────────────────────────────────

class AgentDirector:
    """
    Pool manager.

    Usage
    -----
    director = AgentDirector(llm_callback=my_llm)
    director.add_slot("peer_1", role="PEER")
    director.add_slot("supervisor_1", role="SUPERVISOR")
    result = await director.assign("Summarize the quarterly report", urgency="high")
    await director.stop()
    """

    def __init__(
        self,
        llm_callback=None,
        memory: Optional[HierarchicalMemory] = None,
        tool_registry: Optional[ToolRegistry] = None,
        telemetry: Optional[Any] = None,
        bus: Optional[A2ABus] = None,
        **kernel_kwargs,
    ) -> None:
        self.llm_callback   = llm_callback
        self.tool_registry  = tool_registry
        self.telemetry      = telemetry
        self._kernel_kwargs = kernel_kwargs
        self._memory        = memory or HierarchicalMemory()
        self._bus           = bus or A2ABus(telemetry=self.telemetry)
        self._blackboard    = ResourceBlackboard(bus=self._bus)
        self._negotiator    = NegotiationBroker(
            blackboard=self._blackboard, bus=self._bus
        )
        self._slots:      Dict[str, AgentSlot] = {}
        self._stats: Dict[str, int] = {
            "tasks_assigned": 0,
            "tasks_failed":   0,
            "overflow_routes": 0,
        }
        self._rr_counter: int = 0

    def add_slot(
        self,
        slot_id:  str,
        role:     str = "PEER",
        capacity: int = 4,
    ) -> "AgentDirector":
        """Add a new agent slot to the pool."""
        kernel = V2Kernel(
            llm_callback=self.llm_callback,
            memory=self._memory,
            tool_registry=self.tool_registry,
            telemetry=self.telemetry,
            agent_id=slot_id,
            **self._kernel_kwargs,
        )
        slot = AgentSlot(
            slot_id=slot_id,
            role=role,
            capacity=capacity,
            kernel=kernel,
            memory=self._memory,
        )
        self._slots[slot_id] = slot
        logger.info(f"AgentDirector: added slot '{slot_id}' role={role}")
        return self

    async def stop(self) -> None:
        """Gracefully stop all slots."""
        logger.info("AgentDirector: stopped")

    # ── slot selection ────────────────────────────────────────────────────────

    def _pick_slot(self, urgency: str, excluded: Set[str]) -> Optional[str]:
        """
        Pick the best available slot locally (no A2A round-trip).
        Priority slots get preference on high/emergency urgency tasks.
        Falls back to least-loaded slot if all are at capacity.
        """
        high_prio  = urgency in ("high", "emergency")
        slot_order = {sid: i for i, sid in enumerate(self._slots)}

        candidates = [
            (sid, s) for sid, s in self._slots.items()
            if sid not in excluded and s.available
        ]
        if not candidates:
            # All full — pick least loaded as overflow
            candidates = [
                (sid, s) for sid, s in self._slots.items()
                if sid not in excluded
            ]
        if not candidates:
            return None

        def _key(item):
            sid, s = item
            role_boost = 0 if (high_prio and s.role in PRIORITY_ROLES) else 1
            rr = (slot_order.get(sid, 0) - self._rr_counter) % max(len(self._slots), 1)
            return (role_boost, s.load / max(s.capacity, 1), rr)

        return min(candidates, key=_key)[0]

    # ── task assignment ───────────────────────────────────────────────────────

    async def assign(
        self,
        event: str,
        urgency: str = "medium",
        confidence: float = 0.75,
        contested: bool = False,
        reroute_attempts: int = 0,
        forced_depth: Optional[str] = None,
        _excluded: Optional[Set[str]] = None,
        _hop: int = 0,
    ) -> Dict[str, Any]:
        """
        Assign a task to the best available agent and return the result.
        """
        if _hop > MAX_DELEGATE_HOPS:
            self._stats["tasks_failed"] += 1
            return {"status": "failed", "reason": "max delegation hops exceeded", "event": event}

        task_id  = uuid.uuid4().hex[:10]
        excluded = _excluded or set()
        context  = {
            "urgency": urgency, "confidence": confidence,
            "contested": contested, "reroute_attempts": reroute_attempts,
        }

        winner_id = self._pick_slot(urgency, excluded)
        if winner_id is None:
            self._stats["tasks_failed"] += 1
            return {"status": "failed", "reason": "no eligible agents", "task_id": task_id}

        self._rr_counter += 1
        slot = self._slots[winner_id]

        if slot._load > 0:
            self._stats["overflow_routes"] += 1

        if self.telemetry:
            self.telemetry.log_event("task_assigned", {
                "task_id":     task_id,
                "winner_id":   winner_id,
                "winner_role": slot.role,
                "winner_load": slot.load,
                "hop":         _hop,
                "event":       event[:100],
            })

        self._stats["tasks_assigned"] += 1
        slot._load += 1
        success = False

        try:
            if self._memory:
                self._memory.task_start(winner_id, task_id, event, context)

            result = await slot.kernel.handle(
                event,
                agent_role=slot.role,
                confidence=confidence,
                urgency=urgency,
                contested=contested,
                reroute_attempts=reroute_attempts,
                forced_depth=forced_depth,
                task_id=task_id,
            )
            success = True
            outcome = {"status": "success", "result": result}

        except Exception as exc:
            logger.warning(f"Slot '{winner_id}': task {task_id} failed: {exc}")
            outcome = {"status": "error", "error": str(exc)}

        finally:
            slot._load = max(0, slot._load - 1)
            if self._memory:
                agent_type = (
                    outcome.get("result", {}).get("agent_type", "unknown")
                    if success else "unknown"
                )
                decision = (
                    outcome.get("result", {}).get("decision", {})
                    if success else {}
                )
                action    = decision.get("action", "unknown") if isinstance(decision, dict) else "unknown"
                meta      = decision.get("metadata", {}) if isinstance(decision, dict) else {}
                plan_json = meta.get("plan_json")
                extra     = {"plan_json": plan_json} if plan_json else None
                self._memory.task_done(winner_id, agent_type, action, urgency, success, extra=extra)

        outcome["task_id"]     = task_id
        outcome["assigned_to"] = winner_id
        outcome["hops"]        = _hop
        return outcome

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "slot_loads":  {sid: s.load for sid, s in self._slots.items()},
            "bus_messages": self._bus.message_count,
            "memory":       self._memory.stats,
        }

    @property
    def pool_size(self) -> int:
        return len(self._slots)

    def get_suspended_tasks(self) -> Dict[str, str]:
        """Returns {agent_id: suspended_step_description} for all suspended agents."""
        suspended = {}
        for sid, slot in self._slots.items():
            node = slot.kernel.get_suspended_node()
            if node:
                suspended[sid] = node.description
        return suspended

    def approve_task(self, agent_id: str) -> bool:
        """Resumes a suspended agent's plan. Returns True if found."""
        if agent_id in self._slots:
            self._slots[agent_id].kernel.approve()
            logger.info(f"AgentDirector: approved task on '{agent_id}'")
            return True
        logger.warning(f"AgentDirector: cannot approve unknown agent '{agent_id}'")
        return False
