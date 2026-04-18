"""
SOMA V2 — AgentDirector
========================
Orchestrates a pool of named AgentSlot objects (each wrapping a V2Kernel).
Assigns incoming tasks via A2A bidding; handles delegation and overload.

Lifecycle of a task
-------------------
1.  Director receives task via `assign(event, ...)`.
2.  Director broadcasts TASK_BID to all slots.
3.  Each slot replies BID_RESPONSE: accept if load < capacity, with load score.
4.  Director picks winner (lowest load), sends TASK_CLAIM.
5.  Winner executes via its V2Kernel.handle().
6.  If winner is overloaded mid-execution, it sends TASK_DELEGATE.
7.  Director re-runs bidding excluding the delegating slot (max 2 hops).
8.  Final TASK_RESULT returned to caller.

Pool roles
----------
Slots are typed by agent_role (EMERGENCY, SUPERVISOR, PEER, ROUTINE).
EMERGENCY/SUPERVISOR slots always bid regardless of load — they override.
PEER/ROUTINE slots reject if current_load >= capacity.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from .a2a import A2ABus, A2AMessage, MsgType, ResourceBlackboard
from .negotiation import NegotiationBroker
from .kernel import V2Kernel
from ..memory.hierarchical import HierarchicalMemory

logger = logging.getLogger("SOMA_V2.DIRECTOR")

PRIORITY_ROLES = {"EMERGENCY", "SUPERVISOR"}
MAX_DELEGATE_HOPS = 2
BID_TIMEOUT   = 1.0   # seconds to wait for all bids
RESULT_TIMEOUT = 30.0  # seconds to wait for task result


# ── agent slot ────────────────────────────────────────────────────────────────

@dataclass
class AgentSlot:
    """
    One agent in the pool. Wraps a V2Kernel instance.
    Runs an async listener loop that processes bus messages.
    """
    slot_id:    str
    role:       str          # EMERGENCY | SUPERVISOR | PEER | ROUTINE
    capacity:   int = 4      # max concurrent tasks
    kernel:     Optional[V2Kernel] = field(default=None, repr=False)

    memory:     Optional[HierarchicalMemory] = field(default=None, repr=False)

    _load:      int               = field(default=0,    init=False, repr=False)
    _bus:       Optional[A2ABus]  = field(default=None, init=False, repr=False)
    _queue:     Optional[asyncio.Queue] = field(default=None, init=False, repr=False)
    _listener:  Optional[asyncio.Task]  = field(default=None, init=False, repr=False)
    _pending:   Dict[str, asyncio.Future] = field(default_factory=dict, init=False, repr=False)

    @property
    def load(self) -> int:
        return self._load

    @property
    def available(self) -> bool:
        return self._load < self.capacity

    def attach(self, bus: A2ABus) -> None:
        self._bus   = bus
        self._queue = bus.register(self.slot_id)

    async def start(self) -> None:
        self._listener = asyncio.create_task(self._listen(), name=f"slot-{self.slot_id}")

    async def stop(self) -> None:
        if self._listener:
            self._listener.cancel()

    async def _listen(self) -> None:
        while True:
            try:
                msg = await self._queue.get()
                asyncio.create_task(self._handle_msg(msg))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Slot {self.slot_id} listener error: {exc}")

    async def _handle_msg(self, msg: A2AMessage) -> None:
        if msg.msg_type == MsgType.TASK_BID:
            await self._on_bid(msg)
        elif msg.msg_type == MsgType.TASK_CLAIM:
            await self._on_claim(msg)

    async def _on_bid(self, msg: A2AMessage) -> None:
        urgency   = msg.payload.get("context", {}).get("urgency", "medium")
        high_prio = urgency in ("high", "emergency")
        # Priority roles override capacity only for high/emergency urgency
        accept = (self._load < self.capacity) or (high_prio and self.role in PRIORITY_ROLES)
        load_score = self._load / max(self.capacity, 1)
        response = A2AMessage(
            msg_type=MsgType.BID_RESPONSE,
            sender=self.slot_id,
            recipient=msg.sender,
            task_id=msg.task_id,
            payload={
                "accept":     accept,
                "load_score": round(load_score, 3),
                "role":       self.role,
            },
        )
        await self._bus.send(response)

    async def _on_claim(self, msg: A2AMessage) -> None:
        self._load += 1
        task_id = msg.task_id
        event   = msg.payload["event"]
        ctx     = msg.payload["context"]
        urgency = ctx.get("urgency", "medium")

        if self.memory:
            self.memory.task_start(self.slot_id, task_id, event, ctx)

        logger.info(f"Slot {self.slot_id}: executing task {task_id} (load={self._load})")
        success = False
        try:
            result = await self.kernel.handle(
                event,
                agent_role=self.role,
                confidence=ctx.get("confidence", 0.75),
                urgency=urgency,
                contested=ctx.get("contested", False),
                reroute_attempts=ctx.get("reroute_attempts", 0),
            )
            success = True
            outcome = {"status": "success", "result": result}
        except Exception as exc:
            logger.warning(f"Slot {self.slot_id}: task {task_id} failed: {exc}")
            outcome = {"status": "error", "error": str(exc)}
        finally:
            self._load = max(0, self._load - 1)
            if self.memory:
                agent_type = outcome.get("result", {}).get("agent_type", "unknown") if success else "unknown"
                decision   = outcome.get("result", {}).get("decision", {}) if success else {}
                action     = decision.get("action", "unknown")
                meta       = decision.get("metadata", {})
                extra      = {"plan_json": meta.get("plan_json")} if meta.get("plan_json") else None
                self.memory.task_done(self.slot_id, agent_type, action, urgency, success, extra=extra)

        await self._bus.send(A2AMessage(
            msg_type=MsgType.TASK_RESULT,
            sender=self.slot_id,
            recipient=msg.sender,
            task_id=task_id,
            payload=outcome,
        ))


# ── director ─────────────────────────────────────────────────────────────────

class AgentDirector:
    """
    Pool manager + A2A negotiation coordinator.

    Usage
    -----
    director = AgentDirector(llm_callback=my_llm)
    director.add_slot("peer_1", role="PEER")
    director.add_slot("supervisor_1", role="SUPERVISOR")
    await director.start()
    result = await director.assign("Traffic jam at node 7", urgency="high")
    await director.stop()
    """

    def __init__(self, llm_callback=None, memory: Optional[HierarchicalMemory] = None, actuator: Optional[Any] = None, **kernel_kwargs) -> None:
        self.llm_callback   = llm_callback
        self.actuator       = actuator
        self._kernel_kwargs = kernel_kwargs
        self._memory        = memory or HierarchicalMemory()
        self._bus:        A2ABus             = A2ABus()
        self._blackboard: ResourceBlackboard = ResourceBlackboard(bus=self._bus)
        self._negotiator: NegotiationBroker  = NegotiationBroker(blackboard=self._blackboard, bus=self._bus)
        self._slots: Dict[str, AgentSlot] = {}
        self._director_id = "director"
        self._dir_queue   = self._bus.register(self._director_id)
        self._stats: Dict[str, int]   = {"tasks_assigned": 0, "tasks_delegated": 0, "tasks_failed": 0}
        self._rr_counter: int         = 0

    def add_slot(
        self,
        slot_id: str,
        role: str = "PEER",
        capacity: int = 4,
    ) -> "AgentDirector":
        async def _delegate(sub_task: str, urgency: str = "medium") -> str:
            """Route a plan sub-task back through Director A2A — one hop max."""
            result = await self.assign(sub_task, urgency=urgency, _hop=MAX_DELEGATE_HOPS)
            decision = result.get("result", {}).get("decision", {})
            return decision.get("rationale", result.get("result", {}).get("depth", "delegated"))

        kernel = V2Kernel(
            llm_callback=self.llm_callback, memory=self._memory,
            actuator=self.actuator, delegate_fn=_delegate,
            blackboard=self._blackboard, agent_id=slot_id,
            negotiation_broker=self._negotiator,
            **self._kernel_kwargs,
        )
        slot   = AgentSlot(slot_id=slot_id, role=role, capacity=capacity, kernel=kernel, memory=self._memory)
        slot.attach(self._bus)
        self._slots[slot_id] = slot
        logger.info(f"AgentDirector: added slot '{slot_id}' role={role}")
        return self

    async def start(self) -> None:
        for slot in self._slots.values():
            await slot.start()
        logger.info(f"AgentDirector: started with {len(self._slots)} slots")

    async def stop(self) -> None:
        for slot in self._slots.values():
            await slot.stop()

    # ── public API ────────────────────────────────────────────────────────────

    def _pick_slot(self, urgency: str, excluded: Set[str]) -> Optional[str]:
        """
        Fast local slot selection — no message passing.
        Director reads slot loads directly; avoids A2A roundtrip for normal assigns.
        Priority roles get preference only for high/emergency urgency.
        """
        high_prio  = urgency in ("high", "emergency")
        slot_order = {sid: i for i, sid in enumerate(self._slots)}

        candidates = [
            (sid, s) for sid, s in self._slots.items()
            if sid not in excluded and s.available
        ]
        if not candidates:
            # all slots full — pick least loaded
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
        Assign a task to the best available agent.

        Fast path (default): director picks slot locally from known load state.
        A2A protocol is used only for delegation (slot signals it can't finish).
        """
        if _hop > MAX_DELEGATE_HOPS:
            self._stats["tasks_failed"] += 1
            return {"status": "failed", "reason": "max delegation hops exceeded", "event": event}

        task_id  = uuid.uuid4().hex[:10]
        excluded = _excluded or set()
        context  = {"urgency": urgency, "confidence": confidence,
                    "contested": contested, "reroute_attempts": reroute_attempts}

        winner_id = self._pick_slot(urgency, excluded)
        if winner_id is None:
            self._stats["tasks_failed"] += 1
            return {"status": "failed", "reason": "no eligible agents", "task_id": task_id}

        self._rr_counter += 1
        slot = self._slots[winner_id]
        logger.info(f"AgentDirector: task {task_id} -> '{winner_id}' load={slot.load} (hop={_hop})")

        # Execute directly on the slot — no queue hops for normal path
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
            )
            success = True
            outcome = {"status": "success", "result": result}
        except Exception as exc:
            logger.warning(f"Slot {winner_id}: task {task_id} failed: {exc}")
            outcome = {"status": "error", "error": str(exc)}
        finally:
            slot._load = max(0, slot._load - 1)
            if self._memory:
                agent_type = outcome.get("result", {}).get("agent_type", "unknown") if success else "unknown"
                decision   = outcome.get("result", {}).get("decision", {}) if success else {}
                action     = decision.get("action", "unknown") if isinstance(decision, dict) else "unknown"
                meta       = decision.get("metadata", {}) if isinstance(decision, dict) else {}
                plan_json  = meta.get("plan_json")
                extra      = {"plan_json": plan_json} if plan_json else None
                self._memory.task_done(winner_id, agent_type, action, urgency, success, extra=extra)

        outcome["task_id"]     = task_id
        outcome["assigned_to"] = winner_id
        outcome["hops"]        = _hop
        return outcome

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        slot_loads = {sid: s.load for sid, s in self._slots.items()}
        return {**self._stats, "slot_loads": slot_loads,
                "bus_messages": self._bus.message_count,
                "blackboard":  self._blackboard.stats,
                "negotiation": self._negotiator.stats,
                "memory":      self._memory.stats}

    @property
    def pool_size(self) -> int:
        return len(self._slots)
