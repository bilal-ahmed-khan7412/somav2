"""
SOMA V2 — Agent-to-Agent (A2A) Protocol
=========================================
Lightweight in-process message bus for inter-agent negotiation.

Message flow
------------
1. Director posts TASK_BID to all eligible agents.
2. Each agent replies BID_RESPONSE (accept / reject + load score).
3. Director selects lowest-load acceptor, posts TASK_CLAIM.
4. If the assigned agent cannot complete the task, it posts TASK_DELEGATE
   back to the bus; Director re-assigns.
5. Final result posted as TASK_RESULT by the executing agent.

All messages are routed by (sender, recipient) agent_id strings.
The bus is in-process (asyncio); no external broker required.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("SOMA_V2.A2A")


# ── message types ─────────────────────────────────────────────────────────────

class MsgType(str, Enum):
    TASK_BID        = "TASK_BID"        # director → agents: "who can handle this?"
    BID_RESPONSE    = "BID_RESPONSE"    # agent → director: accept/reject + load
    TASK_CLAIM      = "TASK_CLAIM"      # director → winner: "you own this task"
    TASK_DELEGATE   = "TASK_DELEGATE"   # agent → director: "I can't finish, reassign"
    TASK_RESULT     = "TASK_RESULT"     # agent → director: final outcome
    RESOURCE_CLAIM   = "RESOURCE_CLAIM"   # agent → bus: "I am using unit X"
    RESOURCE_RELEASE = "RESOURCE_RELEASE" # agent → bus: "I am done with unit X"
    STEP_PROPOSAL    = "STEP_PROPOSAL"    # agent_i → agent_j: "execute this step for me"
    STEP_ACCEPT      = "STEP_ACCEPT"      # agent_j → agent_i: "done, here is the result"
    STEP_REJECT      = "STEP_REJECT"      # agent_j → agent_i: "overloaded, can't take it"


# ── message ──────────────────────────────────────────────────────────────────

@dataclass
class A2AMessage:
    msg_type:  MsgType
    sender:    str
    recipient: str                    # "*" = broadcast
    task_id:   str
    payload:   Dict[str, Any]         = field(default_factory=dict)
    timestamp: float                  = field(default_factory=time.time)
    msg_id:    str                    = field(default_factory=lambda: uuid.uuid4().hex[:8])


# ── bus ───────────────────────────────────────────────────────────────────────

class A2ABus:
    """
    In-process pub/sub bus.
    Agents register a queue; the bus delivers by agent_id or broadcast ("*").
    """

    def __init__(self, telemetry: Optional[Any] = None) -> None:
        self._queues:   Dict[str, asyncio.Queue] = {}
        self._handlers: Dict[str, Callable]      = {}
        self._history:  List[A2AMessage]         = []
        self._telemetry = telemetry

    def register(self, agent_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self._queues[agent_id] = q
        logger.debug(f"A2ABus: registered '{agent_id}'")
        return q

    def unregister(self, agent_id: str) -> None:
        self._queues.pop(agent_id, None)

    async def send(self, msg: A2AMessage) -> None:
        self._history.append(msg)
        logger.debug(f"A2ABus: {msg.sender} -> {msg.recipient} [{msg.msg_type}] task={msg.task_id}")

        if self._telemetry:
            self._telemetry.log_event("a2a_msg", {
                "msg_type": msg.msg_type,
                "sender": msg.sender,
                "recipient": msg.recipient,
                "msg_task_id": msg.task_id, # call it msg_task_id to avoid confusion with the tracing task_id
                **msg.payload
            })

        if msg.recipient == "*":
            for q in self._queues.values():
                await q.put(msg)
        elif msg.recipient in self._queues:
            await self._queues[msg.recipient].put(msg)
        else:
            logger.warning(f"A2ABus: unknown recipient '{msg.recipient}'")

    async def recv(self, agent_id: str, timeout: float = 5.0) -> Optional[A2AMessage]:
        q = self._queues.get(agent_id)
        if q is None:
            return None
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    @property
    def message_count(self) -> int:
        return len(self._history)

    def history(self, task_id: Optional[str] = None) -> List[A2AMessage]:
        if task_id is None:
            return list(self._history)
        return [m for m in self._history if m.task_id == task_id]


# ── resource blackboard ───────────────────────────────────────────────────────

class ResourceBlackboard:
    """
    Shared blackboard for physical resource ownership during plan execution.

    Agents claim a unit (drone/sensor/node) before issuing a [CMD] and release
    it when the step completes. Concurrent agents trying to claim the same unit
    wait (up to `timeout_s`) then backtrack if the unit stays busy.

    All claim/release events are published to the A2A bus for observability —
    the same bus used for task bidding, keeping the coordination model unified.

    Design
    ------
    - One asyncio.Event per unit: set when free, cleared when owned.
    - Claims are exclusive: one owner at a time.
    - Reentrant: an agent can re-claim a unit it already owns (depth counter).
    - `timeout_s=0` disables waiting — claim fails immediately on conflict.
    """

    CLAIM_TIMEOUT_S: float = 5.0

    def __init__(self, bus: Optional[A2ABus] = None) -> None:
        self._bus:     Optional[A2ABus]              = bus
        self._owners:  Dict[str, str]                = {}   # unit_id → agent_id
        self._depth:   Dict[str, int]                = {}   # reentrant depth
        self._events:  Dict[str, asyncio.Event]      = {}   # unit_id → free-event
        self._lock:    asyncio.Lock                  = asyncio.Lock()
        self._claims:  int                           = 0
        self._conflicts: int                         = 0

    def _event(self, unit_id: str) -> asyncio.Event:
        if unit_id not in self._events:
            ev = asyncio.Event()
            ev.set()   # free by default
            self._events[unit_id] = ev
        return self._events[unit_id]

    async def claim(
        self,
        unit_id:  str,
        agent_id: str,
        node_id:  str  = "?",
        timeout_s: float = CLAIM_TIMEOUT_S,
    ) -> bool:
        """
        Claim exclusive ownership of `unit_id` for `agent_id`.
        Returns True on success, False on timeout (conflict unresolved).
        """
        ev = self._event(unit_id)

        # Fast path: already owner (reentrant)
        async with self._lock:
            if self._owners.get(unit_id) == agent_id:
                self._depth[unit_id] = self._depth.get(unit_id, 1) + 1
                return True

        # Wait for unit to become free
        if not ev.is_set():
            self._conflicts += 1
            logger.debug(
                f"ResourceBlackboard: {agent_id}/{node_id} waiting for unit '{unit_id}' "
                f"(owned by {self._owners.get(unit_id, '?')})"
            )
            try:
                await asyncio.wait_for(ev.wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                logger.warning(
                    f"ResourceBlackboard: {agent_id}/{node_id} TIMEOUT waiting for '{unit_id}'"
                )
                return False

        async with self._lock:
            # Re-check after wait — another task may have grabbed it
            if unit_id in self._owners:
                if self._owners[unit_id] != agent_id:
                    return False
            self._owners[unit_id] = agent_id
            self._depth[unit_id]  = 1
            ev.clear()   # mark busy
            self._claims += 1

        logger.info(
            f"ResourceBlackboard: '{unit_id}' CLAIMED by {agent_id}/{node_id}"
        )
        if self._bus:
            await self._bus.send(A2AMessage(
                msg_type=MsgType.RESOURCE_CLAIM,
                sender=agent_id, recipient="*",
                task_id=node_id,
                payload={"unit_id": unit_id, "node_id": node_id},
            ))
        return True

    async def release(self, unit_id: str, agent_id: str, node_id: str = "?") -> None:
        """Release ownership of `unit_id`. Safe to call even if not owner."""
        async with self._lock:
            if self._owners.get(unit_id) != agent_id:
                return
            depth = self._depth.get(unit_id, 1) - 1
            if depth > 0:
                self._depth[unit_id] = depth
                return
            del self._owners[unit_id]
            self._depth.pop(unit_id, None)
            ev = self._events.get(unit_id)
            if ev:
                ev.set()   # signal free

        logger.info(f"ResourceBlackboard: '{unit_id}' RELEASED by {agent_id}/{node_id}")
        if self._bus:
            await self._bus.send(A2AMessage(
                msg_type=MsgType.RESOURCE_RELEASE,
                sender=agent_id, recipient="*",
                task_id=node_id,
                payload={"unit_id": unit_id, "node_id": node_id},
            ))

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "active_claims": dict(self._owners),
            "total_claims":  self._claims,
            "conflicts":     self._conflicts,
        }
