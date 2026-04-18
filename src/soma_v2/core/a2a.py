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
    TASK_BID      = "TASK_BID"       # director → agents: "who can handle this?"
    BID_RESPONSE  = "BID_RESPONSE"   # agent → director: accept/reject + load
    TASK_CLAIM    = "TASK_CLAIM"     # director → winner: "you own this task"
    TASK_DELEGATE = "TASK_DELEGATE"  # agent → director: "I can't finish, reassign"
    TASK_RESULT   = "TASK_RESULT"    # agent → director: final outcome


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

    def __init__(self) -> None:
        self._queues:   Dict[str, asyncio.Queue] = {}
        self._handlers: Dict[str, Callable]      = {}
        self._history:  List[A2AMessage]         = []

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
