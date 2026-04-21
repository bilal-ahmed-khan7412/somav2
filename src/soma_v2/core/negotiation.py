"""
SOMA V2 — Inter-Agent Step Negotiation
========================================
When an agent fails to claim a physical unit (blackboard conflict), instead of
blocking it proposes the step to whichever agent currently owns the unit.

Protocol (all messages logged on the shared A2A bus):
  1. agent_i tries to claim unit X → fails (agent_j owns it)
  2. agent_i → NegotiationBroker.propose(step, unit_id)
  3. Broker finds agent_j (owner of X) via ResourceBlackboard
  4. Broker sends STEP_PROPOSAL to agent_j on the A2A bus (observability)
  5. agent_j executes the step using its existing claim (reentrant blackboard)
  6. Broker sends STEP_ACCEPT + result back to agent_i
  7. agent_i marks its node [NEGOTIATED→agent_j] and continues

If agent_j is overloaded (concurrent_steps >= max_load):
  Broker sends STEP_REJECT — agent_i falls back to waiting (existing path).

The broker is a single shared instance held by AgentDirector. Each PlanExecutor
self-registers on construction so the broker can find and invoke it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .a2a import A2ABus, ResourceBlackboard
    from .planner import PlanExecutor

from .a2a import A2AMessage, MsgType

logger = logging.getLogger("SOMA_V2.NEGOTIATION")

MAX_CONCURRENT_NEGOTIATED = 2   # max extra steps an agent accepts via negotiation


@dataclass
class NegotiationResult:
    accepted:  bool
    result:    str
    by_agent:  str
    latency_ms: float


class NegotiationBroker:
    """
    Single shared broker for inter-agent step negotiation.

    AgentDirector creates one instance and passes it to every kernel.
    PlanExecutors self-register on construction.
    """

    def __init__(self, blackboard: "ResourceBlackboard", bus: "A2ABus") -> None:
        self._blackboard  = blackboard
        self._bus         = bus
        self._executors:  Dict[str, "PlanExecutor"] = {}
        self._negotiations: int = 0
        self._accepted:     int = 0
        self._rejected:     int = 0

    def register(self, agent_id: str, executor: "PlanExecutor") -> None:
        self._executors[agent_id] = executor
        logger.debug(f"NegotiationBroker: registered executor for '{agent_id}'")

    async def propose(
        self,
        from_agent: str,
        step_desc:  str,
        unit_id:    str,
        context:    str = "",
        timeout_s:  float = 3.0,
    ) -> NegotiationResult:
        """
        Propose that another agent executes `step_desc` on behalf of `from_agent`.

        Targets the agent that currently owns `unit_id` on the blackboard —
        that agent can execute reentrant commands on the unit without waiting.
        Returns a NegotiationResult regardless of outcome.
        """
        self._negotiations += 1
        t0 = time.perf_counter()

        # Find the agent that owns the unit right now
        target_id = self._blackboard._locks.get(unit_id)
        if not target_id or target_id not in self._executors:
            self._rejected += 1
            logger.debug(
                f"NegotiationBroker: no eligible target for unit '{unit_id}' "
                f"(owner={target_id})"
            )
            return NegotiationResult(
                accepted=False,
                result=f"no eligible agent owns '{unit_id}'",
                by_agent="none",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        target_exec = self._executors[target_id]

        # Reject if target is already handling too many negotiated steps
        if target_exec._negotiated_load >= MAX_CONCURRENT_NEGOTIATED:
            self._rejected += 1
            logger.info(
                f"NegotiationBroker: {from_agent} → {target_id} REJECTED "
                f"(negotiated_load={target_exec._negotiated_load})"
            )
            await self._bus.send(A2AMessage(
                msg_type=MsgType.STEP_REJECT,
                sender=target_id, recipient=from_agent,
                task_id=unit_id,
                payload={"reason": "overloaded", "load": target_exec._negotiated_load},
            ))
            return NegotiationResult(
                accepted=False,
                result="target overloaded",
                by_agent=target_id,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Log proposal on A2A bus
        await self._bus.send(A2AMessage(
            msg_type=MsgType.STEP_PROPOSAL,
            sender=from_agent, recipient=target_id,
            task_id=unit_id,
            payload={"step_desc": step_desc[:120], "unit_id": unit_id},
        ))
        logger.info(
            f"NegotiationBroker: {from_agent} → {target_id} PROPOSE "
            f"unit='{unit_id}' step='{step_desc[:60]}'"
        )

        # Ask target executor to run the step inline
        try:
            result = await asyncio.wait_for(
                target_exec._execute_negotiated_step(step_desc, context),
                timeout=timeout_s,
            )
            self._accepted += 1
            await self._bus.send(A2AMessage(
                msg_type=MsgType.STEP_ACCEPT,
                sender=target_id, recipient=from_agent,
                task_id=unit_id,
                payload={"result": result[:200]},
            ))
            logger.info(
                f"NegotiationBroker: {target_id} ACCEPT → result='{result[:60]}'"
            )
            return NegotiationResult(
                accepted=True,
                result=result,
                by_agent=target_id,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            self._rejected += 1
            await self._bus.send(A2AMessage(
                msg_type=MsgType.STEP_REJECT,
                sender=target_id, recipient=from_agent,
                task_id=unit_id,
                payload={"reason": str(exc)},
            ))
            logger.warning(
                f"NegotiationBroker: {target_id} REJECT (error) — {exc}"
            )
            return NegotiationResult(
                accepted=False,
                result=str(exc),
                by_agent=target_id,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "negotiations": self._negotiations,
            "accepted":     self._accepted,
            "rejected":     self._rejected,
            "accept_rate":  (
                f"{self._accepted / self._negotiations * 100:.0f}%"
                if self._negotiations else "n/a"
            ),
        }
