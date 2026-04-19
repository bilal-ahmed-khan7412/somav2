import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

logger = logging.getLogger("SOMA_V2.BROKER")

@dataclass
class NegotiationResult:
    accepted: bool
    by_agent: Optional[str] = None
    result:   Optional[str] = None
    latency_ms: float = 0.0

class NegotiationBroker:
    """
    Enables peer-to-peer task negotiation.
    If Agent A needs a resource held by Agent B, Agent A can propose
    the step to Agent B directly through this broker.
    """
    
    def __init__(self, blackboard: Any = None, bus: Any = None) -> None:
        self._blackboard = blackboard
        self._bus = bus
        self._agents: Dict[str, Any] = {} # agent_id -> PlanExecutor
        self._negotiations_total = 0
        self._negotiations_success = 0

    @property
    def stats(self) -> dict:
        return {
            "total_attempts": self._negotiations_total,
            "success_rate": round(self._negotiations_success / max(self._negotiations_total, 1), 2),
            "registered_agents": len(self._agents)
        }

    def register(self, agent_id: str, executor: Any) -> None:
        self._agents[agent_id] = executor
        logger.debug(f"Broker: Registered agent '{agent_id}'")

    async def propose(self, from_agent: str, step_desc: str, unit_id: str, context: str) -> NegotiationResult:
        """
        Propose a plan step to the current owner of a unit.
        """
        t0 = time.perf_counter()
        
        # In a real distributed system, we would query the Blackboard to find the owner.
        # For now, we broadcast to any registered agent that isn't the sender.
        # If an agent is 'near' the unit or has lower load, they might accept.
        
        self._negotiations_total += 1
        
        # Find the blackboard owner of the contested unit first; fall back to any agent
        # with spare negotiated-step capacity (< 3 concurrent negotiated steps).
        owner = None
        if self._blackboard:
            owner = self._blackboard._locks.get(unit_id)

        target_agent = None
        # Prefer the actual owner — they already hold the lock and can act
        if owner and owner != from_agent and owner in self._agents:
            if getattr(self._agents[owner], "_negotiated_load", 0) < 3:
                target_agent = owner

        # Fallback: any other registered agent with spare capacity
        if not target_agent:
            for aid, exec_obj in self._agents.items():
                if aid != from_agent and aid != owner:
                    if getattr(exec_obj, "_negotiated_load", 0) < 3:
                        target_agent = aid
                        break
        
        if not target_agent:
            return NegotiationResult(accepted=False)

        logger.info(f"Broker: Negotiating step '{step_desc[:40]}' from '{from_agent}' to '{target_agent}'")

        executor = self._agents[target_agent]
        # Fire-and-forget: caller doesn't block waiting for step completion.
        # Requesting agent continues immediately; step executes in background.
        asyncio.create_task(self._run_step(executor, target_agent, step_desc, context))

        self._negotiations_success += 1
        latency = (time.perf_counter() - t0) * 1000
        return NegotiationResult(
            accepted=True,
            by_agent=target_agent,
            latency_ms=round(latency, 2),
        )

    async def _run_step(self, executor: Any, target_agent: str, step_desc: str, context: str) -> None:
        try:
            await executor._execute_negotiated_step(step_desc, context)
        except Exception as exc:
            logger.warning(f"Broker: background step failed for '{target_agent}': {exc}")
