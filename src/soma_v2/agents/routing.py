"""
RoutingAgent — handles medium tasks via single LLM call.
Equivalent to V1 LLM routing path.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("SOMA_V2.ROUTING")


class RoutingAgent:
    def __init__(self, llm_callback=None):
        self.llm_callback = llm_callback

    async def handle(
        self,
        event: str,
        agent_role: str,
        confidence: float,
        urgency: str,
    ) -> Dict[str, Any]:
        logger.info(f"RoutingAgent: LLM routing '{event[:50]}'")

        rationale = "LLM routing decision"
        if self.llm_callback:
            try:
                prompt = (
                    f"You are a routing agent. Given this event: '{event}'\n"
                    f"Agent role: {agent_role}, Urgency: {urgency}\n"
                    f"Decide: should we reroute, escalate, or exhaust options? "
                    f"Reply in one sentence."
                )
                rationale = await self.llm_callback("routing", prompt)
            except Exception as exc:
                logger.warning(f"RoutingAgent LLM call failed: {exc}")

        return {
            "action":     "reroute",
            "confidence": confidence,
            "urgency":    urgency,
            "rationale":  rationale,
            "steps":      1,
        }
