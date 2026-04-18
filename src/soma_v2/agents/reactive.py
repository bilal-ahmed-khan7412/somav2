"""
ReactiveAgent — handles simple tasks via rule-based fast path.
Zero LLM calls. Equivalent to V1 D1 rule-router.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger("SOMA_V2.REACTIVE")


class ReactiveAgent:
    async def handle(
        self,
        event: str,
        agent_role: str,
        confidence: float,
        urgency: str,
    ) -> Dict[str, Any]:
        logger.info(f"ReactiveAgent: rule-routing '{event[:50]}'")
        return {
            "action":     "reroute",
            "confidence": confidence,
            "urgency":    urgency,
            "rationale":  "D1 rule-route: uncontested simple task",
            "steps":      1,
        }
