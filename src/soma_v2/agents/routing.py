"""
RoutingAgent — single LLM call for medium-complexity tasks.
============================================================
Makes one LLM call, then parses the response into a structured routing
decision: escalate / delegate / exhaust / reroute.

Also extracts a target unit from the LLM's reply or the original event text.
Falls back gracefully to "reroute" if LLM is unavailable or fails.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger("SOMA_V2.ROUTING")

# ── parsing regexes ───────────────────────────────────────────────────────────
_ESCALATE_RE = re.compile(
    r'\b(escalat|supervisor|emergency|critical|high.priority)\b', re.IGNORECASE)
_DELEGATE_RE = re.compile(
    r'\b(delegat|transfer|hand.?off|another.agent|reassign)\b',   re.IGNORECASE)
_EXHAUST_RE  = re.compile(
    r'\b(exhaust|try.all|all.option|retry.all|fallback.all)\b',   re.IGNORECASE)
_UNIT_RE     = re.compile(r'\b([A-Z]\d{1,3})\b')

_ROUTING_PROMPT = (
    "You are a routing agent for an autonomous drone swarm OS.\n"
    "Task: '{event}'\n"
    "Agent role: {role} | Urgency: {urgency}\n\n"
    "In ONE sentence, state: (1) the routing action — one of "
    "[reroute / escalate / delegate / exhaust], (2) the target unit ID "
    "if applicable, and (3) a brief reason."
)


def _parse_action(text: str) -> str:
    """Extract routing action from LLM response. Defaults to 'reroute'."""
    if _ESCALATE_RE.search(text):
        return "escalate"
    if _DELEGATE_RE.search(text):
        return "delegate"
    if _EXHAUST_RE.search(text):
        return "exhaust"
    return "reroute"


def _parse_unit(llm_text: str, event_text: str) -> Optional[str]:
    """Try to extract unit ID from LLM reply first, then from event."""
    m = _UNIT_RE.search(llm_text) or _UNIT_RE.search(event_text)
    return m.group(0) if m else None


class RoutingAgent:
    """
    Single-LLM-call routing for medium-depth (D2) tasks.

    The LLM provides a routing rationale which is parsed into a structured
    action label (escalate / delegate / exhaust / reroute) and optional
    target unit. Falls back to rule-based defaults on LLM failure.
    """

    def __init__(self, llm_callback=None) -> None:
        self.llm_callback = llm_callback

    async def handle(
        self,
        event:      str,
        agent_role: str,
        confidence: float,
        urgency:    str,
    ) -> Dict[str, Any]:
        logger.info(f"RoutingAgent: processing '{event[:60]}'")

        rationale   = "LLM unavailable — defaulting to reroute"
        llm_raw     = ""
        llm_success = False

        if self.llm_callback:
            try:
                prompt    = _ROUTING_PROMPT.format(
                    event=event[:200], role=agent_role, urgency=urgency
                )
                llm_raw     = await self.llm_callback("routing", prompt)
                rationale   = llm_raw.strip() or rationale
                llm_success = True
            except Exception as exc:
                logger.warning(f"RoutingAgent LLM call failed: {exc}")
                rationale = f"LLM error — defaulting to reroute"

        # Parse LLM output → structured action
        action      = _parse_action(rationale)
        target_unit = _parse_unit(llm_raw, event)

        logger.info(f"RoutingAgent: action={action} unit={target_unit} llm={llm_success}")

        result: Dict[str, Any] = {
            "action":     action,
            "confidence": confidence,
            "urgency":    urgency,
            "rationale":  rationale,
            "llm_used":   llm_success,
            "steps":      1,
        }
        if target_unit:
            result["target_unit"] = target_unit
        return result
