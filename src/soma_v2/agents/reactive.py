"""
ReactiveAgent — D1 rule-based fast-path dispatcher.
====================================================
Zero LLM calls. Parses task text via regex keyword rules to produce
a structured action. This is the true D1 "simple" path.

Rule priority (first match wins):
  PING          → ping / heartbeat
  STATUS_CHECK  → check / status / health / online / offline
  REBOOT        → reboot / restart / reset
  RETURN_TO_BASE→ land / return / recall / rtb
  ROUTE         → route / navigate / move / go to / dispatch to
  REPORT        → report / log / record / audit
  OBSERVE       → (default fallback)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger("SOMA_V2.REACTIVE")

# ── regex rule table ──────────────────────────────────────────────────────────
# (compiled_pattern, action_label)  — first match wins
_RULES = [
    (re.compile(r'\b(ping|heartbeat)\b',                        re.IGNORECASE), "PING"),
    (re.compile(r'\b(check|status|health|online|offline)\b',    re.IGNORECASE), "STATUS_CHECK"),
    (re.compile(r'\b(reboot|restart|reset)\b',                  re.IGNORECASE), "REBOOT"),
    (re.compile(r'\b(land|return|recall|rtb)\b',                re.IGNORECASE), "RETURN_TO_BASE"),
    (re.compile(r'\b(route|navigate|move|go\s+to|dispatch\s+to)\b', re.IGNORECASE), "ROUTE"),
    (re.compile(r'\b(report|log|record|audit)\b',               re.IGNORECASE), "REPORT"),
]

# Matches unit IDs like A1, B12, C3, "node 7", "sensor X2"
_UNIT_RE = re.compile(
    r'\b([A-Z]\d{1,3}|node\s*\d+|sensor\s+\w+)\b',
    re.IGNORECASE,
)

# Extracts destination after "to / at / toward"
_DEST_RE = re.compile(
    r'\b(?:to|at|toward)\s+([\w\s]+?)(?:\s*[,.\n]|$)',
    re.IGNORECASE,
)


def _extract_unit(text: str) -> str:
    m = _UNIT_RE.search(text)
    return m.group(0).upper() if m else "UNKNOWN"


def _extract_destination(text: str) -> Optional[str]:
    m = _DEST_RE.search(text)
    return m.group(1).strip() if m else None


class ReactiveAgent:
    """
    Rule-based dispatcher for D1 (simple) tasks.

    Produces a structured {action, unit, destination?} dict from the raw
    task text without making any LLM calls. Latency is sub-millisecond.
    """

    async def handle(
        self,
        event:      str,
        agent_role: str,
        confidence: float,
        urgency:    str,
    ) -> Dict[str, Any]:
        # Match first rule
        action = "OBSERVE"
        for pattern, label in _RULES:
            if pattern.search(event):
                action = label
                break

        unit = _extract_unit(event)
        dest = _extract_destination(event) if action == "ROUTE" else None

        logger.info(f"ReactiveAgent: action={action} unit={unit} role={agent_role}")

        result: Dict[str, Any] = {
            "action":    action,
            "unit":      unit,
            "confidence": confidence,
            "urgency":   urgency,
            "rationale": f"D1 rule-match: {action} on {unit}",
            "steps":     1,
        }
        if dest:
            result["destination"] = dest
        return result
