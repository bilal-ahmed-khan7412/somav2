"""
SOMA V2 — Quickstart Example
==============================
Demonstrates how to use SOMASwarm with the OpenAI API.

Requirements:
  export OPENAI_API_KEY="sk-..."
  pip install -e .

Then run:
  python example_quickstart.py
"""

import asyncio
import logging
import os
import sys

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from soma_v2 import SOMASwarm
from soma_v2.core.tools import RiskLevel


# ── optional: register a custom tool ─────────────────────────────────────────

async def get_current_time() -> str:
    """Example tool: returns the current UTC timestamp."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ── main demo ─────────────────────────────────────────────────────────────────

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[ERROR] OPENAI_API_KEY is not set.")
        print("  export OPENAI_API_KEY='sk-...'  and re-run.\n")
        sys.exit(1)

    print("\n" + "="*60)
    print("  SOMA V2 — OpenAI Multi-Agent Quickstart")
    print("="*60 + "\n")

    # Initialise the swarm
    swarm = SOMASwarm(
        model="openai/gpt-4o-mini",
        slots=2,
        trace_dir=None,              # disable trace files for the demo
        persist_dir="./soma_memory", # ChromaDB cold memory location
    )

    # Register a custom tool
    swarm.register_tool(
        name="get_current_time",
        func=get_current_time,
        description="Get the current UTC time",
        risk=RiskLevel.LOW,
    )

    # ── Task 1: simple status check (handled by ReactiveAgent, no LLM call) ──
    print("→ Task 1 (simple): status check")
    result1 = await swarm.dispatch("Check status of node A5", urgency="low")
    print(f"  depth={result1.get('result', {}).get('depth', '?')} "
          f"agent={result1.get('result', {}).get('agent_type', '?')}")
    print(f"  decision: {result1.get('result', {}).get('decision', {})}\n")

    # ── Task 2: medium routing task (handled by RoutingAgent, 1 LLM call) ────
    print("→ Task 2 (medium): routing decision")
    result2 = await swarm.dispatch(
        "Node B3 is reporting intermittent packet loss. What should we do?",
        urgency="medium",
    )
    print(f"  depth={result2.get('result', {}).get('depth', '?')} "
          f"agent={result2.get('result', {}).get('agent_type', '?')}")
    print(f"  rationale: {result2.get('result', {}).get('decision', {}).get('rationale', '')[:120]}\n")

    # ── Task 3: complex planning task (handled by DeliberativeAgent) ──────────
    print("→ Task 3 (complex): multi-step plan")
    result3 = await swarm.dispatch(
        "Coordinate a full diagnostic sweep across all network nodes, "
        "identify bottlenecks, and produce a remediation report.",
        urgency="high",
        forced_depth="complex",
    )
    r3 = result3.get("result", {})
    decision = r3.get("decision", {})
    print(f"  depth={r3.get('depth', '?')} agent={r3.get('agent_type', '?')}")
    print(f"  steps_done: {decision.get('steps', '?')}")
    print(f"  plan_summary: {decision.get('plan_summary', '')}\n")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print("─" * 60)
    print("Swarm stats:")
    stats = swarm.stats
    print(f"  tasks_assigned : {stats['director']['tasks_assigned']}")
    print(f"  hot cache hits : {stats['memory']['hot']['hits']}")
    print(f"  bus messages   : {stats['bus_messages']}")

    await swarm.close()
    print("\nDone. ✓\n")


if __name__ == "__main__":
    asyncio.run(main())
