"""
Negotiation integration test.

Two agents both need unit A12.
agent_1 holds A12 for 2 seconds (slow step).
agent_0 times out after 0.3s and triggers the negotiation broker.
Expected: agent_0's node output shows [NEGOTIATED→agent_1].
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.soma_v2.core.director import AgentDirector
from src.soma_v2.core.actuators import MockDroneActuator


class SlowActuator(MockDroneActuator):
    """Holds the unit claim for 2s to force a negotiation on agent_0."""
    async def execute_command(self, cmd: str) -> bool:
        if "A12" in cmd:
            await asyncio.sleep(2.0)   # hold the lock long enough
        return await super().execute_command(cmd)


async def main():
    print("=== SOMA V2 Negotiation Test ===\n")

    director = AgentDirector(
        llm_callback=None,   # stub mode — no Ollama needed
        actuator=SlowActuator(),
    )

    # Both agents get a short claim timeout so negotiation fires quickly
    director.add_slot("agent_0", role="PEER", capacity=4)
    director.add_slot("agent_1", role="PEER", capacity=4)

    # Patch both executors with a short claim_timeout_s
    for slot in director._slots.values():
        slot.kernel.deliberative._executor.claim_timeout_s = 0.3

    await director.start()

    # Task that forces both agents to target A12 concurrently
    task_a = director.assign(
        "Deploy unit A12 to sector 7 — navigate A12 to the target zone",
        urgency="high", forced_depth="complex",
    )
    task_b = director.assign(
        "Takeoff A12 and scan A12 area for threats — launch A12 immediately",
        urgency="high", forced_depth="complex",
    )

    results = await asyncio.gather(task_a, task_b, return_exceptions=True)

    print("\n--- Results ---")
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"Task {i}: EXCEPTION — {r}")
            continue
        res = r.get("result", {})
        detail = res.get("plan_detail", {})
        print(f"\nTask {i} (assigned_to={r.get('assigned_to')}):")
        for nid, nd in detail.items():
            out = (nd.get("output") or "")[:120]
            print(f"  [{nd['status']:7s}] {nid}: {out}")

    print("\n--- Director Stats ---")
    stats = director.stats
    print(f"  negotiation: {stats['negotiation']}")
    print(f"  blackboard:  {stats['blackboard']}")

    await director.stop()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
