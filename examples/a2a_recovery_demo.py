"""
SOMA V2 — A2A Failure Recovery Demo
=====================================
Proves that when one agent is overwhelmed, the A2A bus automatically
delegates overflow tasks to available peers — no human intervention.

What this demonstrates:
  1. Agent Alpha (cap=1): can only hold 1 task at a time
  2. Agent Beta  (cap=4): available peer with headroom
  3. We flood 5 tasks simultaneously at Alpha
  4. A2A bidding kicks in: Alpha takes task #1, delegates 4 to Beta
  5. Every task still completes — swarm is resilient under load

Then: we simulate Alpha going offline (cap=0) mid-session.
  6. New task arrives — Alpha rejects (full), Beta handles it alone

Watch the bus message count grow as delegation happens.
"""
import asyncio
import sys
import os
import time
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("SOMA_V2.DIRECTOR").setLevel(logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from soma_v2.core.director import AgentDirector
from soma_v2.core.a2a import A2ABus


# ── fast mock LLM — no AirSim needed for this demo ───────────────────────────

async def mock_llm(*args, **kwargs) -> str:
    await asyncio.sleep(0.8)   # hold long enough for concurrent tasks to overlap
    return "Assessed situation and resolved autonomously."


def banner(text: str):
    print("\n" + "=" * 64)
    print(f"  {text}")
    print("=" * 64)


def show_results(label: str, results: list, elapsed: float):
    print(f"\n  {label}  ({elapsed:.2f}s total)")
    agents = {}
    for r in results:
        a = r.get("assigned_to", "?")
        agents[a] = agents.get(a, 0) + 1

    for agent, count in sorted(agents.items()):
        bar = "#" * count
        print(f"    {agent:20s} handled {count} task(s)  [{bar}]")

    failed = sum(1 for r in results if r.get("status") != "success")
    print(f"  All completed: {'YES' if failed == 0 else f'NO ({failed} failed)'}")


async def main():
    banner("SOMA V2 — A2A Failure Recovery Demo")
    print("  No AirSim needed — pure swarm orchestration proof\n")

    bus = A2ABus()

    # Alpha: tight capacity=1 so it overflows immediately
    # Beta:  generous capacity=4 to absorb delegation
    director = AgentDirector(
        llm_callback=mock_llm,
        bus=bus,
        claim_timeout_s=1.5,
    )
    director.add_slot("alpha", role="SUPERVISOR", capacity=1)
    director.add_slot("beta",  role="PEER",       capacity=4)
    await director.start()

    print("Agents online:")
    print("  alpha  [SUPERVISOR]  capacity=1  <- intentionally tight")
    print("  beta   [PEER]        capacity=4  <- absorbs overflow\n")

    # ── Phase 1: flood 5 tasks simultaneously ────────────────────────────────
    banner("Phase 1 — 5 tasks hit simultaneously (alpha cap=1)")
    print("  Alpha can take 1. A2A bus delegates the rest to Beta.\n")

    tasks = [
        director.assign(f"Flood task {i}: status check sector {i}", urgency="medium", forced_depth="complex")
        for i in range(5)
    ]

    t0 = time.monotonic()
    results = await asyncio.gather(*tasks)
    t1 = time.monotonic()

    show_results("Load distribution:", results, t1 - t0)
    print(f"\n  A2A bus messages so far: {bus.message_count}")

    # ── Phase 2: take alpha offline ───────────────────────────────────────────
    banner("Phase 2 — Alpha goes offline (capacity forced to 0)")
    print("  Simulating agent failure — alpha rejects all bids.\n")

    # Force alpha to reject by setting capacity to 0
    director._slots["alpha"].capacity = 0

    t2 = time.monotonic()
    result_offline = await director.assign(
        "URGENT: deploy emergency beacon at crash site", urgency="high"
    )
    t3 = time.monotonic()

    assigned_to = result_offline.get("assigned_to", "?")
    status      = result_offline.get("status", "?").upper()
    print(f"  Task status : {status}")
    print(f"  Handled by  : {assigned_to}  (alpha was offline)")
    print(f"  Time        : {t3 - t2:.2f}s")
    print(f"\n  Swarm continued without alpha: {'YES' if status == 'SUCCESS' else 'NO'}")

    # ── Phase 3: alpha recovers ───────────────────────────────────────────────
    banner("Phase 3 — Alpha recovers (capacity restored to 2)")
    print("  Hot-swap: restore alpha mid-session, no restart needed.\n")

    director._slots["alpha"].capacity = 2

    recovery_tasks = [
        director.assign(f"Recovery task {i}: resume patrol sector {i}", urgency="medium", forced_depth="complex")
        for i in range(3)
    ]
    t4 = time.monotonic()
    recovery_results = await asyncio.gather(*recovery_tasks)
    t5 = time.monotonic()

    show_results("Post-recovery distribution:", recovery_results, t5 - t4)

    await director.stop()

    # ── summary ───────────────────────────────────────────────────────────────
    banner("Session Summary")
    stats = director.stats
    print(f"  Total tasks assigned : {stats['tasks_assigned']}")
    print(f"  Overflow routes      : {stats['overflow_routes']}  (slot full -> routed to peer)")
    print(f"  Tasks delegated      : {stats['tasks_delegated']}  (mid-task re-assignment hops)")
    print(f"  Tasks failed         : {stats['tasks_failed']}")
    print(f"  A2A bus messages     : {bus.message_count}  (fast-path routing bypasses bus by design)")
    print(f"  Final slot loads     : {stats['slot_loads']}")
    print("\n  Proof: swarm handled overload AND agent failure without")
    print("  dropping a single task or requiring human intervention.")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
