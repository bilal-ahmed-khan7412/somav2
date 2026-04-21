"""
SOMA V2 — Live Ollama Demo
==========================
Proves the deliberative tier with a real LLM (qwen2.5:3b via Ollama).

What this demonstrates:
  1. SOMASwarm high-level API (no director/slot setup needed)
  2. Real LLM generates the plan JSON from plain-English task
  3. Command injection maps LLM output -> [CMD] tokens -> AirSim
  4. Memory: second identical task hits L1 hot cache, skips LLM
  5. Different task: LLM called again, new plan generated

Run AFTER Blocks.exe is running and Ollama is serving.
"""
import asyncio
import sys
import os
import time
import logging

# Show deliberative planner logs so we can see cache hits vs LLM calls
logging.basicConfig(level=logging.WARNING)
logging.getLogger("SOMA_V2.DELIBERATIVE").setLevel(logging.DEBUG)
logging.getLogger("SOMA_V2.ACTUATOR").setLevel(logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from soma_v2.main import SOMASwarm
from soma_v2.core.actuators import AirSimActuator


def banner(text: str):
    print("\n" + "=" * 64)
    print(f"  {text}")
    print("=" * 64)


def show_result(label: str, result: dict, elapsed: float):
    status = result.get("status", "unknown").upper()
    icon   = "OK" if status == "SUCCESS" else "!!"
    print(f"  [{icon}] {label}: {status}  ({elapsed:.1f}s)")
    if result.get("decision"):
        # trim long decisions
        decision = str(result["decision"])[:120]
        print(f"       decision: {decision}")


async def main():
    banner("SOMA V2 — Live Ollama Demo (qwen2.5:3b)")
    print("  Real LLM planning | Memory cache | AirSim execution\n")

    # ── connect AirSim ────────────────────────────────────────────────────────
    print("Connecting to AirSim...")
    try:
        airsim_actuator = AirSimActuator()
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("Make sure Blocks.exe is running.")
        return
    print("AirSim connected.\n")

    # ── boot swarm ────────────────────────────────────────────────────────────
    print("Booting SOMASwarm with qwen2.5:3b...")
    swarm = SOMASwarm(
        model="ollama/qwen2.5:3b",
        slots=2,
        persist_dir="./soma_memory",
        trace_dir="./soma_traces",
        actuator=airsim_actuator,
        llm_timeout_s=90.0,
        llm_max_retries=1,
    )

    await swarm.director.start()
    print("Swarm online. 2 agents ready.\n")

    # ── Task 1: LLM generates plan from scratch ───────────────────────────────
    banner("Task 1 — LLM generates plan (first call, no cache)")
    print("  Task: 'Drone B1 patrol sector alpha, scan for survivors, RTB'")
    print("  Sending to qwen2.5:3b... (may take 5-15s)\n")

    t0 = time.monotonic()
    result1 = await swarm.dispatch(
        "Drone B1 patrol sector alpha, scan for survivors, return to base and land",
        urgency="high",
        forced_depth="complex",
    )
    t1 = time.monotonic()
    show_result("Task 1 (LLM cold)", result1, t1 - t0)

    # ── Task 2: same task — should hit L1 hot cache ───────────────────────────
    banner("Task 2 — Same task again (should hit L1 hot cache)")
    print("  Task: 'Drone B1 patrol sector alpha, scan for survivors, RTB'")
    print("  Expecting: near-instant, no LLM call\n")

    t2 = time.monotonic()
    result2 = await swarm.dispatch(
        "Drone B1 patrol sector alpha, scan for survivors, return to base and land",
        urgency="high"
    )
    t3 = time.monotonic()
    show_result("Task 2 (cache hit)", result2, t3 - t2)

    speedup = (t1 - t0) / max(t3 - t2, 0.01)
    print(f"\n  Cache speedup: {speedup:.1f}x faster than cold LLM call")

    # ── Task 3: different task — LLM called again ─────────────────────────────
    banner("Task 3 — Different task (LLM called again, new plan)")
    print("  Task: 'Deploy emergency supply kit to crash site at grid 12,0 via B1'")
    print("  Sending to qwen2.5:3b...\n")

    t4 = time.monotonic()
    result3 = await swarm.dispatch(
        "Deploy emergency supply kit to crash site at grid 12,0 via B1",
        urgency="high"
    )
    t5 = time.monotonic()
    show_result("Task 3 (LLM new plan)", result3, t5 - t4)

    # ── shutdown ──────────────────────────────────────────────────────────────
    await swarm.close()

    # ── summary ───────────────────────────────────────────────────────────────
    banner("Session Summary")
    stats = swarm.stats
    d = stats["director"]
    m = stats["memory"]
    print(f"  Tasks assigned   : {d.get('tasks_assigned', '?')}")
    print(f"  Tasks delegated  : {d.get('tasks_delegated', '?')}")
    print(f"  Bus messages     : {stats.get('bus_messages', '?')}")
    print(f"  Memory hot hits  : {m.get('hot_hits', '?')}")
    print(f"  Memory cold hits : {m.get('cold_hits', '?')}")
    print(f"  LLM misses       : {m.get('misses', '?')}")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
