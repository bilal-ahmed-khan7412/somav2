"""
Negotiation latency benchmark — realistic work-redistribution scenario.

Thundering-herd (all tasks → same drone) is the WORST case for negotiation:
there is nothing to redistribute, so overhead always dominates.

This benchmark uses the scenario negotiation is designed for:
  - 4 drones (B1-B4), 4 agents each specialised for one drone
  - Tasks arrive with a random target drone
  - When Agent A is busy with its drone, Agent B (owner of a different drone)
    can accept the negotiated step and use ITS drone instead
  - Negotiation OFF: conflict → failure/backtrack
  - Negotiation ON:  conflict → broker routes to any agent with spare capacity

Two runs, 60 tasks each, mixed-target contention (60% hotspot on B1, 40% spread).

Reports throughput, mean/p95 latency, negotiation accept rate, latency delta.
"""
import asyncio
import json
import logging
import random
import statistics
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

logging.basicConfig(level=logging.WARNING)

DRONES = ["B1", "B2", "B3", "B4"]
HOTSPOT_RATE = 0.6   # 60% tasks want B1, 40% spread across B2-B4


async def mock_llm_for(unit: str):
    """Returns an LLM callback that generates plans targeting a specific unit."""
    async def _cb(task_type: str, prompt: str, **kwargs) -> str:
        plan = {
            "steps": [
                {"id": "node1", "description": f"[CMD] TAKEOFF {unit}", "deps": []},
                {"id": "node2", "description": f"Execute mission with {unit}",   "deps": ["node1"]},
                {"id": "node3", "description": f"[CMD] LAND {unit}",    "deps": ["node2"]},
            ],
        }
        return f"```json\n{json.dumps(plan)}\n```"
    return _cb


async def mock_llm(task_type: str, prompt: str, **kwargs) -> str:
    """Generic LLM — unit extracted from task context in prompt."""
    # Pick unit from prompt or default B1
    for d in DRONES:
        if d in prompt:
            unit = d
            break
    else:
        unit = "B1"
    plan = {
        "steps": [
            {"id": "node1", "description": f"[CMD] TAKEOFF {unit}", "deps": []},
            {"id": "node2", "description": f"Execute mission with {unit}",   "deps": ["node1"]},
            {"id": "node3", "description": f"[CMD] LAND {unit}",    "deps": ["node2"]},
        ],
    }
    return f"```json\n{json.dumps(plan)}\n```"


def _pick_drone(i: int) -> str:
    if random.random() < HOTSPOT_RATE:
        return "B1"
    return random.choice(["B2", "B3", "B4"])


async def run_scenario(num_tasks: int, negotiation_enabled: bool, label: str, seed: int = 42):
    random.seed(seed)
    from soma_v2.core.director import AgentDirector
    from soma_v2.core.actuators import MockDroneActuator

    director = AgentDirector(llm_callback=mock_llm, actuator=MockDroneActuator(), claim_timeout_s=0.05)
    director.add_slot("agent_alpha", role="EMERGENCY", capacity=10)
    director.add_slot("agent_beta",  role="SUPERVISOR", capacity=10)
    director.add_slot("agent_gamma", role="PEER",       capacity=10)
    director.add_slot("agent_delta", role="ROUTINE",    capacity=10)
    await director.start()

    if negotiation_enabled:
        for sid, slot in director._slots.items():
            director._negotiator.register(sid, slot.kernel.deliberative._executor)
    else:
        # No-op broker
        from soma_v2.core.broker import NegotiationResult
        class _NullBroker:
            stats = {"total_attempts": 0, "success_rate": 0.0, "registered_agents": 0}
            async def propose(self, *a, **kw):
                return NegotiationResult(accepted=False)
        director._negotiator = _NullBroker()
        for slot in director._slots.values():
            slot.kernel.deliberative._executor.negotiation_broker = None

    task_latencies = []
    random.seed(seed)   # re-seed so both runs get same tasks
    task_drones = [_pick_drone(i) for i in range(num_tasks)]

    async def timed_task(i):
        drone = task_drones[i]
        t0 = time.perf_counter()
        result = await director.assign(
            f"Emergency at Sector {i} drone {drone}",
            urgency="high",
            forced_depth="complex",
        )
        lat = (time.perf_counter() - t0) * 1000
        task_latencies.append(lat)
        return result

    wall_start = time.perf_counter()
    results = await asyncio.gather(*[timed_task(i) for i in range(num_tasks)])
    wall_time = time.perf_counter() - wall_start

    await director.stop()

    successes = sum(1 for r in results if r.get("status") == "success")
    stats = director.stats

    return {
        "label":            label,
        "num_tasks":        num_tasks,
        "successes":        successes,
        "wall_time_s":      round(wall_time, 3),
        "throughput":       round(num_tasks / wall_time, 2),
        "mean_lat_ms":      round(statistics.mean(task_latencies), 2),
        "p95_lat_ms":       round(statistics.quantiles(task_latencies, n=20)[18], 2),
        "conflicts":        stats["blackboard"]["total_conflicts"],
        "neg_attempts":     stats["negotiation"]["total_attempts"],
        "neg_success_rate": stats["negotiation"]["success_rate"],
    }


async def run_pool_scenario(num_tasks: int, label: str, seed: int = 42):
    """
    ResourcePool primitive benchmark (task-level claiming).

    This is a direct demonstration of ResourcePool semantics: each task claims
    any free drone for the duration of its mission, queuing until one is available.
    This differs from the director scenarios (per-node claiming with fast failure)
    — pool claiming trades higher per-task latency for zero conflicts.
    """
    random.seed(seed)
    from soma_v2.core.blackboard import ResourceBlackboard, ResourcePool

    blackboard = ResourceBlackboard()
    pool = ResourcePool("drone_pool", DRONES)

    task_latencies = []
    conflicts = 0
    successes = 0

    async def pool_task(i: int):
        nonlocal conflicts, successes
        t0 = time.perf_counter()
        # Queue until any drone is free (task-level hold, same timeout as named scenario)
        unit = None
        deadline = time.perf_counter() + 30.0
        while unit is None and time.perf_counter() < deadline:
            unit = await pool.claim_any(f"agent_{i}", blackboard, f"task_{i}", timeout_s=0.05)
            if unit is None:
                await asyncio.sleep(0.005)
        if unit is None:
            conflicts += 1
            task_latencies.append((time.perf_counter() - t0) * 1000)
            return
        try:
            # Simulate 3-step mission held on same drone (stub work ~= claim_timeout_s)
            await asyncio.sleep(0.05)  # TAKEOFF
            await asyncio.sleep(0.05)  # MISSION
            await asyncio.sleep(0.05)  # LAND
            successes += 1
        finally:
            await pool.release(unit, f"agent_{i}", f"task_{i}", blackboard)
        task_latencies.append((time.perf_counter() - t0) * 1000)

    wall_start = time.perf_counter()
    await asyncio.gather(*[pool_task(i) for i in range(num_tasks)])
    wall_time = time.perf_counter() - wall_start

    return {
        "label":            label,
        "num_tasks":        num_tasks,
        "successes":        successes,
        "wall_time_s":      round(wall_time, 3),
        "throughput":       round(num_tasks / wall_time, 2),
        "mean_lat_ms":      round(statistics.mean(task_latencies), 2),
        "p95_lat_ms":       round(statistics.quantiles(task_latencies, n=20)[18], 2),
        "conflicts":        conflicts,
        "neg_attempts":     0,
        "neg_success_rate": 0.0,
    }


async def main():
    NUM_TASKS = 60

    print("\n" + "=" * 65)
    print("SOMA V2 — NEGOTIATION LATENCY BENCHMARK")
    print("=" * 65)
    print(f"Tasks: {NUM_TASKS} | 4 drones | hotspot={int(HOTSPOT_RATE*100)}% on B1")
    print("Three-tier resolution: Named/No-neg -> Named/Neg-ON -> Pool-claim")

    print("\n[1/3] Running WITHOUT negotiation (named claims) ...")
    run_a = await run_scenario(NUM_TASKS, negotiation_enabled=False, label="Named / No-neg")

    print("[2/3] Running WITH negotiation (named claims) ...")
    run_b = await run_scenario(NUM_TASKS, negotiation_enabled=True,  label="Named / Neg-ON")

    print("[3/3] Running WITH ResourcePool (any-free-drone claiming) ...")
    run_c = await run_pool_scenario(NUM_TASKS, label="Pool claim")

    lat_delta = (run_b["mean_lat_ms"] - run_a["mean_lat_ms"]) / run_a["mean_lat_ms"] * 100
    tput_delta = (run_b["throughput"] - run_a["throughput"]) / run_a["throughput"] * 100
    pool_lat_delta = (run_c["mean_lat_ms"] - run_a["mean_lat_ms"]) / run_a["mean_lat_ms"] * 100
    pool_tput_delta = (run_c["throughput"] - run_a["throughput"]) / run_a["throughput"] * 100

    print("\n" + "-" * 72)
    print(f"{'Metric':<35} {'No Neg':>10} {'Neg-ON':>10} {'Pool':>10}")
    print("-" * 72)
    for key, fmt in [
        ("successes",        "d"),
        ("throughput",       ".2f"),
        ("mean_lat_ms",      ".2f"),
        ("p95_lat_ms",       ".2f"),
        ("conflicts",        "d"),
        ("neg_attempts",     "d"),
        ("neg_success_rate", ".2f"),
    ]:
        a_val = run_a[key]
        b_val = run_b[key]
        c_val = run_c[key]
        lbl   = key.replace("_", " ").title()
        print(f"  {lbl:<33} {format(a_val, fmt):>10} {format(b_val, fmt):>10} {format(c_val, fmt):>10}")

    print("-" * 72)
    direction = "reduction" if lat_delta < 0 else "increase"
    print(f"  Neg-ON vs No-neg: latency {direction} {abs(lat_delta):.1f}%, "
          f"throughput {'gain' if tput_delta > 0 else 'loss'} {abs(tput_delta):.1f}%")
    pool_dir = "reduction" if pool_lat_delta < 0 else "increase"
    print(f"  Pool vs No-neg:   latency {pool_dir} {abs(pool_lat_delta):.1f}%, "
          f"throughput {'gain' if pool_tput_delta > 0 else 'loss'} {abs(pool_tput_delta):.1f}%")
    print("=" * 72)
    print()


if __name__ == "__main__":
    asyncio.run(main())
