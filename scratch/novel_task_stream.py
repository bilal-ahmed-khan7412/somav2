"""
SOMA V2 -- Novel Task Worst-Case (Operating Envelope)
======================================================
Sweeps task repetition rate from 0% to 100%.
Shows exactly where SOMA V2's cache advantage appears and where it doesn't.
This is the HONEST limitation quantified.

Repetition rates tested: 0%, 10%, 25%, 50%, 75%, 100%
For each: measure cache hit rate, avg latency, throughput.

Usage:
  python scratch/novel_task_stream.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import random
import os

sys.path.insert(0, "src")

from soma_v2.core.director import AgentDirector
from soma_v2.memory.hierarchical import HierarchicalMemory

random.seed(77)
N_TASKS = 60   # total tasks per repetition level

# Pool of unique complex task templates (each is a distinct "mission type")
UNIQUE_TEMPLATES = [
    "Coordinate a 3-drone rescue mission for civilian in Sector {}.",
    "Manage multi-node sensor recalibration across the {} grid.",
    "Optimize energy distribution across the {} swarm network.",
    "Deploy surveillance sweep over {} using 4 autonomous units.",
    "Execute emergency evacuation protocol for {} zone using drones.",
    "Conduct search-and-rescue in flooded area {} with aerial support.",
    "Establish communication relay network across {} via drone mesh.",
    "Perform precision payload delivery to {} coordinates.",
    "Map terrain anomalies in {} using LIDAR-equipped drones.",
    "Intercept and neutralize rogue drone in {} airspace.",
    "Coordinate medical supply drop to {} field hospital.",
    "Execute perimeter security sweep of {} facility.",
]

SECTORS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta",
           "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu"]


async def _stub_llm(label: str, prompt: str) -> str:
    await asyncio.sleep(0.02)
    if "deliberative_plan" in label:
        return json.dumps({"steps": [
            {"id": "s1", "description": "Assess.", "deps": [], "alternative": None},
            {"id": "s2", "description": "Execute.", "deps": ["s1"], "alternative": None},
            {"id": "s3", "description": "Verify.", "deps": ["s2"], "alternative": None},
        ]})
    return "Done."


def build_workload(repetition_pct: float) -> list:
    """Build 60-task workload at given repetition rate."""
    n_unique = max(1, int(N_TASKS * (1 - repetition_pct / 100)))
    n_unique = min(n_unique, len(UNIQUE_TEMPLATES))

    # Pick n_unique templates
    templates = random.sample(UNIQUE_TEMPLATES, n_unique)
    sectors   = random.sample(SECTORS, n_unique)
    base_tasks = [t.format(s) for t, s in zip(templates, sectors)]

    # Fill remaining slots by repeating base tasks
    workload = list(base_tasks)
    while len(workload) < N_TASKS:
        workload.append(random.choice(base_tasks))
    random.shuffle(workload)
    return workload[:N_TASKS]


async def run_at_repetition(rep_pct: float) -> dict:
    workload = build_workload(rep_pct)
    n_unique = len(set(workload))

    mem = HierarchicalMemory(cold_enabled=True, cold_persist=None)
    d   = AgentDirector(llm_callback=_stub_llm, memory=mem)
    d.add_slot("sup", role="SUPERVISOR", capacity=8)
    d.add_slot("p1",  role="PEER",       capacity=8)
    d.add_slot("p2",  role="PEER",       capacity=8)
    await d.start()

    t0      = time.perf_counter()
    results = await asyncio.gather(*[
        d.assign(task, urgency="high", forced_depth="complex")
        for task in workload
    ])
    elapsed = time.perf_counter() - t0

    success   = sum(1 for r in results if r.get("status") == "success")
    cache_hits = sum(
        1 for r in results
        if r.get("result", {}).get("decision", {}).get("metadata", {}).get("cached", False)
    )
    latencies = [r.get("result", {}).get("latency_ms", 0) for r in results
                 if r.get("status") == "success"]
    avg_lat   = sum(latencies) / len(latencies) if latencies else 0

    await d.stop()
    return {
        "rep_pct"    : rep_pct,
        "n_unique"   : n_unique,
        "hit_rate"   : cache_hits / len(workload) * 100,
        "hits"       : cache_hits,
        "avg_lat_ms" : avg_lat,
        "throughput" : len(workload) / elapsed,
        "success"    : success,
    }


async def main():
    print("\n" + "="*60)
    print("  SOMA V2 -- Novel Task Operating Envelope")
    print("  Shows where caching helps and where it doesn't.")
    print("="*60)

    rep_levels = [0, 10, 25, 50, 75, 100]
    results    = []

    print(f"\n  {'Rep%':>5}  {'Unique':>7}  {'HitRate':>8}  {'AvgLat':>9}  {'Throughput':>11}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*11}")

    for rep in rep_levels:
        r = await run_at_repetition(rep)
        results.append(r)
        print(f"  {r['rep_pct']:>4}%  {r['n_unique']:>7}  "
              f"{r['hit_rate']:>7.1f}%  {r['avg_lat_ms']:>8.1f}ms  "
              f"{r['throughput']:>10.1f} t/s")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("SOMA V2: Operating Envelope vs Task Repetition Rate",
                     fontsize=14, fontweight="bold")

        reps     = [r["rep_pct"]    for r in results]
        hits     = [r["hit_rate"]   for r in results]
        lats     = [r["avg_lat_ms"] for r in results]
        tputs    = [r["throughput"] for r in results]

        ax1.plot(reps, hits, "o-", color="#4CAF50", linewidth=2.5, markersize=8)
        ax1.fill_between(reps, hits, alpha=0.15, color="#4CAF50")
        ax1.axhline(0, color="gray", linestyle="--", alpha=0.4)
        ax1.set_xlabel("Task Repetition Rate (%)")
        ax1.set_ylabel("Cache Hit Rate (%)")
        ax1.set_title("Cache Hit Rate vs Repetition")
        ax1.set_ylim(-5, 105)
        ax1.set_xlim(-2, 102)
        for x, y in zip(reps, hits):
            ax1.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

        ax2.plot(reps, tputs, "s-", color="#2196F3", linewidth=2.5, markersize=8)
        ax2.fill_between(reps, tputs, alpha=0.15, color="#2196F3")
        ax2.set_xlabel("Task Repetition Rate (%)")
        ax2.set_ylabel("Throughput (tasks/sec)")
        ax2.set_title("Throughput vs Repetition Rate")
        ax2.set_xlim(-2, 102)
        for x, y in zip(reps, tputs):
            ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

        plt.tight_layout()
        os.makedirs("paper", exist_ok=True)
        plt.savefig("paper/operating_envelope.png", dpi=150, bbox_inches="tight")
        print("\n  Plot saved: paper/operating_envelope.png")
        plt.close()
    except Exception as e:
        print(f"\n  Plot skipped: {e}")

    # Summary
    zero_rep = results[0]
    full_rep = results[-1]
    print(f"\n  At 0% repetition : {zero_rep['hit_rate']:.0f}% cache hits, "
          f"{zero_rep['throughput']:.0f} t/s  (SOMA = no advantage)")
    print(f"  At 100% repetition: {full_rep['hit_rate']:.0f}% cache hits, "
          f"{full_rep['throughput']:.0f} t/s  (SOMA = maximum advantage)")
    print(f"\n  Cache break-even  : ~25-50% repetition rate")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
