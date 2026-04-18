"""
SOMA V2 -- Saturation Sweep (Scale Ceiling Analysis)
=====================================================
Sweeps agent count from 1 to 64 on a fixed 100-task workload.
Finds where throughput plateaus (architectural ceiling).

Usage:
  python scratch/saturation_sweep.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import os
import random

sys.path.insert(0, "src")

from soma_v2.core.director import AgentDirector
from soma_v2.memory.hierarchical import HierarchicalMemory

random.seed(42)

TASK_POOL = [
    ("ping node 7", "low"),
    ("check status drone A1", "low"),
    ("report health sensor B2", "low"),
    ("route drone C3 to charging bay", "medium"),
    ("dispatch team to sector 4", "medium"),
    ("Coordinate a 3-drone rescue mission for civilian in Sector 7.", "high"),
    ("Manage multi-node sensor recalibration across the southern grid.", "high"),
    ("Optimize energy distribution across the urban swarm network.", "high"),
]

WORKLOAD = [random.choice(TASK_POOL) for _ in range(100)]


async def _stub_llm(label: str, prompt: str) -> str:
    await asyncio.sleep(0.02)
    if "deliberative_plan" in label:
        return json.dumps({"steps": [
            {"id": "s1", "description": "Assess.", "deps": [], "alternative": None},
            {"id": "s2", "description": "Execute.", "deps": ["s1"], "alternative": None},
            {"id": "s3", "description": "Verify.", "deps": ["s2"], "alternative": None},
        ]})
    return "Done."


async def run_with_n_agents(n_agents: int, warm: bool = True) -> dict:
    mem = HierarchicalMemory(cold_enabled=False)
    d   = AgentDirector(llm_callback=_stub_llm, memory=mem)
    d.add_slot("sup", role="SUPERVISOR", capacity=8)
    for i in range(n_agents - 1):
        d.add_slot(f"peer_{i}", role="PEER", capacity=8)
    await d.start()

    # Warm cache if requested
    if warm:
        seen = set()
        for text, urg in WORKLOAD:
            if text not in seen and urg == "high":
                seen.add(text)
                await d.assign(text, urgency=urg, forced_depth="complex")

    t0      = time.perf_counter()
    results = await asyncio.gather(*[
        d.assign(text, urgency=urg) for text, urg in WORKLOAD
    ])
    elapsed = time.perf_counter() - t0

    success    = sum(1 for r in results if r.get("status") == "success")
    throughput = len(WORKLOAD) / elapsed
    latencies  = sorted(r.get("result", {}).get("latency_ms", 0)
                        for r in results if r.get("status") == "success")

    def pct(p):
        idx = int(len(latencies) * p / 100)
        return latencies[min(idx, len(latencies) - 1)] if latencies else 0

    await d.stop()
    return {
        "n_agents"  : n_agents,
        "throughput": throughput,
        "p50"       : pct(50),
        "p95"       : pct(95),
        "p99"       : pct(99),
        "success"   : success,
        "elapsed"   : elapsed,
    }


async def main():
    print("\n" + "="*60)
    print("  SOMA V2 -- Agent Scale Saturation Sweep")
    print("  100 tasks, warm cache, varying agent count")
    print("="*60)

    agent_counts = [1, 2, 4, 8, 16, 32, 64]
    results = []

    print(f"\n  {'Agents':>7}  {'t/s':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'OK':>5}")
    print(f"  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*5}")

    for n in agent_counts:
        r = await run_with_n_agents(n)
        results.append(r)
        print(f"  {r['n_agents']:>7}  {r['throughput']:>7.1f}  "
              f"{r['p50']:>7.1f}ms  {r['p95']:>7.1f}ms  "
              f"{r['p99']:>7.1f}ms  {r['success']:>5}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("SOMA V2: Throughput & Latency vs Agent Count (100 Tasks, Warm Cache)",
                     fontsize=13, fontweight="bold")

        agents = [r["n_agents"]   for r in results]
        tputs  = [r["throughput"] for r in results]
        p50s   = [r["p50"]        for r in results]
        p95s   = [r["p95"]        for r in results]
        p99s   = [r["p99"]        for r in results]

        ax1.plot(agents, tputs, "o-", color="#9C27B0", linewidth=2.5, markersize=8)
        ax1.fill_between(agents, tputs, alpha=0.12, color="#9C27B0")
        ax1.set_xlabel("Number of Agent Slots")
        ax1.set_ylabel("Throughput (tasks/sec)")
        ax1.set_title("Throughput Scaling")
        ax1.set_xscale("log", base=2)
        ax1.set_xticks(agents)
        ax1.set_xticklabels([str(a) for a in agents])
        for x, y in zip(agents, tputs):
            ax1.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)

        ax2.plot(agents, p50s, "o-", label="p50", color="#4CAF50", linewidth=2)
        ax2.plot(agents, p95s, "s-", label="p95", color="#FF9800", linewidth=2)
        ax2.plot(agents, p99s, "^-", label="p99", color="#F44336", linewidth=2)
        ax2.set_xlabel("Number of Agent Slots")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Latency Distribution vs Scale")
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(agents)
        ax2.set_xticklabels([str(a) for a in agents])
        ax2.legend()

        plt.tight_layout()
        os.makedirs("paper", exist_ok=True)
        plt.savefig("paper/saturation_sweep.png", dpi=150, bbox_inches="tight")
        print("\n  Plot saved: paper/saturation_sweep.png")
        plt.close()
    except Exception as e:
        print(f"\n  Plot skipped: {e}")

    # Find saturation point
    max_tput = max(r["throughput"] for r in results)
    sat_point = next(r for r in results if r["throughput"] >= max_tput * 0.95)
    print(f"\n  Peak throughput  : {max_tput:.1f} t/s")
    print(f"  Saturation at    : {sat_point['n_agents']} agents")
    print(f"  p99 at saturation: {sat_point['p99']:.1f}ms")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
