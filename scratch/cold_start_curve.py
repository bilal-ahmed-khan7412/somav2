"""
SOMA V2 — Cold-Start Cache Curve
=================================
Runs a sequence of complex tasks and plots how the semantic cache hit rate
grows over time (per-task and rolling average).

This produces the "cache warm-up" figure: shows that SOMA V2 converges to
near-zero LLM plan-generation cost as the L1/L2 caches fill.

Usage
-----
  python scratch/cold_start_curve.py               # mock LLM (fast, proves mechanism)
  python scratch/cold_start_curve.py --use-ollama  # real Ollama (proves latency savings)
  python scratch/cold_start_curve.py --no-plot     # print table only (no matplotlib)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import List, Dict, Any

sys.path.insert(0, "src")

from soma_v2.core.director import AgentDirector
from soma_v2.memory.hierarchical import HierarchicalMemory

# ── task pool ─────────────────────────────────────────────────────────────────
# 6 unique complex task *types* — after normalisation they collapse to 3 keys,
# simulating a realistic swarm workload where the same mission recurs.

_TASK_POOL = [
    # type A — rescue missions (normalise → same key regardless of sector number)
    "Coordinate a 3-drone rescue mission for civilian in Sector 7.",
    "Coordinate a 3-drone rescue mission for civilian in Sector 12.",
    "Coordinate a 3-drone rescue mission for civilian in Sector 3.",
    # type B — multi-node sensor recalibration
    "Manage multi-node sensor recalibration across the southern grid.",
    "Manage multi-node sensor recalibration across the northern grid.",
    "Manage multi-node sensor recalibration across the eastern grid.",
    # type C — energy optimisation
    "Optimize energy distribution across the urban swarm network.",
    "Optimize energy distribution across the northern swarm network.",
    "Optimize energy distribution across the coastal swarm network.",
]

# Repeat pool to give the cache time to reach steady state
import random
random.seed(7)
WORKLOAD: List[str] = []
for _ in range(5):          # 5 passes → 45 tasks total (9 × 5)
    shuffled = _TASK_POOL[:]
    random.shuffle(shuffled)
    WORKLOAD.extend(shuffled)


# ── helpers ───────────────────────────────────────────────────────────────────

def _rolling_avg(hits: List[int], window: int = 5) -> List[float]:
    out = []
    for i in range(len(hits)):
        start = max(0, i - window + 1)
        chunk = hits[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


async def run(use_ollama: bool, model: str) -> List[Dict[str, Any]]:
    call_log: List[str] = []

    async def llm(label: str, prompt: str) -> str:
        call_log.append(label)
        if not use_ollama:
            await asyncio.sleep(0.20)
            if "deliberative_plan" in label:
                return json.dumps({"steps": [
                    {"id": "s1", "description": "Assess situation.", "deps": [], "alternative": None},
                    {"id": "s2", "description": "Coordinate team.",  "deps": ["s1"], "alternative": None},
                    {"id": "s3", "description": "Execute and verify.", "deps": ["s2"], "alternative": None},
                ]})
            return "Step completed."
        import aiohttp
        async with aiohttp.ClientSession() as session:
            payload = {"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "stream": False}
            async with session.post("http://localhost:11434/v1/chat/completions",
                                    json=payload,
                                    timeout=aiohttp.ClientTimeout(total=120)) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    mem      = HierarchicalMemory(cold_enabled=True)  # default persist_dir="./soma_cache"
    director = AgentDirector(llm_callback=llm, memory=mem)
    director.add_slot("agent_0", role="SUPERVISOR")
    director.add_slot("agent_1", role="PEER")
    await director.start()

    records: List[Dict[str, Any]] = []
    for i, task in enumerate(WORKLOAD):
        call_log.clear()
        t0 = time.perf_counter()
        r  = await director.assign(task, urgency="high", forced_depth="complex")
        elapsed = time.perf_counter() - t0

        meta  = r.get("result", {}).get("decision", {}).get("metadata", {})
        hit   = bool(meta.get("cached"))
        level = meta.get("cache_level") or "miss"
        plan_calls = sum(1 for c in call_log if "deliberative_plan" in c)

        records.append({
            "task_num":   i + 1,
            "task":       task[:50],
            "hit":        hit,
            "level":      level,
            "plan_calls": plan_calls,
            "total_calls": len(call_log),
            "latency_ms": elapsed * 1000,
        })
        status = f"[{'HIT ':>4} {level:<8}]" if hit else "[MISS         ]"
        print(f"  {i+1:>3}. {status}  calls={len(call_log)}  {elapsed*1000:7.1f}ms  {task[:45]}")

    await director.stop()
    return records


def print_table(records: List[Dict]) -> None:
    n     = len(records)
    hits  = sum(1 for r in records if r["hit"])
    calls = sum(r["plan_calls"] for r in records)
    print(f"\n  Tasks run        : {n}")
    print(f"  Cache hits       : {hits} / {n}  ({hits/n*100:.0f}%)")
    print(f"  Plan LLM calls   : {calls}  (saved {n - calls} plan generations)")
    # breakdown by quarter
    quarters = [records[i:i+n//4] for i in range(0, n, n//4)]
    print(f"\n  Hit rate by quarter:")
    for qi, q in enumerate(quarters[:4]):
        qhits = sum(1 for r in q if r["hit"])
        print(f"    Q{qi+1} (tasks {q[0]['task_num']:>2}–{q[-1]['task_num']:>2}): {qhits}/{len(q)} = {qhits/len(q)*100:.0f}%")


def plot_curve(records: List[Dict], use_ollama: bool) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("\n  matplotlib not installed — skipping plot (pip install matplotlib)")
        return

    task_nums = [r["task_num"]  for r in records]
    hit_vals  = [1 if r["hit"] else 0 for r in records]
    rolling   = _rolling_avg(hit_vals, window=5)
    plan_calls = [r["plan_calls"] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    mode = "Ollama" if use_ollama else "mock LLM"
    fig.suptitle(f"SOMA V2 — Semantic Cache Cold-Start Curve [{mode}]", fontsize=14, fontweight="bold")

    # ── subplot 1: per-task hit/miss ──────────────────────────────────────────
    ax1 = axes[0, 0]
    colors = ["#2ecc71" if h else "#e74c3c" for h in hit_vals]
    ax1.bar(task_nums, hit_vals, color=colors, width=0.8, alpha=0.8)
    ax1.plot(task_nums, rolling, "k-", linewidth=2, label="5-task rolling avg")
    ax1.set_xlabel("Task number")
    ax1.set_ylabel("Hit (1) / Miss (0)")
    ax1.set_title("Per-task cache hit")
    ax1.set_ylim(-0.1, 1.3)
    ax1.legend()
    hit_patch  = mpatches.Patch(color="#2ecc71", label="Hit")
    miss_patch = mpatches.Patch(color="#e74c3c", label="Miss")
    ax1.legend(handles=[hit_patch, miss_patch, ax1.get_lines()[0]])

    # ── subplot 2: cumulative hit rate ────────────────────────────────────────
    ax2 = axes[0, 1]
    cum_hits = np.cumsum(hit_vals)
    cum_rate = cum_hits / np.arange(1, len(hit_vals) + 1) * 100
    ax2.plot(task_nums, cum_rate, "b-o", markersize=3, linewidth=2)
    ax2.axhline(y=cum_rate[-1], color="gray", linestyle="--", alpha=0.7,
                label=f"Final: {cum_rate[-1]:.0f}%")
    ax2.fill_between(task_nums, cum_rate, alpha=0.15, color="blue")
    ax2.set_xlabel("Task number")
    ax2.set_ylabel("Cumulative hit rate (%)")
    ax2.set_title("Cumulative cache hit rate")
    ax2.set_ylim(0, 105)
    ax2.legend()

    # ── subplot 3: latency (hit vs miss) ──────────────────────────────────────
    ax3 = axes[1, 0]
    hit_lat  = [r["latency_ms"] for r in records if r["hit"]]
    miss_lat = [r["latency_ms"] for r in records if not r["hit"]]
    ax3.scatter([r["task_num"] for r in records if r["hit"]],  hit_lat,
                color="#2ecc71", s=30, label=f"Hit  (n={len(hit_lat)})", zorder=3)
    ax3.scatter([r["task_num"] for r in records if not r["hit"]], miss_lat,
                color="#e74c3c", s=30, label=f"Miss (n={len(miss_lat)})", zorder=3)
    if hit_lat:
        ax3.axhline(y=sum(hit_lat)/len(hit_lat), color="#27ae60", linestyle="--", alpha=0.7,
                    label=f"Avg hit:  {sum(hit_lat)/len(hit_lat):.0f}ms")
    if miss_lat:
        ax3.axhline(y=sum(miss_lat)/len(miss_lat), color="#c0392b", linestyle="--", alpha=0.7,
                    label=f"Avg miss: {sum(miss_lat)/len(miss_lat):.0f}ms")
    ax3.set_xlabel("Task number")
    ax3.set_ylabel("Latency (ms)")
    ax3.set_title("Latency: cache hit vs miss")
    ax3.legend()

    # ── subplot 4: plan LLM calls saved ──────────────────────────────────────
    ax4 = axes[1, 1]
    saved = [1 - c for c in plan_calls]   # 1 if no plan call needed, 0 otherwise
    cum_saved = np.cumsum(saved)
    ax4.bar(task_nums, saved, color=["#2ecc71" if s else "#e74c3c" for s in saved],
            width=0.8, alpha=0.7)
    ax4_r = ax4.twinx()
    ax4_r.plot(task_nums, cum_saved, "b-", linewidth=2, label=f"Total saved: {int(cum_saved[-1])}")
    ax4_r.set_ylabel("Cumulative plan calls saved", color="blue")
    ax4_r.legend(loc="upper left")
    ax4.set_xlabel("Task number")
    ax4.set_ylabel("Plan call saved (1=yes)")
    ax4.set_title(f"Plan LLM calls eliminated by cache")

    plt.tight_layout()
    out = "cold_start_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to: {out}")


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    mode = f"Ollama ({args.model})" if args.use_ollama else "mock LLM"
    print(f"\nSOMA V2 — Cold-Start Cache Curve  [{mode}]")
    print(f"  {len(WORKLOAD)} tasks · 9 unique texts · 3 normalised keys\n")

    records = await run(args.use_ollama, args.model)
    print_table(records)

    if not args.no_plot:
        plot_curve(records, args.use_ollama)


if __name__ == "__main__":
    asyncio.run(main())
