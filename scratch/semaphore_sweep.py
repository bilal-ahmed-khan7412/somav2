import asyncio
import json
import time
import matplotlib.pyplot as plt
import sys
import os

# Ensure src and scratch are in path
sys.path.insert(0, "src")
sys.path.insert(0, "scratch")

from real_competitor_battle import run_soma, generate_workload

async def run_sweep():
    print("SOMA V2  Semaphore Sensitivity Analysis")
    print("========================================")
    
    workload = generate_workload()
    semaphores = [1, 2, 3, 4, 6]
    results = []

    for sem in semaphores:
        print(f"Running with semaphore={sem}...")
        # We use mock mode for the sweep to ensure it runs quickly for verification
        # In a real run, --use-ollama would be passed
        res = await run_soma(workload, use_ollama=False, model="mock", warm_cache=False, semaphore=sem)
        results.append({
            "semaphore": sem,
            "latency": res["latency"],
            "calls": res["calls"]
        })
        print(f"  Latency: {res['latency']:.2f}s, Calls: {res['calls']}")

    # Plotting
    sems = [r["semaphore"] for r in results]
    latencies = [r["latency"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(sems, latencies, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title("SOMA V2: Wall Time vs Semaphore Capacity", fontsize=14)
    plt.xlabel("Semaphore Capacity (Concurrent LLM Calls)", fontsize=12)
    plt.ylabel("Wall Clock Time (s)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xticks(sems)
    
    # Add labels
    for i, txt in enumerate(latencies):
        plt.annotate(f"{txt:.1f}s", (sems[i], latencies[i]), textcoords="offset points", xytext=(0,10), ha='center')

    output_path = "paper/semaphore_sweep.png"
    os.makedirs("paper", exist_ok=True)
    plt.savefig(output_path)
    print(f"\nSweep complete. Plot saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(run_sweep())
