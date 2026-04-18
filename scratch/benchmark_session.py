"""
SOMA V2 — Comprehensive Benchmarking Session
============================================
Benchmarks:
1. Throughput & Latency (N agents, M concurrent tasks)
2. Negotiation Benefit (Contention ON vs OFF)
3. Cache Hit Curve (Extended from cold_start_curve.py)
4. Depth Classifier Distribution (Realistic task mix)
"""

import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Setup paths
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from soma_v2.core.director import AgentDirector
from soma_v2.core.actuators import MockDroneActuator
from soma_v2.memory.hierarchical import HierarchicalMemory
from soma_v2.core.depth_classifier import DEPTH_SIMPLE, DEPTH_MEDIUM, DEPTH_COMPLEX

# Disable logging for cleaner bench output unless needed
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("BENCH")

# ── Mock LLM ─────────────────────────────────────────────────────────────────

async def mock_llm_callback(task_type: str, prompt: str) -> str:
    """Simulates LLM latency and returns appropriate responses for V2 agents."""
    # Simulate thinking time
    await asyncio.sleep(0.02) 
    
    if "routing" in task_type:
        return json.dumps({
            "action": "dispatch",
            "metadata": {"target": "unit_alpha", "priority": 2}
        })
    elif "deliberative_plan" in task_type:
        # Include keywords to trigger [CMD] injection for negotiation tests
        return json.dumps({"steps": [
            {"id": "s1", "description": "Deploy unit and takeoff.", "deps": [], "alternative": None},
            {"id": "s2", "description": "Navigate to target.", "deps": ["s1"], "alternative": None},
            {"id": "s3", "description": "Scan area and verify.", "deps": ["s2"], "alternative": None},
        ]})
    
    return "OK"

# ── Actuators ────────────────────────────────────────────────────────────────

class ContestedActuator(MockDroneActuator):
    """Holds a unit claim to force negotiation."""
    def __init__(self, hold_time: float = 0.6):
        super().__init__()
        self.hold_time = hold_time

    async def execute_command(self, cmd: str) -> bool:
        # Any command involving A12 will be slow
        if "A12" in cmd:
            await asyncio.sleep(self.hold_time)
        return await super().execute_command(cmd)

# ── Helpers ──────────────────────────────────────────────────────────────────

def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0}
    sorted_lat = sorted(latencies)
    return {
        "p50": sorted_lat[int(len(sorted_lat) * 0.50)],
        "p95": sorted_lat[int(len(sorted_lat) * 0.95)],
        "p99": sorted_lat[int(len(sorted_lat) * 0.99)],
    }

# ── Benchmarks ───────────────────────────────────────────────────────────────

async def run_throughput_benchmark():
    print("\n[1/4] Throughput & Latency Benchmark")
    print("-" * 40)
    
    results = []
    configs = [
        (4, 20),   # 4 agents, 20 concurrent tasks
        (8, 40),   # 8 agents, 40 concurrent tasks
        (16, 80),  # 16 agents, 80 concurrent tasks
    ]

    for n_agents, n_tasks in configs:
        director = AgentDirector(llm_callback=mock_llm_callback)
        for i in range(n_agents):
            director.add_slot(f"agent_{i}", role="PEER", capacity=4)
        
        await director.start()
        
        start_time = time.perf_counter()
        # Mix of task types
        tasks = [
            director.assign(f"Routine task {i} status check", urgency="medium")
            for i in range(n_tasks)
        ]
        outcomes = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start_time
        
        latencies = [o.get("result", {}).get("latency_ms", 0) for o in outcomes if o.get("status") == "success"]
        percentiles = calculate_percentiles(latencies)
        
        throughput = n_tasks / elapsed
        print(f"  {n_agents:2d} agents | {n_tasks:2d} tasks | {throughput:6.1f} tasks/s | p50: {percentiles['p50']:5.1f}ms | p99: {percentiles['p99']:5.1f}ms")
        
        await director.stop()
        results.append({
            "agents": n_agents, "tasks": n_tasks, "throughput": throughput, "percentiles": percentiles
        })
    return results

async def run_negotiation_benchmark():
    print("\n[2/4] Negotiation Benefit Benchmark")
    print("-" * 40)
    
    # Tasks that both want A12
    workload = [
        "Mission: Rescue survivor with unit A12",
        "Mission: Emergency A12 deployment"
    ]

    async def run_suite(negotiation_on: bool):
        actuator = ContestedActuator(hold_time=0.6)
        director = AgentDirector(llm_callback=mock_llm_callback, actuator=actuator)
        director.add_slot("agent_0", role="PEER")
        director.add_slot("agent_1", role="PEER")
        
        # Patch claim_timeout_s
        for slot in director._slots.values():
            slot.kernel.deliberative._executor.claim_timeout_s = 0.2 if negotiation_on else 10.0
            
        await director.start()
        
        t0 = time.perf_counter()
        # Run them strictly concurrently
        tasks = [
            director.assign(task, urgency="high", forced_depth="complex")
            for task in workload
        ]
        outcomes = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0
        
        stats = director.stats
        neg_stats = stats.get("negotiation", {})
        
        await director.stop()
        return {
            "elapsed": elapsed,
            "negotiations": neg_stats.get("negotiations", 0),
            "success_rate": sum(1 for o in outcomes if o.get("status") == "success") / len(workload)
        }

    on_results = await run_suite(True)
    off_results = await run_suite(False)
    
    print(f"  Negotiation ON : {on_results['elapsed']:.2f}s | Negs: {on_results['negotiations']} | Success: {on_results['success_rate']*100:.0f}%")
    print(f"  Negotiation OFF: {off_results['elapsed']:.2f}s | Negs: {off_results['negotiations']} | Success: {off_results['success_rate']*100:.0f}%")
    
    benefit = (off_results['elapsed'] - on_results['elapsed'])
    benefit_pct = (benefit / off_results['elapsed'] * 100) if off_results['elapsed'] > 0 else 0
    print(f"  Latency savings: {benefit:.2f}s ({benefit_pct:.1f}%)")

async def run_cache_benchmark():
    print("\n[3/4] Cache Hit Curve Benchmark")
    print("-" * 40)
    
    # Repeat pool to see curve
    base_tasks = [
        "Mission: Rescue survivor in Sector 5",
        "Mission: Rescue survivor in Sector 9",
        "Coordinate: Sensor grid reset North",
        "Coordinate: Sensor grid reset South",
    ]
    
    workload = base_tasks * 5
    random.shuffle(workload)
    
    mem = HierarchicalMemory(cold_enabled=True)
    director = AgentDirector(llm_callback=mock_llm_callback, memory=mem)
    director.add_slot("agent_0", role="PEER")
    await director.start()
    
    hits = []
    latencies = []
    
    for i, task in enumerate(workload):
        t0 = time.perf_counter()
        res = await director.assign(task, urgency="high", forced_depth="complex")
        elapsed = (time.perf_counter() - t0) * 1000
        
        meta = res.get("result", {}).get("decision", {}).get("metadata", {})
        hit = meta.get("cached", False)
        hits.append(1 if hit else 0)
        latencies.append(elapsed)
        
        if (i + 1) % 5 == 0:
            current_hit_rate = sum(hits) / len(hits)
            avg_lat = sum(latencies) / len(latencies)
            print(f"  Tasks {i+1:2d} | Hit Rate: {current_hit_rate*100:3.0f}% | Avg Latency: {avg_lat:5.1f}ms")

    await director.stop()

async def run_depth_distribution_benchmark():
    print("\n[4/4] Depth Classifier Distribution Benchmark")
    print("-" * 40)
    
    mixed_workload = [
        # Simple
        "ping node 7", "check status of drone 1", "what is the battery of A1", "report health",
        # Medium
        "route drone 5 to base", "dispatch team to sector 4", "reassign agent 2 to patrol",
        # Complex
        "Coordinate a 3-drone rescue mission in Sector 7",
        "Optimize energy distribution across the southern grid",
        "Manage multi-node sensor recalibration across the network",
        "Full-scale emergency response coordination for flooding"
    ]
    
    director = AgentDirector(llm_callback=mock_llm_callback)
    director.add_slot("agent_0", role="SUPERVISOR")
    director.add_slot("agent_1", role="PEER")
    await director.start()
    
    for task in mixed_workload:
        await director.assign(task)
    
    # Aggregate from all slots
    combined_counts = {DEPTH_SIMPLE: 0, DEPTH_MEDIUM: 0, DEPTH_COMPLEX: 0}
    for slot in director._slots.values():
        summary = slot.kernel.dispatch_summary
        for k, v in summary.items():
            combined_counts[k] += v
    
    print("  Distribution:")
    total = sum(combined_counts.values()) or 1
    for k, v in combined_counts.items():
        print(f"    {k:12s}: {v:2d} ({v/total*100:4.1f}%)")
        
    await director.stop()

async def main():
    print("=" * 60)
    print(" SOMA V2 BENCHMARKING SESSION")
    print("=" * 60)
    
    await run_throughput_benchmark()
    await run_negotiation_benchmark()
    await run_cache_benchmark()
    await run_depth_distribution_benchmark()
    
    print("\n" + "=" * 60)
    print(" Benchmarking Session Complete")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
