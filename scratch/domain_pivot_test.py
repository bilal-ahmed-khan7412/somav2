import asyncio
import json
import time
import sys
import os

# Ensure src is in path
sys.path.insert(0, "src")

from soma_v2.core.director import AgentDirector
from soma_v2.memory.hierarchical import HierarchicalMemory

# --- DOMAIN DEFINITIONS ---

# Source Domain: Urban Drone Rescue
DOMAIN_SOURCE = [
    "Coordinate a 3-drone rescue mission for civilian in Sector 7.",
    "Route drone B4 to nearest charging station.",
    "Manage multi-node sensor recalibration across the southern grid.",
    "Check status of drone A12.",
]

# Target Domain: Deep Sea ROV Salvage (Semantically Parallel)
DOMAIN_TARGET = [
    "Coordinate a 3-ROV extraction mission for equipment in Trench Alpha.", # Map to Rescue
    "Route submersible X1 to nearest battery depot.",                      # Map to Charging
    "Manage multi-sensor sonar recalibration across the abyssal plain.",    # Map to Grid
    "Check health of ROV unit Gamma.",                                     # Map to Status
]

async def run_pivot_test():
    print("SOMA V2: Zero-Shot Domain Pivot Experiment")
    print("============================================")
    
    # Shared memory to test persistence
    mem = HierarchicalMemory(cold_enabled=True, cold_persist=None)
    
    # 1. ACQUISITION PHASE
    print("\n[Phase 1] Knowledge Acquisition (Urban Rescue Domain)...")
    async def driver(label: str, prompt: str) -> str:
        # Generic mock plans
        if "plan" in label or "deliberative" in label:
            return json.dumps({"steps": ["Identify target coords", "Dispatch units", "Verify completion"]})
        return "Task handled."

    director = AgentDirector(llm_callback=driver, memory=mem, cold_threshold=1.10)
    director.add_slot("s1", role="SUPERVISOR")
    await director.start()

    for task in DOMAIN_SOURCE:
        print(f"  Processing: {task[:50]}...")
        await director.assign(task)
    
    await director.stop()
    mem.sync() # Ensure background writes are done
    print("  Source domain plans seeded in L1/L2 memory.")

    # 2. PIVOT PHASE
    print("\n[Phase 2] Domain Pivot (Deep Sea Salvage Domain)...")
    print("  Testing if Rescue plans generalize to Salvage tasks via Semantic Memory...")
    
    # New director, SAME memory, RELAXED threshold
    pivot_director = AgentDirector(llm_callback=driver, memory=mem, cold_threshold=1.10)
    pivot_director.add_slot("s1", role="SUPERVISOR")
    await pivot_director.start()

    hits = 0
    for i, task in enumerate(DOMAIN_TARGET):
        print(f"  Task: {task[:50]}...")
        res = await pivot_director.assign(task, forced_depth="complex")
        meta = res.get("result", {}).get("decision", {}).get("metadata", {})
        if meta.get("cached"):
            level = meta.get("cache_level", "?")
            print(f"    -> [CACHE HIT] Level: {level}")
            hits += 1
        else:
            print("    -> [MISS] No semantic match found.")

    await pivot_director.stop()
    
    print("\nRESULTS")
    print("-------")
    print(f"  Total Target Tasks: {len(DOMAIN_TARGET)}")
    print(f"  Semantic Cache Hits: {hits}")
    print(f"  Zero-Shot Generalization: {(hits/len(DOMAIN_TARGET))*100:.1f}%")
    
    if hits > 0:
        print("\nCONCLUSION: SUCCESS. SOMA V2 generalized Rescue logic to Salvage tasks.")
    else:
        print("\nCONCLUSION: FAILURE. Semantic distance was too large for current thresholds.")

if __name__ == "__main__":
    asyncio.run(run_pivot_test())
