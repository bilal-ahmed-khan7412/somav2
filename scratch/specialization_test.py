import asyncio
import json
import sys
import os

# Ensure src is in path
sys.path.insert(0, "src")

from soma_v2.core.director import AgentDirector
from soma_v2.memory.hierarchical import HierarchicalMemory

async def run_specialization_test():
    print("SOMA V2: Failure-Driven Specialization Experiment")
    print("==================================================")
    
    mem = HierarchicalMemory(cold_enabled=True, cold_persist=None)
    
    # --- PHASE 1: SEED GENERIC SUCCESS ---
    print("\n[Phase 1] Seeding Generic Success...")
    
    # A generic plan that works for basic tasks
    GENERIC_PLAN = json.dumps({"steps": ["Step 1: Generic move", "Step 2: Generic resolve"]})
    
    async def driver_success(label: str, prompt: str) -> str:
        if "plan" in label: return GENERIC_PLAN
        return "Task success."

    director = AgentDirector(llm_callback=driver_success, memory=mem)
    director.add_slot("s1", role="SUPERVISOR")
    await director.start()

    print("  Processing: 'Coordinate mission'...")
    # Seed a successful generic mission
    await director.assign("Coordinate mission", forced_depth="complex")
    await director.stop()
    mem.sync()

    # --- PHASE 2: TRIGGER SENSITIVE FAILURE ---
    print("\n[Phase 2] Triggering Failure on Sensitive Task...")
    print("  (Simulating that the generic plan fails in deep-sea conditions)")

    async def driver_fail(label: str, prompt: str) -> str:
        if "plan" in label: return GENERIC_PLAN # Tries to use generic plan
        raise RuntimeError("CRITICAL ERROR: High pressure failure!") # Fails during execution

    director_fail = AgentDirector(llm_callback=driver_fail, memory=mem)
    director_fail.add_slot("s1", role="SUPERVISOR")
    await director_fail.start()

    # This ROV task will match the 'Coordinate mission' cache, 
    # but the execution will FAIL.
    print("  Task: 'Coordinate ROV extraction at high pressure'...")
    res = await director_fail.assign("Coordinate ROV extraction at high pressure", forced_depth="complex")
    print(f"    -> Result: {res['status']} (Correctly recorded failure)")
    
    await director_fail.stop()
    mem.sync()

    # --- PHASE 3: VERIFY ADAPTATION ---
    print("\n[Phase 3] Testing Adaptation...")
    print("  The system should now recognize the failure and REFUSE to use the cache.")

    captured_prompts = []
    async def driver_verify(label: str, prompt: str) -> str:
        if "plan" in label:
            captured_prompts.append(prompt)
            return json.dumps({"steps": ["Step 1: Specialized check", "Step 2: Safe move"]})
        return "Task success."

    director_verify = AgentDirector(llm_callback=driver_verify, memory=mem)
    director_verify.add_slot("s1", role="SUPERVISOR")
    await director_verify.start()

    print("  Task: 'Coordinate ROV extraction at high pressure' (RETRY)...")
    res = await director_verify.assign("Coordinate ROV extraction at high pressure", forced_depth="complex")
    
    # CHECK 1: Did it hit the cache?
    meta = res.get("result", {}).get("decision", {}).get("metadata", {})
    cached = meta.get("cached")
    print(f"    -> [CACHE CHECK] Was cached? {cached}")
    
    # CHECK 2: Did it see the failure in the prompt?
    if captured_prompts:
        prompt = captured_prompts[0]
        contains_failure_context = "FAILED" in prompt
        print(f"    -> [MEMORY CHECK] Failure context in prompt? {contains_failure_context}")
        
    await director_verify.stop()
    
    print("\nCONCLUSION")
    print("----------")
    if not cached and contains_failure_context:
        print("  SUCCESS: SOMA V2 adapted. It saw the previous failure and requested a specialized plan.")
    else:
        print("  FAILURE: The system blindly used the broken cache or forgot the failure.")

if __name__ == "__main__":
    asyncio.run(run_specialization_test())
