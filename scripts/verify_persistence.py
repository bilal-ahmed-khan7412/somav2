import asyncio
import logging
import sys
import os
import shutil

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from soma_v2.main import SOMASwarm

# Setup logging
logging.basicConfig(level=logging.ERROR)

PERSIST_DIR = os.path.join(os.getcwd(), "test_soma_memory")

async def verify():
    # Clear existing test memory
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print(f"Cleared {PERSIST_DIR}")

    print("\n--- Phase 1: Creating Memory ---")
    swarm1 = SOMASwarm(model="mock", persist_dir=PERSIST_DIR)
    
    print("Recording a successful mission...")
    # Use the actual task_done method which is the entry point for cold memory writes
    swarm1.memory.task_done(
        agent_id="Slot_0",
        agent_type="deliberative",
        action="recover_sample",
        urgency="low",
        success=True,
        extra={"detail": "crater alpha"}
    )
    
    # Block until background worker finishes
    swarm1.memory.sync()
    
    count1 = swarm1.memory.cold.episode_count
    print(f"Episodes in Swarm 1: {count1}")
    
    await swarm1.close()
    print("Swarm 1 closed.")

    print("\n--- Phase 2: Verifying Persistence ---")
    swarm2 = SOMASwarm(model="mock", persist_dir=PERSIST_DIR)
    
    # Sync just in case there's any startup indexing
    swarm2.memory.sync()
    
    count2 = swarm2.memory.cold.episode_count
    print(f"Episodes in Swarm 2: {count2}")
    
    if count2 == count1 and count1 > 0:
        print("\nSUCCESS: Memory persisted across restart!")
    else:
        print(f"\nFAILURE: Memory mismatch. Count1={count1}, Count2={count2}")
    
    await swarm2.close()

if __name__ == "__main__":
    asyncio.run(verify())
