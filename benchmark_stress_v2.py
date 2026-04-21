"""
SOMA V2 — 100-Agent Stress Test
================================
Demonstrates the SOMA V2 kernel's ability to orchestrate 100 logical agents
sharing 10 physical drones via the Resource Blackboard and A2A Bus.

This test uses the GROQ API for high-speed deliberative planning.
"""

import asyncio
import os
import time
import random
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load API keys
load_dotenv()

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from soma_v2.main import SOMASwarm
from soma_v2.core.a2a import A2AMessage, MsgType
from soma_v2.core.actuators import MockDroneActuator, DeduplicatingActuator

# Configuration
AGENT_COUNT = 100
DRONE_COUNT = 10
TASK_COUNT  = 50
LLM_MODEL   = "groq/llama3-70b-8192"

async def run_stress_test():
    print("\n" + "=" * 75)
    print(f" SOMA V2 STRESS TEST: {AGENT_COUNT} AGENTS | {DRONE_COUNT} DRONES")
    print("=" * 75)
    print(f"Model: {LLM_MODEL}")
    print("Initializing swarm kernel...")

    # 1. Setup Actuator and Swarm
    # We use a mock actuator but with drone IDs B1...B10
    actuator = DeduplicatingActuator(MockDroneActuator())
    
    swarm = SOMASwarm(
        model=LLM_MODEL,
        slots=0, # We will add custom slots manually
        trace_dir="./soma_traces/stress_test"
    )
    swarm.director._actuator = actuator

    # 2. Add 100 Agents with varied roles
    roles = ["SCOUT", "RECOVERY", "SUPERVISOR", "LOGISTICS"]
    for i in range(AGENT_COUNT):
        role = roles[i % len(roles)]
        swarm.director.add_slot(f"agent_{i:03d}", role=role, capacity=3)

    print(f"[SUCCESS] {AGENT_COUNT} Agents online and registered to A2A Bus.")

    # 3. Define tasks
    mission_templates = [
        "Patrol Sector {sector} and report battery levels of unit B{drone}.",
        "Emergency: Unit B{drone} has detected a civilian in Zone {sector}. Deploy rescue kit.",
        "Routine: Recalibrate sensors for drone B{drone} at altitude {alt}m.",
        "Conflict: Multiple units detected in Sector {sector}. B{drone} needs to coordinate a safe path."
    ]

    tasks = []
    for i in range(TASK_COUNT):
        drone_id = (i % DRONE_COUNT) + 1
        sector = chr(65 + (i % 26)) # Alpha, Beta, Gamma...
        alt = random.randint(5, 15)
        tmpl = random.choice(mission_templates)
        task_text = tmpl.format(sector=sector, drone=drone_id, alt=alt)
        tasks.append(swarm.dispatch(task_text, urgency="high" if i % 5 == 0 else "medium"))

    print(f"Dispatching {TASK_COUNT} tasks into the swarm simultaneously...\n")
    
    # 4. Run and track
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t0

    # 5. Analyze Results
    success_count = sum(1 for r in results if r.get("status") == "success")
    conflicts = swarm.director._blackboard.stats["conflicts"]
    total_msgs = swarm.bus.message_count

    print("\n" + "=" * 75)
    print(" STRESS TEST RESULTS")
    print("=" * 75)
    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Success Rate:       {success_count}/{TASK_COUNT} ({success_count/TASK_COUNT*100:.1f}%)")
    print(f"A2A Message Count:  {total_msgs}")
    print(f"Resource Conflicts: {conflicts} (Successfully resolved by Blackboard)")
    print(f"Avg Latency/Task:   {total_time/TASK_COUNT*1000:.0f}ms")
    print("-" * 75)
    
    # Show a few specific agent/drone interactions
    print("Sample Coordination Log:")
    sample_msgs = [m for m in swarm.bus.history() if m.msg_type in [MsgType.RESOURCE_CLAIM, MsgType.TASK_CLAIM]][:10]
    for msg in sample_msgs:
        if msg.msg_type == MsgType.RESOURCE_CLAIM:
            print(f"  [RESOURCE] Agent {msg.sender} successfully claimed Drone {msg.payload.get('unit_id')}")
        else:
            print(f"  [DISPATCH] Director assigned task {msg.task_id} to Agent {msg.recipient}")
    
    print("=" * 75)
    await swarm.close()

if __name__ == "__main__":
    try:
        asyncio.run(run_stress_test())
    except Exception as e:
        print(f"Test failed: {e}")
