"""
SOMA V2 — Parallel Multi-Drone Mission Control
===========================================
REQUIREMENT: You must see TWO drones in AirSim Blocks before running.

Scenario:
  - Agent Alpha: Controls Drone1. Tasked with a low-altitude search.
  - Agent Beta: Controls Drone2. Tasked with a high-altitude surveillance sweep.
  - Both missions run SIMULTANEOUSLY.
"""
import asyncio
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from soma_v2.core.actuators import AirSimActuator
from soma_v2.core.director import AgentDirector

async def mock_llm(task_type: str, prompt: str, **kwargs) -> str:
    import json
    # Use different plans for search vs surveillance
    if "search" in prompt.lower():
        plan = {"steps": [
            {"id": "a1", "description": "Drone1 Takeoff. [CMD] TAKEOFF B1", "deps": [], "alternative": None},
            {"id": "a2", "description": "Drone1 Search Grid. [CMD] GOTO B1 10 -5 -3", "deps": ["a1"], "alternative": None},
            {"id": "a3", "description": "Drone1 Land. [CMD] LAND B1", "deps": ["a2"], "alternative": None},
        ]}
    else:
        plan = {"steps": [
            {"id": "b1", "description": "Drone2 Takeoff. [CMD] TAKEOFF B2", "deps": [], "alternative": None},
            {"id": "b2", "description": "Drone2 High Sweep. [CMD] GOTO B2 0 10 -10", "deps": ["b1"], "alternative": None},
            {"id": "b3", "description": "Drone2 Return. [CMD] GOTO B2 0 5 -3", "deps": ["b2"], "alternative": None},
            {"id": "b4", "description": "Drone2 Land. [CMD] LAND B2", "deps": ["b3"], "alternative": None},
        ]}
    return f"```json\n{json.dumps(plan)}\n```"

async def main():
    print("\n" + "="*65)
    print(" SOMA V2 — Parallel Multi-Drone Mission Control")
    print("="*65)
    print("  Two agents. Two physical drones. Zero overlap.")
    
    try:
        actuator = AirSimActuator()
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("Make sure Blocks.exe is running with 2 drones visible.")
        return

    director = AgentDirector(llm_callback=mock_llm, actuator=actuator)
    
    # Add two specialized agents
    director.add_slot("agent_alpha", role="EMERGENCY", capacity=5)
    director.add_slot("agent_beta",  role="ROUTINE",   capacity=5)
    await director.start()

    print("\n[SYSTEM] Launching parallel missions...")
    print("  - Drone1: Low-altitude Search (Alpha)")
    print("  - Drone2: High-altitude Surveillance (Beta)\n")

    # Dispatch BOTH at the same time using asyncio.gather
    t0 = time.monotonic()
    
    # Use unique task strings to bypass potential caching
    task1 = f"SEARCH mission for Drone1 at grid 10,-5 (id:{time.time()})"
    task2 = f"SURVEILLANCE mission for Drone2 at 10m altitude (id:{time.time()+1})"
    
    results = await asyncio.gather(
        director.assign(task1, urgency="high", forced_depth="complex"),
        director.assign(task2, urgency="low", forced_depth="complex")
    )
    t1 = time.monotonic()

    print("\n" + "="*65)
    print(f"Parallel Execution Complete in {t1-t0:.1f}s")
    for i, res in enumerate(results):
        status = res.get("status", "unknown").upper()
        agent = res.get("assigned_to", "?")
        print(f" Mission {i+1}: {status} | Assigned to: {agent}")
    print("="*65)

    await director.stop()

if __name__ == "__main__":
    asyncio.run(main())
