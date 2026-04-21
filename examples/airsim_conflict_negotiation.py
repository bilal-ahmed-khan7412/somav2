
import asyncio
import logging
import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from soma_v2.core.director import AgentDirector
from soma_v2.core.actuators import AirSimActuator

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("SOMA_DEMO")

# Fixed Mock LLM for the demo
async def mock_llm_callback(task_type, prompt):
    """Returns the correct JSON structure for the DeliberativeAgent."""
    
    if "surveillance scan" in prompt.lower():
        # Alpha's long-running plan
        return json.dumps({
            "steps": [
                {"id": "a1", "description": "Take off drone B1. [CMD] TAKEOFF B1", "deps": []},
                {"id": "a2", "description": "Navigate to Sector A. [CMD] GOTO B1 5 5 -3", "deps": ["a1"]},
                {"id": "a3", "description": "Scanning area for anomalies. [CMD] SCAN B1 AREA_A", "deps": ["a2"]},
                {"id": "a4", "description": "Final transit. [CMD] GOTO B1 10 10 -3", "deps": ["a3"]},
                {"id": "a5", "description": "Mission complete, landing. [CMD] LAND B1", "deps": ["a4"]}
            ]
        })
    elif "medical supplies" in prompt.lower():
        # Beta's emergency mission on the same drone
        return json.dumps({
            "steps": [
                {"id": "b1", "description": "Divert to rescue site. [CMD] GOTO B1 15 5 -5", "deps": []},
                {"id": "b2", "description": "Release medical kit. [CMD] DEPLOY B1 MEDICAL_KIT", "deps": ["b1"]}
            ]
        })
    return json.dumps({"steps": [{"id": "s1", "description": "Wait.", "deps": []}]})

async def run_conflict_demo():
    print("\n" + "="*65)
    print(" SOMA V2 \u2014 Conflict Negotiation Demo")
    print(" Scenario: Two agents fighting for Drone1 (B1)")
    print("="*65)

    # 1. Setup Infrastructure
    actuator = AirSimActuator()
    director = AgentDirector(llm_callback=mock_llm_callback, actuator=actuator)

    # 2. Add Agent Slots
    # Slot Alpha: Routine Surveillance
    director.add_slot("alpha", role="ROUTINE", capacity=2)
    # Slot Beta: Emergency Rescue
    director.add_slot("beta", role="EMERGENCY", capacity=2)

    await director.start()

    print("\n[SYSTEM] Slots 'alpha' (Routine) and 'beta' (Emergency) are online.")
    print("[SYSTEM] Both will attempt to use Drone B1.\n")

    # 3. Dispatch Overlapping Missions
    print("[DISPATCH] Sending Routine Mission (Alpha) to Drone B1...")
    task_alpha = asyncio.create_task(director.assign(
        "Perform a surveillance scan of SECTOR_A using unit B1", 
        urgency="low",
        forced_depth="complex"
    ))
    
    await asyncio.sleep(8.0) # Let Alpha take off and get deep into the mission

    print("\n[DISPATCH] CRITICAL: Sending Emergency Mission (Beta) for SAME Drone B1...")
    task_beta = asyncio.create_task(director.assign(
        "Emergency! DEPLOY medical supplies at coordinate (15, 5, -5) using unit B1", 
        urgency="emergency",
        forced_depth="complex"
    ))

    # 4. Wait for both
    results = await asyncio.gather(task_alpha, task_beta)

    print("\n" + "="*65)
    print("Coordination Summary:")
    for res in results:
        print(f" - Task {res['task_id']} | Assigned to: {res['assigned_to']} | Status: {res['status']}")
    
    print(f"\nNegotiation Stats: {director.stats['negotiation']}")
    print("="*65)

    await director.stop()

if __name__ == "__main__":
    try:
        asyncio.run(run_conflict_demo())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Demo failed: {e}")
