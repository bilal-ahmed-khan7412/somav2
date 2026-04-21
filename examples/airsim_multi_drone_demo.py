"""
SOMA V2 — Multi-Drone AirSim Coordination Demo
==============================================
This demo showcases how SOMA V2 orchestrates multiple drones (B1 and B2)
using a shared Resource Blackboard to prevent contention and an A2A bus
for task negotiation.

Requirements:
  1. AirSim (Blocks environment) running with TWO drones configured:
     - Drone1 (mapped to B1)
     - Drone2 (mapped to B2)
  2. pip install airsim httpx python-dotenv
"""

import asyncio
import sys
import os
import json
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from soma_v2.core.actuators import AirSimActuator, DeduplicatingActuator
from soma_v2.core.director import AgentDirector
from soma_v2.core.a2a import A2ABus, ResourceBlackboard

async def multi_drone_mock_llm(task_type: str, prompt: str, **kwargs) -> str:
    """
    Mocks a complex multi-drone plan.
    In a real scenario, this would come from Google/Groq/Deepseek.
    """
    plan = {
        "steps": [
            {
                "id": "s1",
                "description": "B1: Take off and scout Sector Alpha. [CMD] TAKEOFF B1",
                "deps": [],
                "alternative": None,
            },
            {
                "id": "s2",
                "description": "B2: Take off and standby for recovery. [CMD] TAKEOFF B2",
                "deps": [],
                "alternative": None,
            },
            {
                "id": "s3",
                "description": "B1: Scan Sector Alpha for anomalies. [CMD] SCAN B1 ALPHA",
                "deps": ["s1"],
                "alternative": None,
            },
            {
                "id": "s4",
                "description": "B1: Target located. B2 move to drop zone. [CMD] GOTO B2 10 10 -5",
                "deps": ["s3", "s2"],
                "alternative": None,
            },
            {
                "id": "s5",
                "description": "B2: Deploy rescue kit. [CMD] DEPLOY B2 KIT_ALPHA",
                "deps": ["s4"],
                "alternative": None,
            },
            {
                "id": "s6",
                "description": "B1: Return to base. [CMD] GOTO B1 0 0 -3",
                "deps": ["s5"],
                "alternative": None,
            },
            {
                "id": "s7",
                "description": "B2: Return to base. [CMD] GOTO B2 0 0 -3",
                "deps": ["s5"],
                "alternative": None,
            },
            {
                "id": "s8",
                "description": "B1: Land. [CMD] LAND B1",
                "deps": ["s6"],
                "alternative": None,
            },
            {
                "id": "s9",
                "description": "B2: Land. [CMD] LAND B2",
                "deps": ["s7"],
                "alternative": None,
            },
        ]
    }
    return f"```json\n{json.dumps(plan)}\n```"

async def main():
    print("\n" + "=" * 70)
    print(" SOMA V2 — MULTI-DRONE COORDINATED MISSION (B1 & B2)")
    print("=" * 70)
    print("Scenario: B1 Scouts, B2 Recovers. Shared Resource Blackboard enabled.")
    
    # 1. Setup shared components
    bus = A2ABus()
    blackboard = ResourceBlackboard(bus=bus)
    
    # 2. Setup Actuator (with Deduplication)
    try:
        raw_actuator = AirSimActuator()
        actuator = DeduplicatingActuator(raw_actuator)
        print("[SUCCESS] Connected to AirSim (Drone1 & Drone2 recognized)")
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        print("Falling back to Mock Actuator for demonstration...")
        from soma_v2.core.actuators import MockDroneActuator
        actuator = MockDroneActuator()

    # 3. Setup Director with 2 slots
    director = AgentDirector(
        llm_callback=multi_drone_mock_llm, # Change to get_llm_callback("groq/llama3-70b") for real LLM
        actuator=actuator,
        bus=bus,
        blackboard=blackboard,
        claim_timeout_s=5.0
    )
    
    # Assign roles
    director.add_slot("agent_alpha", role="SCOUT", capacity=5)
    director.add_slot("agent_beta", role="RECOVERY", capacity=5)
    
    await director.start()

    print("\nDispatching multi-drone mission: 'Scout Alpha with B1, then deploy kit with B2'...")
    
    # 4. Dispatch the task
    result = await director.assign(
        "Coordinate B1 and B2: Scout Sector Alpha for survivors. If found, B2 must deploy kit. Both RTB.",
        urgency="high",
        forced_depth="complex"
    )

    await director.stop()

    # 5. Report results
    status = result.get("status")
    print(f"\n{'=' * 70}")
    print(f"MISSION STATUS: {status.upper()}")
    print(f"Total A2A Messages: {bus.message_count}")
    print(f"Resource Conflicts Resolved: {blackboard.stats['conflicts']}")
    print(f"{'=' * 70}")
    
    if status == "success":
        plan_detail = result.get("result", {}).get("plan_detail", {})
        for node_id, outcome in plan_detail.items():
            desc = outcome.get("description", "No description")
            print(f"  [OK] {node_id}: {desc}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
