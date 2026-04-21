"""
SOMA V2 + AirSim Demo — Complex Single-Drone Mission
=====================================================
Run this AFTER:
  1. Blocks.exe is running (from AirSim v1.8.1 Windows)
  2. venv set up: .venv\\Scripts\\python examples/airsim_demo.py

Mission: Drone B1 patrols three sectors, locates a civilian,
         deploys a rescue kit, then returns to base and lands.
"""
import asyncio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from soma_v2.core.actuators import AirSimActuator
from soma_v2.core.director import AgentDirector


async def mock_llm(task_type: str, prompt: str, **kwargs) -> str:
    import json
    plan = {
        "steps": [
            {
                "id": "s1",
                "description": "Take off and ascend to patrol altitude. [CMD] TAKEOFF B1",
                "deps": [],
                "alternative": None,
            },
            {
                "id": "s2",
                "description": "Patrol Sector Alpha — northwest quadrant. [CMD] GOTO B1 15 -10 -5",
                "deps": ["s1"],
                "alternative": None,
            },
            {
                "id": "s3",
                "description": "Scan Sector Alpha for survivors. [CMD] SCAN B1 ALPHA",
                "deps": ["s2"],
                "alternative": None,
            },
            {
                "id": "s4",
                "description": "Patrol Sector Beta — northeast quadrant. [CMD] GOTO B1 15 10 -5",
                "deps": ["s3"],
                "alternative": None,
            },
            {
                "id": "s5",
                "description": "Scan Sector Beta for survivors. [CMD] SCAN B1 BETA",
                "deps": ["s4"],
                "alternative": None,
            },
            {
                "id": "s6",
                "description": "Civilian located — move to Sector Gamma for extraction. [CMD] GOTO B1 5 0 -4",
                "deps": ["s5"],
                "alternative": None,
            },
            {
                "id": "s7",
                "description": "Scan Sector Gamma to confirm civilian position. [CMD] SCAN B1 GAMMA",
                "deps": ["s6"],
                "alternative": None,
            },
            {
                "id": "s8",
                "description": "Deploy emergency rescue kit to civilian. [CMD] DEPLOY B1 RESCUE_KIT",
                "deps": ["s7"],
                "alternative": None,
            },
            {
                "id": "s9",
                "description": "Return to base coordinates. [CMD] GOTO B1 0 0 -3",
                "deps": ["s8"],
                "alternative": None,
            },
            {
                "id": "s10",
                "description": "Descend and land at base. [CMD] LAND B1",
                "deps": ["s9"],
                "alternative": None,
            },
        ]
    }
    return f"```json\n{json.dumps(plan)}\n```"


async def main():
    print("\n" + "=" * 60)
    print("SOMA V2 — Complex Single-Drone Rescue Mission")
    print("=" * 60)
    print("Mission: Patrol 3 sectors, locate civilian, deploy kit, RTB")
    print("Connecting to AirSim...")

    try:
        actuator = AirSimActuator()
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("Make sure Blocks.exe is running before launching this script.")
        return

    print("Connected. Setting up SOMA director...\n")

    director = AgentDirector(
        llm_callback=mock_llm,
        actuator=actuator,
        claim_timeout_s=2.0,
    )
    director.add_slot("agent_alpha", role="EMERGENCY", capacity=10)
    await director.start()

    print("Dispatching 10-step rescue mission to drone B1...")
    print("Watch the drone patrol sectors in AirSim...\n")

    result = await director.assign(
        "Search and rescue — locate civilian across Sectors Alpha, Beta, Gamma. Deploy kit. RTB.",
        urgency="high",
        forced_depth="complex",
    )

    await director.stop()

    status = result.get("status")
    print(f"\n{'=' * 60}")
    print(f"Mission status: {status.upper()}")
    print("=" * 60)
    if status == "success":
        plan_detail = result.get("result", {}).get("plan_detail", {})
        for node_id, outcome in plan_detail.items():
            icon = "OK" if outcome["status"] == "success" else "FAIL"
            print(f"  [{icon}] {node_id}: {outcome['status']} ({outcome['latency_ms']:.0f}ms)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
