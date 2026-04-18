"""
SOMA V2 — Actuator Demo
========================
Proves the physical actuation bridge works end-to-end through the real
Director → Kernel → DeliberativeAgent → PlanExecutor chain.

The actuator is passed at construction time — no post-hoc monkey-patching.
When a plan step contains a [CMD] tag, PlanExecutor calls the actuator
directly instead of (or in addition to) the LLM step execution.
"""
import asyncio
import json
import sys

sys.path.insert(0, "src")

from soma_v2.core.director import AgentDirector
from soma_v2.core.actuators import MockDroneActuator
from soma_v2.memory.hierarchical import HierarchicalMemory


async def mission_driver(label: str, prompt: str) -> str:
    """Minimal LLM stub — returns a plan with [CMD] tags for planning calls."""
    if "plan" in label:
        return json.dumps({"steps": [
            {"id": "takeoff",   "description": "Power up unit [CMD] TAKEOFF B4",              "deps": [],           "alternative": None},
            {"id": "navigate",  "description": "Fly to target  [CMD] GOTO B4 45.3305 -122.67","deps": ["takeoff"],  "alternative": None},
            {"id": "land",      "description": "Return to base [CMD] LAND B4",                 "deps": ["navigate"], "alternative": None},
        ]})
    return "Step confirmed."


async def run_actuator_demo() -> None:
    print("SOMA V2: Real-World Bridge (Actuator) Demo")
    print("===========================================")

    actuator = MockDroneActuator()
    memory   = HierarchicalMemory(cold_enabled=False)

    # actuator passed at construction — flows Director → Kernel → DeliberativeAgent → PlanExecutor
    director = AgentDirector(
        llm_callback=mission_driver,
        memory=memory,
        actuator=actuator,
    )
    director.add_slot("drone_1", role="SUPERVISOR")
    await director.start()

    print("\n[Scenario] Tactical extraction mission for unit B4")
    print("--- BEGIN MISSION ---")

    result = await director.assign(
        "Execute tactical extraction for B4",
        urgency="high",
        forced_depth="complex",
    )

    print("\n--- MISSION COMPLETE ---")
    decision = result.get("result", {}).get("decision", {})
    print(f"Summary  : {decision.get('plan_summary')}")
    print(f"Steps    : {decision.get('steps')}")
    print(f"Cached   : {decision.get('metadata', {}).get('cached')}")

    await director.stop()


if __name__ == "__main__":
    asyncio.run(run_actuator_demo())
