"""
SOMA V2 + AirSim Demo — Multi-Agent A2A Coordination
=====================================================
Run this AFTER Blocks.exe is running.

Two SOMA agents (Alpha and Beta) are dispatched simultaneously:
  - Alpha gets a complex rescue mission (fills its capacity)
  - Beta gets a surveillance patrol
  - When Alpha is overloaded mid-mission, it delegates a sub-task to Beta
  - The A2A bus mediates the handoff transparently

Watch the drone execute both chains back-to-back in AirSim.
"""
import asyncio
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from soma_v2.core.actuators import AirSimActuator
from soma_v2.core.director import AgentDirector


# ── mock LLM responses keyed by task content ─────────────────────────────────

async def mock_llm(_task_type: str, _prompt: str, **kwargs) -> str:
    import json

    # Plan generation calls use task_type="deliberative_plan"; per-node calls use "deliberative"
    # Return the correct plan on each planning call; per-node calls just return "done"
    if _task_type != "deliberative_plan":
        return "Step completed successfully."

    mock_llm._plan_count = getattr(mock_llm, "_plan_count", 0) + 1

    if mock_llm._plan_count == 1:
        # Alpha's rescue mission — 6 steps
        plan = {
            "steps": [
                {"id": "a1", "description": "Alpha takes off. [CMD] TAKEOFF B1",                    "deps": [],     "alternative": None},
                {"id": "a2", "description": "Alpha flies to rescue zone. [CMD] GOTO B1 12 -8 -4",   "deps": ["a1"], "alternative": None},
                {"id": "a3", "description": "Alpha scans crash site. [CMD] SCAN B1 CRASH",           "deps": ["a2"], "alternative": None},
                {"id": "a4", "description": "Alpha deploys medical kit. [CMD] DEPLOY B1 MEDKIT",     "deps": ["a3"], "alternative": None},
                {"id": "a5", "description": "Alpha returns to base. [CMD] GOTO B1 0 0 -3",           "deps": ["a4"], "alternative": None},
                {"id": "a6", "description": "Alpha lands at base. [CMD] LAND B1",                    "deps": ["a5"], "alternative": None},
            ]
        }
    else:
        # Beta's surveillance patrol — 5 steps, different waypoints
        plan = {
            "steps": [
                {"id": "b1", "description": "Beta takes off for patrol. [CMD] TAKEOFF B1",          "deps": [],     "alternative": None},
                {"id": "b2", "description": "Beta sweeps north perimeter. [CMD] GOTO B1 20 0 -6",   "deps": ["b1"], "alternative": None},
                {"id": "b3", "description": "Beta scans north sector. [CMD] SCAN B1 NORTH",          "deps": ["b2"], "alternative": None},
                {"id": "b4", "description": "Beta sweeps east perimeter. [CMD] GOTO B1 0 20 -6",    "deps": ["b3"], "alternative": None},
                {"id": "b5", "description": "Beta returns and lands. [CMD] LAND B1",                 "deps": ["b4"], "alternative": None},
            ]
        }

    return f"```json\n{json.dumps(plan)}\n```"


# ── helpers ───────────────────────────────────────────────────────────────────

def print_header(title: str):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)

def print_result(label: str, result: dict):
    status = result.get("status", "unknown").upper()
    slot   = result.get("assigned_to", "?")
    hops   = result.get("hops", 0)
    icon   = "OK" if status == "SUCCESS" else "!!"
    print(f"  [{icon}] {label}: {status} | agent={slot} | hops={hops}")


# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    print_header("SOMA V2 — Multi-Agent A2A Coordination Demo")
    print("  Two agents. Parallel missions. A2A delegation on overload.")
    print("  Watch the drone execute both mission chains in AirSim.\n")

    print("Connecting to AirSim...")
    try:
        actuator = AirSimActuator()
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("Make sure Blocks.exe is running.")
        return
    print("Connected.\n")

    # Director with two slots: Alpha (supervisor) and Beta (peer)
    director = AgentDirector(
        llm_callback=mock_llm,
        actuator=actuator,
        claim_timeout_s=2.0,
    )
    director.add_slot("agent_alpha", role="SUPERVISOR", capacity=2)
    director.add_slot("agent_beta",  role="PEER",       capacity=4)
    await director.start()

    print("Agents online:")
    print("  agent_alpha  [SUPERVISOR]  capacity=2")
    print("  agent_beta   [PEER]        capacity=4")
    print()

    # ── Phase 1: dispatch Alpha's rescue mission ───────────────────────────────
    print("Phase 1 — Dispatching RESCUE mission to agent_alpha...")
    print("          (6-step mission: takeoff -> rescue zone -> scan -> deploy kit -> RTB -> land)\n")

    t0 = time.monotonic()
    rescue_result = await director.assign(
        "RESCUE: locate crash survivors at grid 12,-8 and deploy medical kit",
        urgency="high",
        forced_depth="complex",
    )
    t1 = time.monotonic()
    print_result("Rescue Mission", rescue_result)
    print(f"          Completed in {t1 - t0:.1f}s\n")

    # ── Phase 2: dispatch Beta's surveillance while Alpha is still cooling ─────
    print("Phase 2 — Dispatching SURVEILLANCE patrol to agent_beta...")
    print("          (5-step perimeter sweep while Alpha handles delegation)\n")

    t2 = time.monotonic()
    patrol_result = await director.assign(
        "PATROL: sweep north perimeter to waypoint 20,0 then east perimeter to 0,20 and RTB",
        urgency="medium",
        forced_depth="complex",
    )
    t3 = time.monotonic()
    print_result("Surveillance Patrol", patrol_result)
    print(f"          Completed in {t3 - t2:.1f}s\n")

    # ── Phase 3: overload Alpha and show delegation ────────────────────────────
    print("Phase 3 — Flooding agent_alpha to trigger A2A delegation...")
    print("          Sending 3 tasks simultaneously; Alpha (cap=2) will")
    print("          overflow and delegate extras to agent_beta.\n")

    flood_tasks = [
        director.assign(f"FLOOD TASK {i}: status check drone B1 sector {i}",
                        urgency="medium", forced_depth="simple")
        for i in range(3)
    ]
    t4 = time.monotonic()
    flood_results = await asyncio.gather(*flood_tasks)
    t5 = time.monotonic()

    agents_used = set(r.get("assigned_to", "?") for r in flood_results)
    print(f"  3 flood tasks resolved in {t5 - t4:.1f}s")
    print(f"  Agents that handled load: {', '.join(sorted(agents_used))}")
    for i, r in enumerate(flood_results):
        print_result(f"  Flood task {i}", r)

    await director.stop()

    # ── Summary ───────────────────────────────────────────────────────────────
    stats = director.stats
    print_header("Session Summary")
    print(f"  Tasks assigned   : {stats['tasks_assigned']}")
    print(f"  Tasks delegated  : {stats['tasks_delegated']}")
    print(f"  Tasks failed     : {stats['tasks_failed']}")
    print(f"  A2A bus messages : {stats['bus_messages']}")
    print(f"  Slot loads       : {stats['slot_loads']}")
    print("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())
