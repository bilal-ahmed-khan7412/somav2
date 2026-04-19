import asyncio
import logging
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from soma_v2.core.director import AgentDirector
from soma_v2.core.actuators import MockDroneActuator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BENCHMARK")

async def mock_llm(task_type: str, prompt: str, **kwargs) -> str:
    """Mock LLM that generates a simple plan with unit locks."""
    # Force contention on a single unit to trigger Blackboard/Broker
    unit = "B1"
    
    plan = {
        "thought": "I need to survey the zone and lock a drone.",
        "steps": [
            {"id": "node1", "description": f"[CMD] TAKEOFF {unit}", "deps": []},
            {"id": "node2", "description": f"Surveying sector alpha with {unit}", "deps": ["node1"]},
            {"id": "node3", "description": f"[CMD] LAND {unit}", "deps": ["node2"]}
        ]
    }
    import json
    return f"```json\n{json.dumps(plan)}\n```"

async def run_benchmark():
    print("\n" + "="*60)
    print("SOMA V2 BENCHMARK: THE THUNDERING HERD")
    print("="*60)
    print(f"Scenario: 50 Tasks | 4 Agents | 4 Shared Resource Units")
    
    actuator = MockDroneActuator()
    director = AgentDirector(llm_callback=mock_llm, actuator=actuator, claim_timeout_s=0.1)
    
    # Add 4 agents with different roles
    director.add_slot("agent_alpha", role="EMERGENCY", capacity=10)
    director.add_slot("agent_beta",  role="SUPERVISOR", capacity=10)
    director.add_slot("agent_gamma", role="PEER", capacity=10)
    director.add_slot("agent_delta", role="ROUTINE", capacity=10)
    
    await director.start()
    
    # Register agents with broker for negotiation
    for sid, slot in director._slots.items():
        director._negotiator.register(sid, slot.kernel.deliberative._executor)
    
    tasks = []
    num_tasks = 100
    t0 = time.perf_counter()
    
    print(f"\nDispatching {num_tasks} tasks (FORCED CONTENTION ON B1)...")
    for i in range(num_tasks):
        tasks.append(director.assign(f"Emergency at Sector {i}", urgency="high", forced_depth="complex"))
    
    results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - t0
    
    await director.stop()
    
    # Analysis
    successes = [r for r in results if r.get("status") == "success"]
    failures = [r for r in results if r.get("status") != "success"]
    
    throughput = len(results) / duration
    
    print("\n" + "-"*60)
    print(f"BENCHMARK RESULTS")
    print(f"Total Tasks:  {len(results)}")
    print(f"Successes:    {len(successes)}")
    print(f"Failures:     {len(failures)}")
    print(f"Total Time:   {duration:.2f}s")
    print(f"Throughput:   {throughput:.2f} tasks/s")
    print("-"*60)
    
    stats = director.stats
    print(f"Total Resource Claims:        {stats['blackboard']['total_claims']}")
    print(f"Resource Conflicts (Timeouts): {stats['blackboard']['total_conflicts']}")
    print(f"Negotiation Attempts:         {stats['negotiation']['total_attempts']}")
    print(f"Negotiation Success Rate:     {stats['negotiation']['success_rate']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
