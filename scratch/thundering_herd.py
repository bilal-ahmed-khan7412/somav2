import asyncio
import time
import logging
from soma_v2.core.director import AgentDirector
from soma_v2.core.kernel import V2Kernel

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("THUNDERING_HERD")

async def mock_llm_callback(task_type, prompt):
    # Simulate realistic LLM latency (2s - 5s)
    await asyncio.sleep(2.0 + (time.time() % 3.0))
    return "Resolved via stress test"

async def run_stress_test(num_tasks=120):
    print(f"\n[STRESS TEST] Launching {num_tasks} concurrent tasks (Thundering Herd)...")
    
    director = AgentDirector(
        llm_callback=mock_llm_callback,
    )
    # Add 10 agents to the pool
    for i in range(10):
        director.add_slot(f"agent_{i}", role="PEER", capacity=4)
        
    await director.start()

    events = [
        f"Critical resource conflict in sector {i % 10} - needs resolution"
        for i in range(num_tasks)
    ]

    t0 = time.perf_counter()
    tasks = [director.assign(ev) for ev in events]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.perf_counter() - t0

    successes = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    errors = sum(1 for r in results if not isinstance(r, dict))
    
    print(f"\n[RESULTS]")
    print(f"Total Tasks     : {num_tasks}")
    print(f"Successes       : {successes}")
    print(f"Errors/Exceptions: {errors}")
    print(f"Total Wall Time : {total_time:.2f}s")
    print(f"Avg Throughput  : {num_tasks / total_time:.2f} tasks/sec")
    
    stats = director.stats
    loads = stats["slot_loads"]
    print(f"Slot Loads      : {loads}")
    
    neg = stats["negotiation"]
    print(f"Negotiation     : {neg['success_rate']*100:.1f}% success ({neg['total_attempts']} attempts)")

    await director.stop()

if __name__ == "__main__":
    import sys
    num = 120
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    asyncio.run(run_stress_test(num))
