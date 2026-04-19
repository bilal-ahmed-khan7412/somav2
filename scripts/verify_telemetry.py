import asyncio
import os
import shutil
import json
from soma_v2.main import SOMASwarm

async def test_telemetry():
    persist_dir = "./test_telemetry_data"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    
    # Initialize swarm with telemetry
    swarm = SOMASwarm(
        persist_dir=persist_dir,
        trace_dir=os.path.join(persist_dir, "traces")
    )
    
    # Register a dummy tool
    async def scan_tool():
        return "Scan complete: 0 survivors found"
    swarm.register_tool("SCAN", scan_tool, "Scan area for survivors")

    print("--- Dispatching complex task (Direct assignment to test deliberative + tool) ---")
    # We use director.assign directly to ensure we hit the deliberative path for tool testing
    event = "Deploy unit D105 to Sector 7, then SCAN the area and RTB."
    result = await swarm.director.assign(
        event, 
        urgency="high",
        forced_depth="complex"
    )
    print(f"Result: {result['status']}")
    
    # Shutdown to sync telemetry
    await swarm.close()
    
    # Check for trace file
    trace_dir = os.path.join(persist_dir, "traces")
    if not os.path.exists(trace_dir):
        print("FAIL: No traces directory found!")
        return

    trace_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
    if not trace_files:
        print("FAIL: No .jsonl trace files found!")
        return

    print(f"SUCCESS: Found {len(trace_files)} trace files.")
    
    # Read last trace file
    last_trace = sorted(trace_files)[-1]
    trace_path = os.path.join(trace_dir, last_trace)
    
    print(f"--- Trace Content ({last_trace}) ---")
    with open(trace_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            ev_type = entry['event']
            tid = entry.get('task_id', 'N/A')
            print(f"[{entry['timestamp']}] EVENT: {ev_type} | TASK: {tid}")
            
            if ev_type == "task_assigned":
                print(f"  > Winner: {entry.get('winner_id')} | Hop: {entry.get('hop')}")
            elif ev_type == "kernel_dispatch":
                print(f"  > Depth: {entry.get('depth')} | Prob: {entry.get('depth_prob')}")
            elif ev_type == "tool_call":
                print(f"  > Tool: {entry.get('tool')} | Cmd: {entry.get('cmd')}")
            elif ev_type == "task_end":
                print(f"  > Status: {entry.get('status')} | Duration: {entry.get('duration_s')}s")

if __name__ == "__main__":
    asyncio.run(test_telemetry())
