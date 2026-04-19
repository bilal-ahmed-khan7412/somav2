import asyncio
import os
import shutil
from soma_v2.main import SOMASwarm

async def generate():
    persist_dir = "./sample_mission_data"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    
    swarm = SOMASwarm(
        persist_dir=persist_dir,
        trace_dir=os.path.join(persist_dir, "traces")
    )
    
    # 1. Simple task (D1/Reactive)
    print("Running D1 task...")
    await swarm.dispatch("Ping drone D101 and verify status.")
    
    # 2. Medium task (D2/Routing)
    print("Running D2 task...")
    await swarm.dispatch("Navigate unit D102 to Sector 5 and wait for further instructions.")
    
    # 3. Complex task (D3/Deliberative)
    # We'll force it to ensure tool calls etc.
    print("Running D3 task...")
    async def scan_tool():
        return "Thermal scan: No anomalies detected."
    swarm.register_tool("SCAN", scan_tool, "Execute a thermal scan")
    
    # Note: We use director.assign to force depth for deterministic report testing
    await swarm.director.assign(
        "Deploy D105 to Sector 7, then SCAN the perimeter and return to base.",
        forced_depth="complex"
    )
    
    await swarm.close()
    print(f"Mission complete. Traces saved to {persist_dir}/traces")

if __name__ == "__main__":
    asyncio.run(generate())
