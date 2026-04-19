import asyncio
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from soma_v2.main import SOMASwarm
from soma_v2.core.tools import RiskLevel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HITL_DEMO")

async def mock_deploy_payload(unit_id: str, payload_type: str):
    logger.info(f"!!! EXECUTING HIGH RISK ACTION: Deploying {payload_type} on {unit_id} !!!")
    await asyncio.sleep(1)
    return f"Payload {payload_type} deployed successfully on {unit_id}."

async def main():
    # 1. Initialize Swarm (using a mock or existing model)
    swarm = SOMASwarm(model="ollama/qwen2.5:3b", slots=2)

    # 2. Register a HIGH RISK tool
    swarm.tool_registry.register_tool(
        name="deploy_sensor_net",
        func=mock_deploy_payload,
        description="Deploy a physical sensor network in the current sector. HIGH RISK: Irreversible.",
        risk=RiskLevel.HIGH
    )

    # 3. Dispatch a task that will trigger this tool
    print("\n--- Dispatching High Risk Task ---")
    dispatch_task = asyncio.create_task(
        swarm.dispatch("Deploy a sensor network at sector 7 to monitor seismic activity.")
    )

    # 4. Monitor for suspension
    print("Monitoring for suspended tasks...")
    while True:
        suspended = swarm.get_suspended_tasks()
        if suspended:
            print(f"\n[ALERT] Human Intervention Required!")
            for agent_id, desc in suspended.items():
                print(f"  > Agent: {agent_id}")
                print(f"  > Task : {desc}")
                
                # Simulate user thinking and then approving
                print("\nSimulating user approval in 3 seconds...")
                await asyncio.sleep(3)
                swarm.approve(agent_id)
                print(f"Approval sent to {agent_id}.")
            break
        await asyncio.sleep(0.5)

    # 5. Wait for final result
    result = await dispatch_task
    print("\n--- Final Swarm Result ---")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
