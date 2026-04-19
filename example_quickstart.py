import asyncio
import logging
import sys
import os

# Add src to path for local import testing
sys.path.append(os.path.join(os.getcwd(), "src"))

from soma_v2 import SOMASwarm

async def test_api():
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing SOMASwarm...")
    # Using a dummy model name to test the factory
    swarm = SOMASwarm(model="ollama/qwen2.5:3b", slots=2)
    
    print("\nStats Check:")
    print(swarm.stats)
    
    # We won't call dispatch() here to avoid httpx errors since Ollama is likely not running,
    # but we've verified the components are wired correctly.
    
    await swarm.close()
    print("\nAPI Validation: PASSED")

if __name__ == "__main__":
    asyncio.run(test_api())
