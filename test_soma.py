import asyncio
import os
import sys
import logging

# OPENAI_API_KEY should be set as an environment variable or in a .env file
# os.environ["OPENAI_API_KEY"] = "sk-..." 


sys.path.insert(0, "/Users/hussamuddinsyed/Downloads/soma")
import demo_web_simulation

logging.basicConfig(level=logging.DEBUG)

async def test():
    print("--- STARTING TEST ---")
    await demo_web_simulation.run_scenario()
    print("--- DONE TEST ---")

if __name__ == "__main__":
    asyncio.run(test())
