import asyncio
from soma_v2.main import SOMASwarm

async def test_bus_sync():
    print("Testing SOMASwarm Bus Synchronization...")
    swarm = SOMASwarm()
    
    # Check if they share the same object ID
    if id(swarm.bus) == id(swarm.director._bus):
        print("SUCCESS: Facade and Director share the same A2ABus instance.")
    else:
        print("FAILURE: Facade and Director have different A2ABus instances.")
        print(f"Facade Bus ID: {id(swarm.bus)}")
        print(f"Director Bus ID: {id(swarm.director._bus)}")

if __name__ == "__main__":
    asyncio.run(test_bus_sync())
