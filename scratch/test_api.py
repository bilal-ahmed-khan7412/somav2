import httpx
import asyncio
import time

async def test_api():
    url = "http://localhost:8000"
    
    print("1. Checking API Health...")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{url}/health")
            print(f"Health: {resp.json()}")
        except Exception as e:
            print(f"Server not reached: {e}")
            return

    print("\n2. Dispatching Mission...")
    payload = {"task": "Verify sensor health at sector 7", "urgency": "medium"}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{url}/mission", json=payload)
        data = resp.json()
        mission_id = data["mission_id"]
        print(f"Mission Dispatched: {mission_id}")

    print("\n3. Polling Status...")
    for _ in range(10):
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{url}/mission/{mission_id}")
            data = resp.json()
            print(f"Status: {data['status']}")
            if data["status"] in ["COMPLETED", "FAILED"]:
                print(f"Final Result: {data.get('result') or data.get('error')}")
                break
        await asyncio.sleep(2)

    print("\n4. Checking Metrics...")
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{url}/metrics")
        print(f"Metrics: {resp.json()}")

if __name__ == "__main__":
    asyncio.run(test_api())
