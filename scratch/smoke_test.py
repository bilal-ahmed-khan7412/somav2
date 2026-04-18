import asyncio, sys, json
sys.path.insert(0, "src")

async def smoke():
    from soma_v2.core.director import AgentDirector
    from soma_v2.memory.hierarchical import HierarchicalMemory

    async def stub(label, prompt):
        await asyncio.sleep(0.01)
        if "deliberative" in label:
            return json.dumps({"steps": [
                {"id": "s1", "description": "Execute mission.", "deps": [], "alternative": None}
            ]})
        return "Reroute to supervisor agent."

    d = AgentDirector(llm_callback=stub, memory=HierarchicalMemory(cold_enabled=False))
    d.add_slot("sup", role="SUPERVISOR", capacity=4)
    await d.start()

    tests = [
        ("ping node 7",                                                   "low",  None),
        ("check status drone A1",                                         "low",  None),
        ("route drone C3 to charging base",                               "medium", None),
        ("Coordinate a 3-drone rescue mission for civilian in Sector 7.", "high", "complex"),
        ("Manage multi-node sensor recalibration across the southern grid.", "high", "complex"),
    ]

    print()
    for task, urg, fd in tests:
        r   = await d.assign(task, urgency=urg, forced_depth=fd)
        res = r.get("result", {})
        dec = res.get("decision", {})
        act = dec.get("action", "?")
        aty = res.get("agent_type", "?")
        unit = dec.get("unit", dec.get("target_unit", "-"))
        print(f"  [{aty:<14}] action={act:<20} unit={unit:<12} | {task[:50]}")

    stats = d.stats
    hot = stats["memory"]["hot"]
    print(f"\n  HotMemory stats: {hot}")
    await d.stop()
    print("\n  [OK] All smoke tests passed.")

asyncio.run(smoke())
