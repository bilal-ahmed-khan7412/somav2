"""
SOMA V2 — Multi-Agent Concurrent Demo
======================================

Scenario: 3:00 AM Production Incident Flood
─────────────────────────────────────────────
Your monitoring system fires 3 alerts at the exact same moment:

  [agent_0] PRIMARY DATABASE is down — investigate, run failover, page team
  [agent_1] API gateway returning 503s — diagnose and decide routing action
  [agent_2] Load balancer heartbeat check — quick status ping

All 3 arrive at the SAME TIME.
All 3 agents handle them IN PARALLEL.

Run:
    export OPENAI_API_KEY="sk-..."
    python3 demo_multiagent.py
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
CYAN    = "\033[36m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
RED     = "\033[31m"
WHITE   = "\033[97m"

# One colour per agent slot
SLOT_COLOURS = {
    "agent_0": MAGENTA,   # complex task — deliberative
    "agent_1": YELLOW,    # medium task  — routing
    "agent_2": GREEN,     # simple task  — reactive
}

SLOT_LABELS = {
    "agent_0": "agent_0 [SUPERVISOR]",
    "agent_1": "agent_1 [PEER]     ",
    "agent_2": "agent_2 [PEER]     ",
}

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def log(slot: str, msg: str, colour: str = None) -> None:
    c = colour or SLOT_COLOURS.get(slot, WHITE)
    label = SLOT_LABELS.get(slot, slot)
    print(f"  {DIM}[{ts()}]{RESET}  {c}{BOLD}{label}{RESET}  {msg}")

def separator(title: str = "") -> None:
    if title:
        print(f"\n{BOLD}{CYAN}{'─'*20} {title} {'─'*(39-len(title))}{RESET}\n")
    else:
        print(f"\n{DIM}{'─'*64}{RESET}\n")


# ── Simulated tools ───────────────────────────────────────────────────────────

async def check_db_replication() -> str:
    await asyncio.sleep(0.2)
    return "Replica db-02: lag=1.4s, status=LAGGING — can promote"

async def promote_replica() -> str:
    await asyncio.sleep(0.4)
    return "db-02 promoted to PRIMARY. DNS updated. Connections rerouting."

async def page_oncall_team() -> str:
    await asyncio.sleep(0.1)
    return "PagerDuty alert sent to db-oncall. ETA 5 minutes."

async def check_api_gateway() -> str:
    await asyncio.sleep(0.15)
    return "api-gw-01: 503_rate=78% upstream_timeouts=high pool_exhausted=True"

async def check_lb_health() -> str:
    await asyncio.sleep(0.1)
    return "Load balancer LB-01: status=HEALTHY  active_nodes=4/4  rps=2847"


# ── Result printer ────────────────────────────────────────────────────────────

def print_result(slot: str, result: dict, wall_ms: float) -> None:
    c       = SLOT_COLOURS.get(slot, WHITE)
    label   = SLOT_LABELS.get(slot, slot)
    inner   = result.get("result", {})
    decision= inner.get("decision", {}) if isinstance(inner, dict) else {}

    depth      = inner.get("depth", "?")
    agent_type = inner.get("agent_type", "?")
    latency    = inner.get("latency_ms", 0)
    status     = result.get("status", "?")

    depth_sym = {"simple": "●", "medium": "◆", "complex": "★"}.get(depth, "?")

    print(f"\n  {c}{'━'*62}{RESET}")
    print(f"  {c}{BOLD}  RESULT — {label}{RESET}")
    print(f"  {c}{'━'*62}{RESET}")
    print(f"  {c}  Status     :{RESET} {GREEN if status=='success' else RED}{status.upper()}{RESET}")
    print(f"  {c}  Depth      :{RESET} {depth_sym} {BOLD}{depth.upper()}{RESET}")
    print(f"  {c}  Agent tier :{RESET} {agent_type}")
    print(f"  {c}  Latency    :{RESET} {latency:.1f} ms  {DIM}(wall: {wall_ms:.1f} ms){RESET}")

    if agent_type in ("reactive", "hybrid_router"):
        print(f"  {c}  Action     :{RESET} {decision.get('action','?')}")
        print(f"  {c}  Unit       :{RESET} {decision.get('unit','?')}")

    elif agent_type == "routing":
        rationale = decision.get("rationale", "")[:110]
        print(f"  {c}  Action     :{RESET} {decision.get('action','?')}")
        print(f"  {c}  Rationale  :{RESET} {rationale}")

    elif agent_type == "deliberative":
        steps    = decision.get("steps", 0)
        summary  = decision.get("plan_summary", "")
        detail   = decision.get("plan_detail", {})
        print(f"  {c}  Steps done :{RESET} {steps}")
        print(f"  {c}  Summary    :{RESET} {summary}")
        if detail:
            print()
            for node_id, outcome in detail.items():
                sym = f"{GREEN}✓{RESET}" if outcome["status"] == "done" else f"{RED}✗{RESET}"
                out = (outcome.get("output") or "")[:100]
                print(f"  {c}    {sym} {BOLD}{node_id}{RESET}  {DIM}({outcome['latency_ms']:.0f}ms){RESET}")
                if out:
                    print(f"  {c}      ↳ {DIM}{out}{RESET}")
    print()


# ── Main scenario ─────────────────────────────────────────────────────────────

async def main():
    logging.basicConfig(level=logging.WARNING)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"\n{RED}{BOLD}ERROR:{RESET} OPENAI_API_KEY not set.\n"
              f"  export OPENAI_API_KEY='sk-...'  and re-run.\n")
        sys.exit(1)

    from soma_v2 import SOMASwarm
    from soma_v2.core.tools import RiskLevel

    # ── 3 slots: one per simultaneous incident ──────────────────────────────
    swarm = SOMASwarm(
        model="openai/gpt-4o-mini",
        slots=3,              # exactly 3 — one for each concurrent task
        trace_dir=None,
        persist_dir="./soma_multiagent_memory",
    )

    # Register tools the deliberative agent can use
    swarm.register_tool("check_db_replication", check_db_replication,
                        "Check database replica status and lag")
    swarm.register_tool("promote_replica",      promote_replica,
                        "Promote a replica to primary and update DNS",
                        risk=RiskLevel.HIGH)
    swarm.register_tool("page_oncall_team",     page_oncall_team,
                        "Send PagerDuty alert to on-call engineer",
                        risk=RiskLevel.MEDIUM)
    swarm.register_tool("check_api_gateway",   check_api_gateway,
                        "Check API gateway error rates and upstream health")
    swarm.register_tool("check_lb_health",     check_lb_health,
                        "Check load balancer node health and throughput")

    # ── Scenario setup ──────────────────────────────────────────────────────
    print(f"\n{BOLD}{BLUE}{'═'*64}{RESET}")
    print(f"{BOLD}{WHITE}   SOMA V2 — 3-Agent Concurrent Incident Response{RESET}")
    print(f"{BOLD}{BLUE}{'═'*64}{RESET}\n")

    print(f"  Scenario  : {WHITE}3:00 AM — three alerts fire simultaneously{RESET}")
    print(f"  Agents    : {WHITE}3 slots (SUPERVISOR + 2 PEERs){RESET}")
    print(f"  Model     : {WHITE}openai/gpt-4o-mini{RESET}\n")

    separator("THE INCIDENT")

    incidents = [
        {
            "slot_hint": "agent_0",
            "label":     "CRITICAL — Database primary is unreachable",
            "task":      (
                "PRIMARY DATABASE db-01 is unreachable — all writes are failing. "
                "Check replication status, promote the replica to primary, "
                "and page the on-call team immediately."
            ),
            "urgency":      "emergency",
            "forced_depth": "complex",
        },
        {
            "slot_hint": "agent_1",
            "label":     "HIGH — API gateway returning 503s",
            "task":      (
                "API gateway is returning 503 errors to 78% of requests. "
                "Check the gateway health and decide next steps."
            ),
            "urgency":      "high",
            "forced_depth": None,
        },
        {
            "slot_hint": "agent_2",
            "label":     "LOW — Load balancer heartbeat check",
            "task":      "Health check on load balancer LB-01",
            "urgency":      "low",
            "forced_depth": None,
        },
    ]

    for inc in incidents:
        c = SLOT_COLOURS[inc["slot_hint"]]
        print(f"  {c}{BOLD}[{inc['slot_hint']}]{RESET}  {inc['label']}")
        print(f"  {DIM}         → \"{inc['task'][:80]}{'…' if len(inc['task'])>80 else ''}\"  "
              f"urgency={inc['urgency']}{RESET}\n")

    separator("ALL 3 DISPATCHED SIMULTANEOUSLY")
    print(f"  {DIM}asyncio.gather() fires all 3 at once.{RESET}")
    print(f"  {DIM}Each goes to a different slot — they run in parallel.{RESET}\n")

    # Track start times per slot for wall-clock measurement
    start_times: dict[str, float] = {}
    finish_times: dict[str, float] = {}

    async def dispatch_and_track(inc: dict) -> dict:
        slot_hint = inc["slot_hint"]
        log(slot_hint, f"{BOLD}RECEIVED{RESET}  urgency={inc['urgency']}")
        start_times[slot_hint] = time.perf_counter()

        result = await swarm.dispatch(
            inc["task"],
            urgency=inc["urgency"],
            forced_depth=inc.get("forced_depth"),
        )

        finish_times[slot_hint] = time.perf_counter()
        elapsed = (finish_times[slot_hint] - start_times[slot_hint]) * 1000

        assigned = result.get("assigned_to", "?")
        inner    = result.get("result", {})
        depth    = inner.get("depth", "?") if isinstance(inner, dict) else "?"
        log(slot_hint,
            f"{GREEN}{BOLD}DONE{RESET}  "
            f"depth={BOLD}{depth}{RESET}  "
            f"assigned_to={assigned}  "
            f"wall={elapsed:.0f}ms")
        return result, elapsed

    # ── Fire all 3 at the same time ─────────────────────────────────────────
    t_global = time.perf_counter()

    results_raw = await asyncio.gather(
        dispatch_and_track(incidents[0]),
        dispatch_and_track(incidents[1]),
        dispatch_and_track(incidents[2]),
    )

    global_elapsed = (time.perf_counter() - t_global) * 1000

    # ── Print each agent's detailed result ──────────────────────────────────
    separator("RESULTS")

    for i, (inc, (result, wall_ms)) in enumerate(zip(incidents, results_raw)):
        print_result(inc["slot_hint"], result, wall_ms)

    # ── Timeline visualisation ───────────────────────────────────────────────
    separator("CONCURRENCY TIMELINE")
    print(f"  {DIM}Shows what each agent was doing over time{RESET}\n")

    global_start = min(start_times.values())

    # Determine relative start/end in 100ms buckets
    bar_width = 40
    total_ms  = global_elapsed

    for inc in incidents:
        slot = inc["slot_hint"]
        c    = SLOT_COLOURS[slot]
        rel_start = (start_times[slot]  - global_start) * 1000
        rel_end   = (finish_times[slot] - global_start) * 1000

        start_pct = rel_start / total_ms
        end_pct   = rel_end   / total_ms
        s_pos = int(start_pct * bar_width)
        e_pos = min(int(end_pct * bar_width), bar_width)

        bar = (
            " " * s_pos
            + "█" * max(e_pos - s_pos, 1)
            + "░" * (bar_width - e_pos)
        )

        label = SLOT_LABELS[slot]
        dur   = finish_times[slot] - start_times[slot]
        print(f"  {c}{label}{RESET}  │{c}{bar}{RESET}│  {dur*1000:.0f}ms")

    print(f"\n  {DIM}  {'0ms':>6}{'':>{bar_width-10}}{'total: ' + f'{global_elapsed:.0f}ms':>14}{RESET}")
    print(f"\n  {BOLD}All 3 agents ran concurrently.{RESET}  "
          f"Total wall clock: {BOLD}{global_elapsed:.0f}ms{RESET}  "
          f"{DIM}(longest single task would have been "
          f"~{max((finish_times[s]-start_times[s])*1000 for s in finish_times):.0f}ms alone){RESET}\n")

    # ── Stats ────────────────────────────────────────────────────────────────
    separator("SWARM STATS")
    stats = swarm.stats
    d     = stats["director"]
    m     = stats["memory"]["hot"]

    print(f"  Tasks assigned   : {BOLD}{d['tasks_assigned']}{RESET}  "
          f"(3 tasks → 3 agents)")
    print(f"  Tasks failed     : {d['tasks_failed']}")
    print(f"  Overflow routes  : {d['overflow_routes']}")
    print(f"  Cache hit rate   : {BOLD}{m['hit_rate']}{RESET}")
    print(f"  Episodes stored  : {stats['memory']['cold']['episode_count']}")

    print(f"\n  {GREEN}{BOLD}Incident response complete.{RESET}  "
          f"All 3 agents resolved their tasks simultaneously.\n")

    await swarm.close()


if __name__ == "__main__":
    asyncio.run(main())
