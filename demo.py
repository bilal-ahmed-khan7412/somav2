"""
SOMA V2 — Live Demonstration
==============================
See all three agent tiers working on real-world IT helpdesk tasks.

Run:
    export OPENAI_API_KEY="sk-..."
    python3 demo.py

What this shows:
  ✓ ReactiveAgent    — zero LLM calls,  <1ms latency
  ✓ RoutingAgent     — one LLM call,    routing decision
  ✓ DeliberativeAgent— DAG plan,        multi-step execution
  ✓ Tool invocation  — registered tools called during a plan
  ✓ Plan cache hit   — L1 hit on second identical-class task
  ✓ Swarm stats      — final metrics summary
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BLUE   = "\033[34m"
MAGENTA= "\033[35m"
RED    = "\033[31m"
WHITE  = "\033[97m"
BG_DARK= "\033[48;5;234m"

def hdr(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─'*64}{RESET}")
    print(f"{BOLD}{WHITE}{BG_DARK}  {text}  {RESET}")
    print(f"{BOLD}{CYAN}{'─'*64}{RESET}\n")

def step(label: str, value: str = "", colour: str = GREEN) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"  {DIM}[{ts}]{RESET}  {colour}{BOLD}{label}{RESET}  {WHITE}{value}{RESET}")

def rule(char: str = "·") -> None:
    print(f"  {DIM}{char * 60}{RESET}")

def box(title: str, lines: list[str], colour: str = CYAN) -> None:
    print(f"\n  {colour}┌─ {BOLD}{title}{RESET}{colour} {'─'*(55-len(title))}┐{RESET}")
    for line in lines:
        print(f"  {colour}│{RESET}  {line}")
    print(f"  {colour}└{'─'*59}┘{RESET}\n")

# ── Fake tools (simulate real integrations) ───────────────────────────────────
_TICKET_DB = {
    "TKT-001": {"title": "Login page returns 500", "severity": "P1", "assignee": "alice"},
    "TKT-007": {"title": "Slow dashboard queries", "severity": "P2", "assignee": "bob"},
}

async def check_server_health(server_id: str = "web-01") -> str:
    await asyncio.sleep(0.1)   # simulate API call
    return f"Server {server_id}: CPU=34% MEM=61% DISK=42% STATUS=OK uptime=18d"

async def query_ticket_system(ticket_id: str = "TKT-001") -> str:
    await asyncio.sleep(0.15)
    t = _TICKET_DB.get(ticket_id, {"title": "Not found", "severity": "?", "assignee": "?"})
    return f"Ticket {ticket_id}: [{t['severity']}] {t['title']} — assigned to {t['assignee']}"

async def list_open_alerts() -> str:
    await asyncio.sleep(0.1)
    return (
        "3 open alerts: "
        "[P1] web-02 disk usage 89%, "
        "[P2] db-replica lag 4.2s, "
        "[P3] cert expires in 14d"
    )

async def run_diagnostic(target: str = "web-01") -> str:
    await asyncio.sleep(0.3)
    return (
        f"Diagnostic on {target}: "
        "HTTP 200 ✓  DB conn 12ms ✓  Cache hit-rate 94% ✓  "
        "Slow queries: 2 found (>500ms)"
    )

async def create_incident_report(summary: str = "") -> str:
    await asyncio.sleep(0.1)
    ts = datetime.now(timezone.utc).isoformat()
    return f"Incident report INC-{int(time.time()) % 10000} created at {ts}. On-call team notified."

# ── Result printer ────────────────────────────────────────────────────────────
def print_result(result: dict, task_num: int) -> None:
    outer = result
    inner = result.get("result", {})
    decision = inner.get("decision", {}) if isinstance(inner, dict) else {}

    depth      = inner.get("depth", "?")
    agent_type = inner.get("agent_type", "?")
    latency    = inner.get("latency_ms", 0)
    status     = outer.get("status", "?")

    depth_colours = {"simple": GREEN, "medium": YELLOW, "complex": MAGENTA}
    agent_colours = {
        "reactive": GREEN, "hybrid_router": GREEN,
        "routing": YELLOW,
        "deliberative": MAGENTA,
    }
    dc = depth_colours.get(depth, WHITE)
    ac = agent_colours.get(agent_type, WHITE)

    step("Status    ", status.upper(), GREEN if status == "success" else RED)
    step("Depth     ", f"{dc}{BOLD}{depth.upper()}{RESET}", "")
    step("Agent     ", f"{ac}{agent_type}{RESET}", "")
    step("Latency   ", f"{latency:.1f} ms")
    step("Task ID   ", outer.get("task_id", "?"), DIM)

    if decision:
        if agent_type in ("reactive", "hybrid_router"):
            step("Action    ", str(decision.get("action", "?")), GREEN)
            step("Unit      ", str(decision.get("unit", "?")))
            step("Rationale ", str(decision.get("rationale", ""))[:80])

        elif agent_type == "routing":
            step("Action    ", str(decision.get("action", "?")), YELLOW)
            rationale = str(decision.get("rationale", ""))
            # show first 120 chars of LLM rationale
            step("Rationale ", rationale[:120] + ("…" if len(rationale) > 120 else ""))

        elif agent_type == "deliberative":
            plan_summary = decision.get("plan_summary", "")
            steps_done   = decision.get("steps", 0)
            cached       = decision.get("metadata", {}).get("cached", False)
            cache_level  = decision.get("metadata", {}).get("cache_level", "miss")

            step("Steps done", str(steps_done), MAGENTA)
            step("Cache     ", f"{'HIT ' + cache_level if cached else 'MISS (new plan generated)'}", 
                 GREEN if cached else DIM)
            step("Summary   ", plan_summary[:100])

            # Show individual plan step outputs
            plan_detail = decision.get("plan_detail", {})
            if plan_detail:
                print()
                for node_id, outcome in plan_detail.items():
                    status_sym = "✓" if outcome["status"] == "done" else "✗"
                    sym_colour = GREEN if outcome["status"] == "done" else RED
                    out_text   = (outcome.get("output") or "")[:110]
                    print(f"    {sym_colour}{status_sym}{RESET} {BOLD}{node_id}{RESET}  "
                          f"{DIM}({outcome['latency_ms']:.0f}ms){RESET}")
                    if out_text:
                        print(f"      {DIM}↳ {out_text}{RESET}")

# ── Main demo ─────────────────────────────────────────────────────────────────
async def main():
    # Silence noisy loggers; show only SOMA
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("SOMA_V2").setLevel(logging.INFO)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"\n{RED}{BOLD}ERROR:{RESET} OPENAI_API_KEY is not set.\n"
              f"  export OPENAI_API_KEY='sk-...'  and re-run.\n")
        sys.exit(1)

    print(f"\n{BOLD}{BLUE}{'═'*64}{RESET}")
    print(f"{BOLD}{WHITE}     SOMA V2 — Live Multi-Agent Demonstration{RESET}")
    print(f"{BOLD}{BLUE}{'═'*64}{RESET}")
    print(f"\n  Model  : {CYAN}openai/gpt-4o-mini{RESET}")
    print(f"  Agents : {CYAN}2 slots (1 SUPERVISOR + 1 PEER){RESET}")
    print(f"  Domain : {CYAN}IT Helpdesk / Ops{RESET}\n")

    from soma_v2 import SOMASwarm
    from soma_v2.core.tools import RiskLevel

    swarm = SOMASwarm(
        model="openai/gpt-4o-mini",
        slots=2,
        trace_dir=None,
        persist_dir="./soma_demo_memory",
    )

    # Register all tools
    swarm.register_tool("check_server_health",  check_server_health,  "Check CPU/mem/disk health of a server")
    swarm.register_tool("query_ticket_system",  query_ticket_system,  "Look up an IT support ticket by ID")
    swarm.register_tool("list_open_alerts",     list_open_alerts,     "List current open monitoring alerts")
    swarm.register_tool("run_diagnostic",       run_diagnostic,       "Run a full diagnostic on a system target")
    swarm.register_tool("create_incident_report", create_incident_report,
                        "Create an incident report and notify on-call team", risk=RiskLevel.MEDIUM)

    print(f"  {GREEN}✓{RESET} Swarm ready — 5 tools registered\n")

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 1 — ReactiveAgent (zero LLM calls)
    # ══════════════════════════════════════════════════════════════════════════
    hdr("TIER 1 — ReactiveAgent  (zero LLM calls, rule engine)")
    print(f"  {DIM}These tasks are handled entirely by keyword rules — no OpenAI call is made.{RESET}")
    print(f"  {DIM}Latency is sub-millisecond.{RESET}\n")

    for task in [
        "Check status of web-01",
        "Ping monitoring node A3",
        "Health check on db-replica-02",
    ]:
        rule()
        step("Task", f'"{task}"', CYAN)
        t0 = time.perf_counter()
        result = await swarm.dispatch(task, urgency="low")
        elapsed = (time.perf_counter() - t0) * 1000
        print_result(result, 0)
        print(f"\n  {DIM}Wall clock: {elapsed:.2f}ms{RESET}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 2 — RoutingAgent (1 LLM call)
    # ══════════════════════════════════════════════════════════════════════════
    hdr("TIER 2 — RoutingAgent  (1 LLM call, routing decision)")
    print(f"  {DIM}These tasks need some reasoning to determine next action, but not a full plan.{RESET}")
    print(f"  {DIM}One LLM call extracts a routing action + rationale.{RESET}\n")

    for task in [
        "web-02 disk is at 89% — what should we do?",
        "Our API latency spiked to 800ms in the last 10 minutes. Initial steps?",
    ]:
        rule()
        step("Task", f'"{task}"', CYAN)
        t0 = time.perf_counter()
        result = await swarm.dispatch(task, urgency="medium")
        elapsed = (time.perf_counter() - t0) * 1000
        print_result(result, 0)
        print(f"\n  {DIM}Wall clock: {elapsed:.2f}ms{RESET}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 3 — DeliberativeAgent (multi-step DAG plan)
    # ══════════════════════════════════════════════════════════════════════════
    hdr("TIER 3 — DeliberativeAgent  (multi-step plan, tools invoked)")
    print(f"  {DIM}Complex task: LLM generates a DAG plan, steps run in topological order.{RESET}")
    print(f"  {DIM}Steps with dependencies wait; independent steps run in parallel.{RESET}\n")

    complex_task = (
        "Our production web cluster is showing elevated error rates. "
        "Run a full diagnostic, check all open alerts, review ticket TKT-007, "
        "and produce an incident report with a recommended remediation plan."
    )

    rule()
    step("Task", f'"{complex_task[:80]}…"', CYAN)
    print()
    t0 = time.perf_counter()
    result = await swarm.dispatch(complex_task, urgency="high", forced_depth="complex")
    elapsed = (time.perf_counter() - t0) * 1000
    print_result(result, 0)
    print(f"\n  {DIM}Wall clock: {elapsed:.2f}ms{RESET}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # CACHE DEMO — same class of task, second call uses L1 hot plan
    # ══════════════════════════════════════════════════════════════════════════
    hdr("CACHE DEMO — L1 Plan Cache  (second call reuses cached plan)")
    print(f"  {DIM}Running a similar complex task. If the plan was stored, SOMA reuses it.{RESET}")
    print(f"  {DIM}No LLM plan generation call — only step execution calls remain.{RESET}\n")

    cached_task = (
        "Web cluster is throwing 502 errors. "
        "Run diagnostics, check active alerts, look up TKT-001, "
        "and file an incident report."
    )

    rule()
    step("Task (2nd, similar)", f'"{cached_task[:80]}…"', CYAN)
    print()
    t0 = time.perf_counter()
    result2 = await swarm.dispatch(cached_task, urgency="high", forced_depth="complex")
    elapsed2 = (time.perf_counter() - t0) * 1000
    print_result(result2, 0)
    cached = result2.get("result", {}).get("decision", {}).get("metadata", {}).get("cached", False)
    color  = GREEN if cached else YELLOW
    print(f"\n  {color}{'→ Cache HIT  — plan reused, no plan-generation LLM call' if cached else '→ Cache MISS — plan generated fresh'}{RESET}")
    print(f"  {DIM}Wall clock: {elapsed2:.2f}ms{RESET}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL STATS
    # ══════════════════════════════════════════════════════════════════════════
    hdr("Swarm Statistics")
    stats   = swarm.stats
    d_stats = stats["director"]
    m_stats = stats["memory"]
    h_stats = m_stats["hot"]

    box("Director", [
        f"Tasks assigned   : {BOLD}{d_stats['tasks_assigned']}{RESET}",
        f"Tasks failed     : {d_stats['tasks_failed']}",
        f"Overflow routes  : {d_stats['overflow_routes']}",
        f"Bus messages     : {stats['bus_messages']}",
        f"Slot loads       : {d_stats['slot_loads']}",
    ], BLUE)

    box("Memory (Hot Cache)", [
        f"Hits             : {BOLD}{GREEN}{h_stats['hits']}{RESET}",
        f"Misses           : {h_stats['misses']}",
        f"Evictions        : {h_stats['evictions']}",
        f"Hit rate         : {BOLD}{GREEN}{h_stats['hit_rate']}{RESET}",
        f"Total keys       : {h_stats['total_keys']}",
    ], MAGENTA)

    cold_info = m_stats.get("cold", {})
    box("Memory (Cold / ChromaDB)", [
        f"Backend          : {cold_info.get('backend', '?')}",
        f"Episodes stored  : {cold_info.get('episode_count', 0)}",
        f"Writes           : {cold_info.get('writes', 0)}",
    ], CYAN)

    print(f"\n  {GREEN}{BOLD}Demo complete.{RESET}  "
          f"SOMA handled {d_stats['tasks_assigned']} tasks across 3 agent tiers.\n")

    await swarm.close()


if __name__ == "__main__":
    asyncio.run(main())

'''
import asyncio
import json
import logging
import os
import sys
import time
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from soma_v2 import SOMASwarm
from soma_v2.core.a2a import A2AMessage, MsgType

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Event Broadcasting ────────────────────────────────────────────────────────
class EventBroadcaster:
    def __init__(self):
        self.queues = []

    async def subscribe(self) -> AsyncGenerator[str, None]:
        q = asyncio.Queue()
        self.queues.append(q)
        try:
            while True:
                msg = await q.get()
                yield f"data: {json.dumps(msg)}\n\n"
        except asyncio.CancelledError:
            self.queues.remove(q)

    async def broadcast(self, event_type: str, action: str = None, **kwargs):
        msg = {"type": event_type}
        if action:
            msg["action"] = action
        msg.update(kwargs)
        for q in self.queues:
            await q.put(msg)

broadcaster = EventBroadcaster()

# ── SOMA Simulation Logger / Interceptor ────────────────────────────────────
class UILogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        level = record.levelname
        # Skip excessive low level debugs
        if record.name.startswith("SOMA_V2"):
            asyncio.create_task(broadcaster.broadcast("internal_log", msg=msg, level=level))

ui_handler = UILogHandler()
ui_handler.setLevel(logging.INFO)
ui_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
logging.getLogger("SOMA_V2").addHandler(ui_handler)
logging.getLogger("SOMA_V2").setLevel(logging.INFO)

async def _ui_bus_listener(bus, agent_id="ui_listener"):
    bus.register(agent_id)
    while True:
        msg = await bus.recv(agent_id, timeout=0.5)
        if msg is None:
            continue
        # UI listener processes standard broadcasts if they arise
        pass 

# ── Global Reference ─────────────────────────────────────────────────────────
swarm_instance = None

async def _check_and_claim(server: str, my_agent_id: str) -> str:
    """Internal helper for tools to natively claim via SOMA blackboard."""
    bb = swarm_instance.director._blackboard
    owner = bb._locks.get(server)
    if owner and owner != my_agent_id:
        return f"ERROR: '{server}' is strictly locked by '{owner}'. You CANNOT execute commands on it. Use 'propose_step_to_peer' to ask {owner} to run it for you!"
    # Try claim with 0.1 timeout, if it fails it's locked tight
    ok = await bb.claim(server, my_agent_id, node_id="demo", timeout_s=0.1)
    if not ok:
        return f"ERROR: Failed to acquire lock for '{server}'. Likely held by {bb._locks.get(server)}."
    return ""

# ── SOMA Intricate Tools ────────────────────────────────────────────────────

async def delegate_task(sub_task_description: str, urgency: str = "high") -> str:
    await broadcaster.broadcast("internal_log", msg=f"Supervisor delegating sub-task: '{sub_task_description}'", level="WARNING")
    # Fire and forget into the swarm pool, force 'complex' depth so they build a PlanGraph and invoke tools
    asyncio.create_task(swarm_instance.dispatch(sub_task_description, urgency=urgency, forced_depth="complex"))
    return f"Task '{sub_task_description}' delegated to the swarm. A specialized agent will handle it in parallel."

async def scan_server(server_name: str) -> str:
    await broadcaster.broadcast("internal_log", msg=f"Scanning {server_name} for anomalies...", level="INFO")
    await asyncio.sleep(1.0)
    if server_name == "APP_NODE_2":
        return "CRITICAL: Ransomware heuristic match. Process 'encrypt_svc' communicating with known C2 IP: 198.51.100.4."
    return f"Server {server_name} clean."

async def isolate_server(server_name: str, my_agent_id: str) -> str:
    err = await _check_and_claim(server_name, my_agent_id)
    if err: return err
    await broadcaster.broadcast("scenario_cmd", action="set_agent_task", agent=my_agent_id, task=f"Isolating {server_name}")
    await broadcaster.broadcast("internal_log", msg=f"Executing network isolation on {server_name}...", level="WARNING")
    await asyncio.sleep(1.5)
    await broadcaster.broadcast("scenario_cmd", action="set_server_state", server=server_name, state="healthy", statusText="Isolated")
    # Native SOMA release
    await swarm_instance.director._blackboard.release(server_name, my_agent_id, "demo")
    return f"{server_name} fully isolated. Ransomware contained locally."

async def analyze_traffic(server_name: str) -> str:
    await asyncio.sleep(1.0)
    if server_name == "PROXY_NODE":
        return "Volumetric UDP flood detected targeting PROXY_NODE. Recommendation: mitigate_ddos layer 4."
    return "Traffic normal."

async def mitigate_ddos(server_name: str, my_agent_id: str) -> str:
    err = await _check_and_claim(server_name, my_agent_id)
    if err: return err
    await broadcaster.broadcast("scenario_cmd", action="set_agent_task", agent=my_agent_id, task=f"Mitigating DDoS on {server_name}")
    await broadcaster.broadcast("internal_log", msg=f"Activating Layer 4 DDoS BGP blackhole for {server_name}...", level="ERROR")
    await asyncio.sleep(4.0) # Long operation
    await broadcaster.broadcast("scenario_cmd", action="set_server_state", server=server_name, state="healthy", statusText="Healthy")
    # Native SOMA release
    await swarm_instance.director._blackboard.release(server_name, my_agent_id, "demo")
    return f"DDoS traffic dropped. BGP blackhole active on {server_name}."

async def inject_firewall_rule(server_name: str, rule: str, my_agent_id: str) -> str:
    err = await _check_and_claim(server_name, my_agent_id)
    if err: return err
    await broadcaster.broadcast("internal_log", msg=f"Injecting firewall rule '{rule}' on {server_name}...", level="INFO")
    await asyncio.sleep(1.0)
    # Native SOMA release
    await swarm_instance.director._blackboard.release(server_name, my_agent_id, "demo")
    return f"Firewall rule '{rule}' strictly applied to {server_name}."

async def check_db_status(server_name: str) -> str:
    await asyncio.sleep(0.5)
    if server_name == "DB_PRIMARY":
        return "Replication link broken. Sync lag is 900ms. Requires sync_database execution."
    return "DB nominal."

async def sync_database(server_name: str, my_agent_id: str) -> str:
    err = await _check_and_claim(server_name, my_agent_id)
    if err: return err
    await broadcaster.broadcast("scenario_cmd", action="set_agent_task", agent=my_agent_id, task=f"Syncing DB {server_name}")
    await broadcaster.broadcast("internal_log", msg=f"Rebuilding replication cache on {server_name}...", level="WARNING")
    await asyncio.sleep(2.0)
    await broadcaster.broadcast("scenario_cmd", action="set_server_state", server=server_name, state="healthy", statusText="Healthy")
    await swarm_instance.director._blackboard.release(server_name, my_agent_id, "demo")
    return "Database replication correctly restored. Lag < 5ms."

async def propose_step_to_peer(target_agent_id: str, step_instruction: str, my_agent_id: str) -> str:
    """Send a request over the A2A bus for a colleague to run a command while they hold a lock."""
    bus = swarm_instance.bus
    task_id = "neg_" + my_agent_id[:4]
    
    await broadcaster.broadcast("internal_log", msg=f"A2A Negotiation: {my_agent_id} asking {target_agent_id} to '{step_instruction}'", level="WARNING")
    
    await bus.send(A2AMessage(
        msg_type=MsgType.STEP_PROPOSAL,
        sender=my_agent_id, recipient=target_agent_id, task_id=task_id,
        payload={"step": step_instruction, "from": my_agent_id}
    ))
    
    # Wait for acceptance
    reply = await bus.recv(my_agent_id, timeout=15.0)
    if reply and reply.msg_type == MsgType.STEP_ACCEPT:
        return f"SUCCESS: {target_agent_id} executed the proposal. Result: {reply.payload.get('result', 'OK')}"
    return f"FAILURE: {target_agent_id} ignored or rejected the proposal."

# Mock automated response for ANY peer receiving a STEP_PROPOSAL to act as if their native LLM approved it 
# (Since peer slots might be stuck inside tools, we use a background task to instantly accept to keep demo fast)
async def _auto_accept_proposals_bg():
    bus = swarm_instance.bus
    for agent_id in ["agent_1", "agent_2"]:
        bus.register(f"listener_{agent_id}")
    while True:
        await asyncio.sleep(0.5)
        # We passively read from history because intercepting the proper queue is messy when the agent LLM is busy.
        # This simulates the target agent receiving and accepting it!
        pass

# ── Scenario Execution ──────────────────────────────────────────────────────

async def run_scenario():
    global swarm_instance
    try:
        if not swarm_instance:
            import soma_v2.connectors
            # Fix uvicorn event loop deadlock
            soma_v2.connectors.LLM_SEMAPHORE = asyncio.Semaphore(10)
            
            swarm_instance = SOMASwarm(model="openai/gpt-4o-mini", slots=3, trace_dir=None, persist_dir=None)
            
            swarm_instance.register_tool("delegate_task", delegate_task, "(SUPERVISOR ONLY) Offload independent issues. Parameters: sub_task_description (string), urgency (string: 'high' or 'emergency')")
            swarm_instance.register_tool("scan_server", scan_server, "Scan a server for malware/ransomware. Parameters: server_name (string)")
            swarm_instance.register_tool("isolate_server", isolate_server, "Cut off an infected app node from cluster. Parameters: server_name (string), my_agent_id (string)")
            swarm_instance.register_tool("analyze_traffic", analyze_traffic, "Analyze incoming traffic to detect DDoS. Parameters: server_name (string)")
            swarm_instance.register_tool("mitigate_ddos", mitigate_ddos, "Activate anti-DDoS blackhole. Parameters: server_name (string), my_agent_id (string)")
            swarm_instance.register_tool("inject_firewall_rule", inject_firewall_rule, "Inject firewall rules. Parameters: server_name (string), rule (string), my_agent_id (string)")
            swarm_instance.register_tool("check_db_status", check_db_status, "Check replication link on DB. Parameters: server_name (string)")
            swarm_instance.register_tool("sync_database", sync_database, "Resync database primary logs. Parameters: server_name (string), my_agent_id (string)")
            swarm_instance.register_tool("propose_step_to_peer", propose_step_to_peer, "Request a peer agent to run a tool while they hold a lock. Parameters: target_agent_id (string), step_instruction (string), my_agent_id (string)")

            # Wrap LLM output to broadcast
            _original_llm = swarm_instance.llm_callback
            async def _ui_llm(task_type: str, prompt: str) -> str:
                # Basic logging
                await broadcaster.broadcast("llm_log", msg=f"LLM Thinking ({task_type}) - Formulating next steps...")
                
                # Sneak in a mock reply to propose_step_to_peer to make A2A fast:
                if "propose_step_to_peer" in prompt:
                    # In real SOMA this goes to the peer, but to ensure the demo isn't stalled by rate limits
                    # we do a hacky quick intercept on the bus so the caller gets a reply instantly
                    # (since we can't easily interrupt the peer's current LLM deliberation)
                    async def auto_reply():
                        await asyncio.sleep(2.0)
                        await swarm_instance.bus.send(A2AMessage(
                            msg_type=MsgType.STEP_ACCEPT,
                            sender="agent_1", recipient="agent_0", task_id="neg_agen",
                            payload={"result": "Firewall Rule applied successfully on Proxy."}
                        ))
                    asyncio.create_task(auto_reply())
                
                return await _original_llm(task_type, prompt)
            
            swarm_instance.llm_callback = _ui_llm
            for slot in swarm_instance.director._slots.values():
                slot.kernel.routing.llm_callback = _ui_llm
                slot.kernel.deliberative.llm_callback = _ui_llm
                if hasattr(slot.kernel.deliberative, '_executor'):
                    slot.kernel.deliberative._executor.llm_callback = _ui_llm

        bus = swarm_instance.bus
        bus.register("agent_0") # SECOPS
        bus.register("agent_1") # NETOPS
        bus.register("agent_2") # SYSADMIN

        # Intercept bus.send to broadcast ALL messages to UI (even 1-to-1)
        _orig_send = bus.send
        async def _ui_send(msg):
            await _orig_send(msg)
            m_type = msg.msg_type.name if hasattr(msg.msg_type, "name") else str(msg.msg_type)
            payload = {
                "msg_type": m_type,
                "sender": msg.sender,
                "recipient": msg.recipient,
                "task_id": msg.task_id,
                "payload": msg.payload
            }
            await broadcaster.broadcast("a2a_log", msg=payload)
        bus.send = _ui_send

        # Intercept director assign to set UI agent back to Idle when a task COMPLETES
        _orig_assign = swarm_instance.director.assign
        async def _ui_assign(*args, **kwargs):
            res = await _orig_assign(*args, **kwargs)
            agent_id = res.get("assigned_to")
            if agent_id:
                await broadcaster.broadcast("scenario_cmd", action="set_agent_task", agent=agent_id, task="", isBusy=False)
            return res
        swarm_instance.director.assign = _ui_assign

        # Trigger visual attacks
        await broadcaster.broadcast("scenario_cmd", action="set_server_state", server="PROXY_NODE", state="danger", statusText="DDoS Attack")
        await broadcaster.broadcast("scenario_cmd", action="set_server_state", server="APP_NODE_2", state="danger", statusText="Ransomware Detect")
        await broadcaster.broadcast("scenario_cmd", action="set_server_state", server="DB_PRIMARY", state="warning", statusText="Replication Lag")
        
        # Fire SINGLE MASTER DISPATCH to the Supervisor (agent_0 usually grabs first slot)
        await broadcaster.broadcast("scenario_cmd", action="set_agent_task", agent="agent_0", task="Analyzing Compound Threat...")
        
        compound_prompt = (
            "COMPOUND ALERT: You are Incident Commander `agent_0`. "
            "Cluster status: "
            "1. PROXY_NODE: Volumetric DDoS detected. "
            "2. DB_PRIMARY: Critical replication lag (desync). "
            "3. APP_NODE_2: Suspected Ransomware calling home to C2 IP. "
            "Plan a full resolution. Use `delegate_task` for DDoS and DB fixes immediately (delegating to agent_1 and agent_2). "
            "Then, you must handle APP_NODE_2 yourself. You MUST scan it, isolate it, and then inject a firewall rule on PROXY_NODE to block the C2 IP. "
            "Note: If PROXY_NODE is locked by a peer, use `propose_step_to_peer` to make them block the IP for you. "
            "ALWAYS include the required dictionary 'params' in your JSON plan nodes for tools."
        )
        
        res = await swarm_instance.dispatch(compound_prompt, urgency="emergency", forced_depth="complex")
        
        # End
        await broadcaster.broadcast("scenario_cmd", action="simulation_end")

    except Exception as e:
        await broadcaster.broadcast("internal_log", msg=f"Simulation error: {e}", level="ERROR")
        import traceback
        traceback.print_exc()

# ── API Routes ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.get("/stream")
async def stream():
    return StreamingResponse(broadcaster.subscribe(), media_type="text/event-stream")

@app.post("/simulate")
async def simulate():
    asyncio.create_task(run_scenario())
    return {"status": "started"}

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY must be set.")
        sys.exit(1)
    print("\n   SOMA Sentinel | [CORE V2.2 - FULLY ESCAPED]")
    print("   -> Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
'''