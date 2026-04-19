import json
import os
import sys
import time
from datetime import datetime
from collections import defaultdict

# ANSI Colors for premium look
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}   _____ ____  __  ___ ___     _   __ ___ 
  / ___// __ \/  |/  //   |   | | / /|__ \\
  \__ \/ / / / /|_/ // /| |   | |/ / __/ /
 ___/ / /_/ / /  / // ___ |   |  _// __/ 
/____/\____/_/  /_//_/  |_|   |_| /____/ {Colors.ENDC}
{Colors.BLUE}{Colors.BOLD}   >>> Swarm OS Deep Observability Report <<<{Colors.ENDC}
    """
    print(banner)

def format_time(ts):
    return datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f')[:-3]

def parse_trace(filepath):
    events = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except:
                continue
    return events

def generate_report(trace_dir):
    if not os.path.exists(trace_dir):
        print(f"{Colors.RED}Error: Trace directory '{trace_dir}' not found.{Colors.ENDC}")
        return

    # Find the most recent trace file
    files = [f for f in os.listdir(trace_dir) if f.endswith('.jsonl')]
    if not files:
        print(f"{Colors.YELLOW}No trace files found in {trace_dir}{Colors.ENDC}")
        return

    latest_file = sorted(files)[-1]
    filepath = os.path.join(trace_dir, latest_file)
    
    print_banner()
    print(f"{Colors.BOLD}Source:{Colors.ENDC} {latest_file}")
    print(f"{Colors.BOLD}Target:{Colors.ENDC} {os.path.abspath(filepath)}")
    print("-" * 60)

    all_events = parse_trace(filepath)
    
    # Aggregates
    tasks = {}  # task_id -> task_data
    stats = {
        "success": 0,
        "failed": 0,
        "total_latency": 0.0,
        "routing": defaultdict(int),
        "tools": defaultdict(int),
        "agents": defaultdict(lambda: {"tasks": 0, "load": 0}),
        "llm": {"attempts": 0, "timeouts": 0, "errors": 0, "total_lat": 0.0},
        "cache": {"hits": 0, "misses": 0, "levels": defaultdict(int)},
        "negotiation": {"proposals": 0, "accepted": 0, "rejected": 0}
    }

    # Chronological Log
    print(f"\n{Colors.UNDERLINE}MISSION LOG (Chronological){Colors.ENDC}")
    
    for ev in all_events:
        ts = format_time(ev['timestamp'])
        ev_type = ev['event']
        tid = ev.get('task_id', 'N/A')
        
        # Color coding by event type
        color = Colors.ENDC
        if ev_type == "task_assigned": color = Colors.BLUE
        elif ev_type == "task_start": color = Colors.BLUE
        elif ev_type == "kernel_dispatch": color = Colors.YELLOW
        elif ev_type == "tool_call": color = Colors.CYAN
        elif ev_type == "llm_attempt": color = Colors.BOLD
        elif ev_type == "a2a_msg": color = Colors.YELLOW
        elif ev_type == "task_end": 
            color = Colors.GREEN if ev.get('status') == "success" else Colors.RED

        # Log line
        print(f"[{Colors.BOLD}{ts}{Colors.ENDC}] {color}{ev_type.upper():<16}{Colors.ENDC} | {tid[:10]:<10} | ", end="")
        
        if ev_type == "task_assigned":
            winner = ev.get('winner_id', '?')
            load = ev.get('winner_load', 0)
            role = ev.get('winner_role', '?')
            stats['agents'][winner]['tasks'] += 1
            stats['agents'][winner]['load'] = load
            print(f"Assign -> {Colors.BOLD}{winner}{Colors.ENDC} ({role}, load={load})")
        elif ev_type == "task_start":
            print(f"Ingress: '{ev.get('event_text', '')[:40]}...'")
        elif ev_type == "kernel_dispatch":
            depth = ev.get('depth', '?')
            stats['routing'][depth] += 1
            print(f"Routing: {Colors.BOLD}{depth}{Colors.ENDC} (p={ev.get('depth_prob', 0):.2f})")
        elif ev_type == "tool_call":
            tool = ev.get('tool', '?')
            stats['tools'][tool] += 1
            print(f"Tool: {Colors.CYAN}{tool}{Colors.ENDC}")
        elif ev_type == "llm_attempt":
            stats['llm']['attempts'] += 1
            stats['llm']['total_lat'] += ev.get('latency_ms', 0)
            status = ev.get('status')
            if status == "timeout": stats['llm']['timeouts'] += 1
            elif status == "error": stats['llm']['errors'] += 1
            print(f"LLM {ev.get('task_type')} {ev.get('attempt')}: {Colors.BOLD}{status.upper()}{Colors.ENDC} in {ev.get('latency_ms', 0):.1f}ms")
        elif ev_type == "cache_query":
            hit = ev.get('hit', False)
            level = ev.get('level', 'miss')
            if hit: 
                stats['cache']['hits'] += 1
                stats['cache']['levels'][level] += 1
            else: 
                stats['cache']['misses'] += 1
            print(f"Cache {level.upper()}: {Colors.GREEN if hit else Colors.YELLOW}{'HIT' if hit else 'MISS'}{Colors.ENDC}")
        elif ev_type == "a2a_msg":
            mtype = ev.get('msg_type', '?')
            if "STEP_PROPOSAL" in mtype: stats['negotiation']['proposals'] += 1
            elif "STEP_ACCEPT" in mtype: stats['negotiation']['accepted'] += 1
            elif "STEP_REJECT" in mtype: stats['negotiation']['rejected'] += 1
            print(f"A2A {Colors.BOLD}{mtype}{Colors.ENDC}: {ev.get('sender')} -> {ev.get('recipient')}")
        elif ev_type == "task_end":
            status = ev.get('status', 'unknown')
            latency = ev.get('latency_ms', 0)
            if status == "success": stats['success'] += 1
            else: stats['failed'] += 1
            stats['total_latency'] += latency
            print(f"Outcome: {color}{status.upper()}{Colors.ENDC} in {latency:.1f}ms")
        else:
            print(f"Data: {ev}")

    # Summary Section
    total_tasks = stats['success'] + stats['failed']
    avg_latency = stats['total_latency'] / max(total_tasks, 1)
    
    print("\n" + "=" * 60)
    print(f"{Colors.HEADER}{Colors.BOLD}MISSION SUMMARY{Colors.ENDC}")
    print("=" * 60)
    
    # Success Rate & Latency
    sr = (stats['success'] / max(total_tasks, 1)) * 100
    sr_color = Colors.GREEN if sr > 80 else (Colors.YELLOW if sr > 50 else Colors.RED)
    print(f"• {Colors.BOLD}Success Rate:{Colors.ENDC}   {sr_color}{sr:.1f}%{Colors.ENDC} ({stats['success']}/{total_tasks} tasks)")
    print(f"• {Colors.BOLD}Avg Latency:{Colors.ENDC}    {Colors.BOLD}{avg_latency:.1f}ms{Colors.ENDC}")
    
    # Per-Agent Distribution
    print(f"\n• {Colors.BOLD}Agent Distribution:{Colors.ENDC}")
    for agent, adata in sorted(stats['agents'].items()):
        print(f"  - {agent:<12}: {adata['tasks']} tasks (final load: {adata['load']})")

    # Routing Tier Distribution
    print(f"\n• {Colors.BOLD}Tier Distribution:{Colors.ENDC}")
    for tier, count in sorted(stats['routing'].items()):
        perc = (count / max(total_tasks, 1)) * 100
        print(f"  - {tier:<12}: {perc:>5.1f}% ({count} hits)")

    # Cache Performance
    total_queries = stats['cache']['hits'] + stats['cache']['misses']
    if total_queries:
        hit_rate = (stats['cache']['hits'] / total_queries) * 100
        print(f"\n• {Colors.BOLD}Semantic Cache:{Colors.ENDC}  {Colors.GREEN if hit_rate > 50 else Colors.YELLOW}{hit_rate:.1f}% Hit Rate{Colors.ENDC}")
        for lvl, count in stats['cache']['levels'].items():
            print(f"  - {lvl:<12}: {count} hits")

    # Negotiation Performance
    if stats['negotiation']['proposals']:
        print(f"\n• {Colors.BOLD}Negotiation:{Colors.ENDC}")
        print(f"  - Proposals   : {stats['negotiation']['proposals']}")
        print(f"  - Accepted    : {Colors.GREEN}{stats['negotiation']['accepted']}{Colors.ENDC}")
        print(f"  - Rejected    : {Colors.RED}{stats['negotiation']['rejected']}{Colors.ENDC}")

    # LLM Performance
    if stats['llm']['attempts']:
        avg_llm = stats['llm']['total_lat'] / stats['llm']['attempts']
        print(f"\n• {Colors.BOLD}LLM Reliability:{Colors.ENDC}")
        print(f"  - Avg Call    : {avg_llm:.1f}ms")
        print(f"  - Timeouts    : {Colors.RED if stats['llm']['timeouts'] else Colors.GREEN}{stats['llm']['timeouts']}{Colors.ENDC}")
        print(f"  - Errors      : {Colors.RED if stats['llm']['errors'] else Colors.GREEN}{stats['llm']['errors']}{Colors.ENDC}")

    # Tools Summary
    if stats['tools']:
        print(f"\n• {Colors.BOLD}Tool Actuation:{Colors.ENDC}")
        for tool, count in sorted(stats['tools'].items()):
            print(f"  - {tool:<12}: {count} calls")
    
    print("=" * 60)

if __name__ == "__main__":
    path = "./soma_traces"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    generate_report(path)
