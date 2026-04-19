# SOMA V2 Swarm OS - Production Guide

SOMA V2 is a high-performance heterogeneous multi-agent kernel designed for autonomous swarm operations. This guide covers the production-ready features added in the V2.x series.

## 1. Persistent Episodic Memory
SOMA V2 now supports long-term learning across sessions. By default, it uses ChromaDB for vector-based episodic recall. If ChromaDB is not available, it transparently falls back to a JSONL-based flat store.

### Configuration
```python
from soma_v2.main import SOMASwarm

swarm = SOMASwarm(
    model="ollama/qwen2.5:3b",
    persist_dir="./my_swarm_memory"  # Memory will be saved here
)
```

### Manual Sync
Cold memory writes are handled by a background worker to ensure zero latency for the agents. To ensure all data is flushed to disk before exiting:
```python
await swarm.close() # Automatically syncs memory
```

## 2. Human-In-The-Loop (HITL) Approval
For high-risk environments, SOMA V2 implements a dual-layer safety system.

### Risk Policy
Tools are registered with a `RiskLevel`:
- `RiskLevel.LOW`: Default. No intervention.
- `RiskLevel.MEDIUM`: LLM determines if an interrupt is needed.
- `RiskLevel.HIGH`: Deterministic interrupt. Approval **always** required.

```python
from soma_v2.main import SOMASwarm, RiskLevel

swarm = SOMASwarm()

@swarm.register_tool(name="detonate_payload", risk=RiskLevel.HIGH)
def detonate():
    return "Boom"

# This task will SUSPEND before calling detonate_payload
await swarm.dispatch("Navigate to target and detonate payload")
```

### Managing Suspended Tasks
```python
# Get all tasks waiting for approval
suspended = swarm.get_suspended_tasks()

for agent_id, status in suspended.items():
    print(f"Agent {agent_id} is waiting at node: {status['node_id']}")
    # Approve the action
    swarm.approve(agent_id)
```

## 3. Architecture
The V2 Kernel uses a 3-tier dispatch system:
1. **Reactive (D1)**: Zero-LLM, rule-based routing for simple tasks.
2. **Routing (D2)**: Single-LLM call for medium complexity.
3. **Deliberative (D3)**: Multi-step planner for complex missions.


## 4. Deep Observability (Structured Tracing)
SOMA V2 captures every decision point, tool call, and routing outcome in high-fidelity JSONL trace files. This allows for post-mission audit and real-time performance monitoring.

### Configuration
```python
swarm = SOMASwarm(
    trace_dir="./soma_traces" # Traces will be saved here as .jsonl files
)
```

### Trace Events
The telemetry system records the following event types:
- `task_assigned`: Initial ingress and slot selection.
- `kernel_dispatch`: Routing tier selection (D1/D2/D3) and decision rationale.
- `tool_call`: When a deliberative agent executes a physical tool.
- `task_end`: Final outcome, result data, and precise latency (ms).

Use these traces to debug "why" the swarm made a specific decision or to benchmark agent performance under load.
