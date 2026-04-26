# SOMA V2

**S**warm **O**rchestration **M**ulti-Agent **A**rchitecture — Version 2.

A lightweight, production-ready multi-agent kernel with:
- **Three-tier agent routing** (Reactive → Routing → Deliberative)
- **DAG-based planning** with backtracking and plan memoization
- **Hierarchical memory** (in-process LRU + ChromaDB episodic recall)
- **OpenAI-native** via the official `openai` Python SDK (also supports Groq, DeepSeek, Ollama)
- **FastAPI REST interface** for remote task dispatch

---

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import asyncio
from soma_v2 import SOMASwarm

async def main():
    swarm = SOMASwarm(model="openai/gpt-4o-mini", slots=3)
    result = await swarm.dispatch("Analyse our Q1 sales data and identify underperforming regions")
    print(result)
    await swarm.close()

asyncio.run(main())
```

Set your API key before running:

```bash
export OPENAI_API_KEY="sk-..."
python example_quickstart.py
```

---

## Supported Models

| Model string | Provider | Env var required |
|---|---|---|
| `openai/gpt-4o-mini` | OpenAI | `OPENAI_API_KEY` |
| `openai/gpt-4o` | OpenAI | `OPENAI_API_KEY` |
| `groq/llama3-8b-8192` | Groq | `GROQ_API_KEY` |
| `deepseek/deepseek-chat` | DeepSeek | `DEEPSEEK_API_KEY` |
| `ollama/mistral` | Local Ollama | — |

---

## Agent Routing

Each task is classified by the `DepthClassifier` (keyword rules + optional ML model) into one of three depths:

| Depth | Agent | LLM calls | Use case |
|---|---|---|---|
| `simple` | `ReactiveAgent` | 0 | Status checks, pings, heartbeats |
| `medium` | `RoutingAgent` | 1 | Routing decisions, triage |
| `complex` | `DeliberativeAgent` | N | Multi-step planning, analysis |

You can override the depth classification:

```python
result = await swarm.dispatch("...", forced_depth="complex")
```

---

## Registering Tools

Agents can call registered async functions during plan execution:

```python
async def search_database(query: str) -> str:
    ...  # your implementation

swarm.register_tool(
    name="search_database",
    func=search_database,
    description="Search the internal knowledge base",
    risk=RiskLevel.LOW,
)
```

The tool name appears in the planner prompt; the LLM sets `"command": "search_database"` on a step node to invoke it.

---

## Human-in-the-Loop

Steps marked `"interrupt": true` in the plan pause execution until approved:

```python
suspended = swarm.get_suspended_tasks()
# {"agent_0": "Deploy updated configuration to all production nodes"}

swarm.approve("agent_0")   # resumes execution
```

---

## REST API

Start the server:

```bash
uvicorn soma_v2.api.server:app --reload
```

Or set a custom model:

```bash
SOMA_MODEL=openai/gpt-4o uvicorn soma_v2.api.server:app --reload
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/mission` | Dispatch a task (async) |
| `GET` | `/mission/{id}` | Poll task result |
| `GET` | `/suspended` | List HITL-paused tasks |
| `POST` | `/approve/{agent_id}` | Resume a suspended task |
| `GET` | `/metrics` | Swarm statistics |
| `GET` | `/health` | Liveness probe |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Architecture

```
SOMASwarm (main.py)
  └─ AgentDirector (core/director.py)        — pool manager
       ├─ AgentSlot × N
       │    └─ V2Kernel (core/kernel.py)     — per-agent kernel
       │         ├─ DepthClassifier          — task complexity routing
       │         ├─ ReactiveAgent            — rule-based (0 LLM calls)
       │         ├─ RoutingAgent             — single LLM call
       │         └─ DeliberativeAgent        — DAG planner
       │              └─ PlanExecutor        — wave-ordered DAG execution
       ├─ HierarchicalMemory                 — hot (LRU) + cold (ChromaDB)
       ├─ A2ABus                             — in-process message bus
       └─ TelemetryStore                     — JSONL trace writer
```

---

## Project Structure

```
soma/
├── src/soma_v2/
│   ├── main.py              # SOMASwarm — top-level API
│   ├── connectors.py        # OpenAI SDK LLM connector
│   ├── agents/
│   │   ├── reactive.py      # D1 rule-based agent
│   │   ├── routing.py       # D2 single-LLM agent
│   │   └── deliberative.py  # D3 DAG planner
│   ├── core/
│   │   ├── kernel.py        # Per-agent kernel
│   │   ├── director.py      # Pool manager
│   │   ├── planner.py       # PlanGraph + PlanExecutor
│   │   ├── depth_classifier.py
│   │   ├── a2a.py           # In-process message bus
│   │   ├── blackboard.py    # Resource locking
│   │   ├── broker.py        # Negotiation broker
│   │   ├── telemetry.py     # JSONL tracer
│   │   └── tools.py         # Tool registry
│   ├── memory/
│   │   ├── hierarchical.py  # Memory facade
│   │   ├── hot.py           # LRU cache
│   │   └── cold.py          # ChromaDB episodic store
│   └── api/
│       └── server.py        # FastAPI REST server
├── example_quickstart.py
├── pyproject.toml
└── requirements.txt
```

---

## License

MIT
