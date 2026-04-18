# SOMA V2 — Urban Swarm OS

<div align="center">

**A memory-augmented, negotiation-aware operating system for autonomous drone swarms.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-LaTeX-red)](paper/main.tex)

</div>

---

## What Is This?

SOMA V2 is a **multi-agent kernel** for coordinating physical drone swarms. It does things that general-purpose agent frameworks (AutoGen, LangGraph) fundamentally cannot:

| Capability | AutoGen | LangGraph | **SOMA V2** |
|---|:---:|:---:|:---:|
| Semantic plan reuse (cache) | ❌ | ❌ | ✅ **>16,000× speedup** |
| Cross-domain zero-shot transfer | ❌ | ❌ | ✅ **50% hit rate** |
| Inter-agent resource negotiation | ❌ | ❌ | ✅ **36.1% latency reduction** |
| Failure-driven self-repair | ❌ | ❌ | ✅ **0 manual interventions** |
| Physical actuator bridge `[CMD]` | ❌ | ❌ | ✅ |
| 500-task thundering herd | ❌ | ❌ | ✅ **500/500, 44 t/s** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AgentDirector                          │
│   A2A Bus  ·  ResourceBlackboard  ·  NegotiationBroker      │
└────────────────────┬────────────────────────────────────────┘
                     │ dispatch (load-balanced)
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    AgentSlot    AgentSlot   AgentSlot ...
         │
    V2Kernel  ←── DepthClassifier (simple / medium / complex)
         │
    ┌────┴──────────────────┐
    │   ReactiveAgent  D1   │  Rule-based, 0 LLM calls
    │   RoutingAgent   D2   │  Single LLM call
    │   DeliberativeAgent D3│  Multi-step planner + cache
    └───────────────────────┘
         │
    HierarchicalMemory
    ├── HotMemory   (LRU, <0.1ms)  ← L1
    └── ColdMemory  (ChromaDB)     ← L2 semantic search
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/urban-swarm-os-v2
cd urban-swarm-os-v2
pip install -e ".[dev]"
```

### 2. Run the Capability Showcase (no Ollama needed)

```bash
python scratch/soma_showcase.py
```

Expected output:
```
CAP 1: Semantic Plan Reuse     → >16,000x speedup (cold vs warm)
CAP 2: Cross-Domain Transfer   → 50% zero-shot hit rate
CAP 3: Resource Negotiation    → 36.1% latency reduction
CAP 4: Failure-Driven Replan   → self-healed in <200ms
CAP 5: Concurrency Stress      → 89 tasks/s, 100% success
```

### 3. Run the Adversarial Attack Suite

```bash
python scratch/adversarial_stress.py
```

### 4. Run with a Real LLM (Ollama)

```bash
# Install Ollama: https://ollama.com
ollama pull qwen2.5:3b
python scratch/soma_showcase.py --ollama
```

---

## Key Results

| Metric | Value |
|---|---|
| Cache cold → warm speedup | **>16,000×** (16.3s → <1ms, `qwen2.5:3b`) |
| Negotiation latency reduction | **36.1%** (4.15s vs 6.49s) |
| Thundering herd (500 tasks) | **500/500** success, 44 tasks/s |
| Agent saturation point | **16 agents** (84 tasks/s peak) |
| Adversarial attacks survived | **6/8** |
| Prompt injection to hardware | **0** bad commands |
| Cross-domain zero-shot hits | **50%** |

---

## Run All Benchmarks

```bash
# Operating envelope (cache hit rate vs repetition rate)
python scratch/novel_task_stream.py

# Agent scale ceiling (1 → 64 agents)
python scratch/saturation_sweep.py

# Multi-model generalization
python scratch/multimodel_test.py --stub       # no Ollama
python scratch/multimodel_test.py              # auto-detects Ollama models
```

---

## Project Structure

```
urban-swarm-os-v2/
├── src/soma_v2/
│   ├── core/
│   │   ├── kernel.py          # V2Kernel — depth classify + dispatch
│   │   ├── director.py        # AgentDirector — pool + A2A bus
│   │   ├── negotiation.py     # NegotiationBroker — step delegation
│   │   ├── a2a.py             # A2A message bus + ResourceBlackboard
│   │   ├── actuators.py       # [CMD] token → drone command bridge
│   │   ├── planner.py         # DAG plan executor
│   │   └── depth_classifier.py# ML + keyword depth classifier
│   ├── agents/
│   │   ├── reactive.py        # D1: rule-based, 0 LLM calls
│   │   ├── routing.py         # D2: single LLM call + action parsing
│   │   └── deliberative.py    # D3: multi-step planner + semantic cache
│   └── memory/
│       ├── hierarchical.py    # Two-tier memory facade
│       ├── hot.py             # L1: LRU + TTL (<0.1ms)
│       └── cold.py            # L2: ChromaDB semantic store
├── scratch/
│   ├── soma_showcase.py       # 5-capability showcase
│   ├── adversarial_stress.py  # 8-attack robustness suite
│   ├── novel_task_stream.py   # Operating envelope benchmark
│   ├── saturation_sweep.py    # Agent scale ceiling
│   └── multimodel_test.py     # Multi-model generalization
├── paper/
│   ├── main.tex               # Full LaTeX paper
│   └── soma_v2_paper_final.zip# Ready for Overleaf
└── experiments/               # Historical benchmark results
```

---

## Using SOMA V2 in Your Own Code

```python
import asyncio
from soma_v2.core.director import AgentDirector
from soma_v2.memory.hierarchical import HierarchicalMemory

async def main():
    # Plug in any LLM backend
    async def my_llm(label: str, prompt: str) -> str:
        # call Ollama, OpenAI, Anthropic, etc.
        return "..."

    director = AgentDirector(
        llm_callback=my_llm,
        memory=HierarchicalMemory(),
    )
    director.add_slot("agent_1", role="SUPERVISOR", capacity=8)
    director.add_slot("agent_2", role="PEER",       capacity=8)

    await director.start()

    result = await director.assign(
        "Coordinate 3-drone rescue mission in Sector 7",
        urgency="high",
    )
    print(result)

    await director.stop()

asyncio.run(main())
```

---

## Requirements

- Python 3.10+
- `sentence-transformers` (L2 semantic search)
- `chromadb` (episodic cold store)
- `scikit-learn` (depth classifier)
- `aiohttp` (Ollama interface)
- `matplotlib` (benchmark plots)
- Optional: [Ollama](https://ollama.com) + `qwen2.5:3b` for real LLM inference

---

## Paper

The full research paper is in [`paper/main.tex`](paper/main.tex).  
Compiled PDF available via [Overleaf](https://overleaf.com) — upload `paper/soma_v2_paper_final.zip`.

**Key sections:**
- §3 Empirical Evaluation (cache, negotiation, stress, adversarial)
- §3.8 Adversarial Robustness (8 attack categories)
- §3.9 Scale Ceiling Analysis (1–64 agents)
- §3.10 Operating Envelope (cache break-even at 25–50% task repetition)
- §4 Limitations (honest cold-start penalty, domain specialization)

---

## License

MIT — use freely, cite if you publish.
