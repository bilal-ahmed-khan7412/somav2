"""
SOMA V2 — SOMASwarm
====================
High-level entry point. Assembles all kernel components into a single
usable object with a clean public API.

Usage
-----
    import asyncio
    from soma_v2 import SOMASwarm

    async def main():
        swarm = SOMASwarm(model="openai/gpt-4o-mini", slots=3)
        swarm.register_tool("search_web", my_search_fn, "Search the web for information")
        result = await swarm.dispatch("Summarise the top AI papers from Q1 2025")
        print(result)
        await swarm.close()

    asyncio.run(main())
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .core.kernel    import V2Kernel
from .core.director  import AgentDirector
from .core.a2a       import A2ABus
from .core.telemetry import TelemetryStore
from .core.tools     import ToolRegistry, Tool, RiskLevel
from .memory.hierarchical import HierarchicalMemory
from .connectors import get_llm_callback

logger = logging.getLogger("SOMA_V2.MAIN")


class SOMASwarm:
    """
    SOMA V2 — Heterogeneous Multi-Agent Swarm.

    Parameters
    ----------
    model : str
        LLM model string. Format: '<provider>/<model-name>'.
        Examples:
          'openai/gpt-4o-mini'       — OpenAI (reads OPENAI_API_KEY)
          'openai/gpt-4o'            — OpenAI GPT-4o
          'groq/llama3-8b-8192'      — Groq (reads GROQ_API_KEY)
          'deepseek/deepseek-chat'   — DeepSeek (reads DEEPSEEK_API_KEY)
          'ollama/mistral'           — local Ollama (no key needed)
    slots : int
        Number of agent slots in the pool. Slot 0 is the SUPERVISOR;
        remaining slots are PEERs.
    hot_cache_cap : int
        Max entries in the in-process LRU plan cache.
    persist_dir : str
        Directory for ChromaDB cold memory persistence.
    trace_dir : str | None
        Directory to write JSONL telemetry traces. None = disabled.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        slots: int = 3,
        hot_cache_cap: int = 128,
        persist_dir: str = "./soma_memory",
        trace_dir: Optional[str] = "./soma_traces",
        **kernel_kwargs,
    ):
        self.bus       = A2ABus()
        self.telemetry = TelemetryStore(trace_dir=trace_dir)
        self.memory    = HierarchicalMemory(
            hot_capacity=hot_cache_cap,
            cold_persist=persist_dir,
        )
        self.tool_registry  = ToolRegistry()
        self.llm_callback   = get_llm_callback(model)

        self.director = AgentDirector(
            llm_callback=self.llm_callback,
            memory=self.memory,
            tool_registry=self.tool_registry,
            telemetry=self.telemetry,
            bus=self.bus,
            **kernel_kwargs,
        )

        self._setup_default_slots(slots)
        logger.info(f"SOMASwarm ready — model={model} slots={slots}")

    # ── slot initialisation ───────────────────────────────────────────────────

    def _setup_default_slots(self, count: int) -> None:
        for i in range(count):
            role = "SUPERVISOR" if i == 0 else "PEER"
            self.director.add_slot(
                slot_id=f"agent_{i}",
                role=role,
                capacity=5,
            )

    # ── tool registration ─────────────────────────────────────────────────────

    def register_tool(
        self,
        name: str,
        func,
        description: str,
        risk: RiskLevel = RiskLevel.LOW,
    ) -> None:
        """
        Register a callable tool that agents can invoke during plan execution.

        Parameters
        ----------
        name : str        — unique tool name referenced in plan steps
        func              — async callable with no required arguments
        description : str — human-readable description shown in the planner prompt
        risk : RiskLevel  — LOW / MEDIUM / HIGH (HIGH forces human approval)
        """
        self.tool_registry.register(
            Tool(name=name, func=func, description=description, risk=risk)
        )
        logger.info(f"SOMASwarm: registered tool '{name}' [RISK={risk.value}]")

    # ── task dispatch ─────────────────────────────────────────────────────────

    async def dispatch(
        self,
        task: str,
        urgency: str = "medium",
        forced_depth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dispatch a natural-language task to the swarm.

        Parameters
        ----------
        task : str
            Task description in plain English.
        urgency : str
            'low' | 'medium' | 'high' | 'emergency'
        forced_depth : str | None
            Override depth classification: 'simple' | 'medium' | 'complex'

        Returns
        -------
        Dict containing status, result, task_id, assigned_to, and latency info.
        """
        try:
            return await self.director.assign(
                task, urgency=urgency, forced_depth=forced_depth
            )
        except Exception as exc:
            logger.error(f"SOMASwarm.dispatch failed: {exc}")
            return {
                "status": "error",
                "error":  str(exc),
                "task":   task,
            }

    # ── human-in-the-loop ─────────────────────────────────────────────────────

    def get_suspended_tasks(self) -> Dict[str, str]:
        """Returns {agent_id: step_description} for all paused tasks."""
        return self.director.get_suspended_tasks()

    def approve(self, agent_id: str) -> bool:
        """
        Resume a task suspended for human approval.

        Parameters
        ----------
        agent_id : str — the agent whose task should resume

        Returns True if found and resumed, False otherwise.
        """
        return self.director.approve_task(agent_id)

    # ── stats + lifecycle ─────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "memory":       self.memory.stats,
            "director":     self.director.stats,
            "bus_messages": self.bus.message_count,
        }

    async def close(self) -> None:
        """Shutdown the swarm and flush pending memory writes."""
        await self.director.stop()
        self.memory.sync()
        logger.info("SOMASwarm: shutdown complete, memory synced.")
