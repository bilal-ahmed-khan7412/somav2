import asyncio
import logging
from typing import List, Optional, Dict, Any

from .core.kernel import V2Kernel
from .core.director import AgentDirector
from .core.a2a import A2ABus
from .memory.hierarchical import HierarchicalMemory
from .core.tools import ToolRegistry, Tool, RiskLevel
from .connectors import get_llm_callback

logger = logging.getLogger("SOMA_V2.MAIN")

class SOMASwarm:
    """
    SOMASwarm: The High-Level Entry Point for SOMA V2.
    
    This class orchestrates all kernel components (Director, Memory, Blackboard)
    into a single usable object.
    
    Usage:
        swarm = SOMASwarm(model="ollama/qwen2.5:3b", slots=3)
        swarm.register_tool("check_status", my_func, "Check current drone status")
        result = await swarm.dispatch("Analyze sensor drift at node 5")
    """
    
    def __init__(
        self, 
        model: str = "ollama/qwen2.5:3b", 
        slots: int = 3,
        hot_cache_cap: int = 128,
        persist_dir: str = "./soma_memory",
        trace_dir: Optional[str] = "./soma_traces"
    ):
        from .core.telemetry import TelemetryStore
        self.bus = A2ABus()
        self.telemetry = TelemetryStore(trace_dir=trace_dir)
        self.memory = HierarchicalMemory(
            hot_capacity=hot_cache_cap,
            cold_persist=persist_dir
        )
        self.tool_registry = ToolRegistry()
        self.llm_callback = get_llm_callback(model)
        
        self.director = AgentDirector(
            llm_callback=self.llm_callback, 
            memory=self.memory,
            tool_registry=self.tool_registry,
            telemetry=self.telemetry
        )
        
        # Initialise standard slots
        self._setup_default_slots(slots)
        
        logger.info(f"SOMASwarm initialized with model={model}, slots={slots}")

    def register_tool(self, name: str, func, description: str, risk: RiskLevel = RiskLevel.LOW):
        """Register a new capability for the swarm agents to use."""
        self.tool_registry.register(Tool(name=name, func=func, description=description, risk=risk))
        logger.info(f"Registered tool: {name} [RISK={risk.value}]")

    def _setup_default_slots(self, count: int):
        """Creates a pool of kernels with varied roles."""
        for i in range(count):
            role = "SUPERVISOR" if i == 0 else "PEER"
            self.director.add_slot(
                slot_id=f"agent_{i}",
                role=role,
                capacity=5
            )

    async def dispatch(self, task: str, urgency: str = "medium") -> Dict[str, Any]:
        """
        Dispatches a task to the swarm.
        Returns the final TASK_RESULT payload.
        """
        try:
            result = await self.director.assign(task, urgency=urgency)
            return result
        except Exception as e:
            logger.error(f"SOMASwarm: dispatch failed - {e}")
            return {
                "status": "error",
                "error": str(e),
                "decision": f"Kernel fault during dispatch: {e}"
            }

    @property
    def stats(self) -> Dict[str, Any]:
        """Returns unified performance metrics for the swarm."""
        return {
            "memory": self.memory.stats,
            "director": self.director.stats,
            "bus_messages": self.bus.message_count
        }

    def get_suspended_tasks(self) -> Dict[str, str]:
        """Returns tasks currently paused for human intervention."""
        return self.director.get_suspended_tasks()

    def approve(self, agent_id: str):
        """Resumes the swarm's execution for a specific agent's plan."""
        self.director.approve_task(agent_id)

    async def close(self):
        """Shutdown the swarm and its components."""
        await self.director.stop()
        self.memory.sync()
        logger.info("SOMASwarm shutdown complete and memory synced.")
