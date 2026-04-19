from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Optional, Awaitable
from enum import Enum

class RiskLevel(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"

@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk: RiskLevel = RiskLevel.LOW

    def to_prompt(self) -> str:
        """Returns a string representation for the LLM prompt."""
        return f"- {self.name} [RISK: {self.risk.value}]: {self.description}"

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def get_prompt(self) -> str:
        if not self._tools:
            return "No specialized tools available. Use standard planning."
        return "\n".join([t.to_prompt() for t in self._tools.values()])
