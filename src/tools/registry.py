"""
Tool Registry - OpenAI-format function definitions for Ollama native tool calling.

Each tool is registered with:
- name: unique tool identifier
- description: what the tool does (sent to LLM)
- parameters: JSON Schema for function arguments
- handler: callable(**args) -> ToolResult
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: str
    tool_name: str = ""
    elapsed_ms: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    Registry of available tools in OpenAI function-calling format.

    Tools are registered at startup and their definitions are passed
    to Ollama's /api/chat via the 'tools' parameter.
    """

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, Callable] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any],
                 handler: Callable[..., ToolResult]):
        """Register a tool with its OpenAI-format definition and handler."""
        self._tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        self._handlers[name] = handler
        logger.debug(f"Tool registered: {name}")

    def get_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions in Ollama/OpenAI format."""
        return list(self._tools.values())

    def get_handler(self, name: str) -> Optional[Callable]:
        """Get the handler callable for a tool name."""
        return self._handlers.get(name)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
