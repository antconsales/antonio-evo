"""
Tool Executor - Executes tools with security validation and result formatting.

Security:
- Output truncated to MAX_OUTPUT_CHARS to prevent prompt bloat
- Callbacks emitted for WebSocket real-time events
- Errors caught and returned gracefully
"""

import time
import logging
from typing import Callable, Dict, Any, Optional

from .registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 4000


class ToolExecutor:
    """Execute tools from the registry with security checks."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, tool_name: str, arguments: Dict[str, Any],
                callback: Optional[Callable] = None) -> ToolResult:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: registered tool name
            arguments: parsed function arguments from LLM
            callback: optional callback(event_type, data) for WS events
        """
        handler = self.registry.get_handler(tool_name)
        if not handler:
            return ToolResult(
                success=False,
                output=f"Unknown tool: {tool_name}",
                tool_name=tool_name,
                error=f"Tool '{tool_name}' not found in registry",
            )

        # Emit tool_action_start
        if callback:
            try:
                callback("tool_action_start", {
                    "tool": tool_name,
                    "arguments": arguments,
                })
            except Exception:
                pass

        start = time.time()
        try:
            result = handler(**arguments)
            result.tool_name = tool_name
            result.elapsed_ms = int((time.time() - start) * 1000)

            # Truncate to prevent prompt bloat
            if len(result.output) > MAX_OUTPUT_CHARS:
                result.output = result.output[:MAX_OUTPUT_CHARS] + "\n... (truncated)"

            logger.info(f"Tool {tool_name}: success={result.success}, {result.elapsed_ms}ms, output={len(result.output)} chars")

        except Exception as e:
            logger.error(f"Tool {tool_name} error: {e}")
            result = ToolResult(
                success=False,
                output=f"Tool execution error: {str(e)}",
                tool_name=tool_name,
                elapsed_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

        # Emit tool_action_end
        if callback:
            try:
                callback("tool_action_end", {
                    "tool": tool_name,
                    "success": result.success,
                    "elapsed_ms": result.elapsed_ms,
                    "output_preview": result.output[:200],
                })
            except Exception:
                pass

        return result
