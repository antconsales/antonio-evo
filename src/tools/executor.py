"""
Tool Executor - Executes tools with governance gating and result formatting.

Security (v8.5):
- Every tool call classified by GovernanceEngine before execution
- HIGH/CRITICAL actions require explicit human approval
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
    """Execute tools from the registry with governance gating."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._governance = None

    def set_governance(self, governance_engine) -> None:
        """Inject GovernanceEngine for risk classification and approval gating."""
        self._governance = governance_engine

    def execute(self, tool_name: str, arguments: Dict[str, Any],
                callback: Optional[Callable] = None,
                plan_id: Optional[str] = None,
                session_id: Optional[str] = None) -> ToolResult:
        """
        Execute a tool by name with given arguments.

        Every call passes through the governance gate:
        1. classify_risk() — deterministic risk level
        2. request_approval() — auto-approve or pending
        3. If pending → return with action_id for human approval
        4. check_constraints() → verify arguments within bounds
        5. Execute tool
        6. record_execution() → immutable audit

        Args:
            tool_name: registered tool name
            arguments: parsed function arguments from LLM
            callback: optional callback(event_type, data) for WS events
            plan_id: optional plan ID for governance tracking
            session_id: optional session ID for governance tracking
        """
        handler = self.registry.get_handler(tool_name)
        if not handler:
            return ToolResult(
                success=False,
                output=f"Unknown tool: {tool_name}",
                tool_name=tool_name,
                error=f"Tool '{tool_name}' not found in registry",
            )

        # --- Governance Gate (v8.5) ---
        action_id = None
        if self._governance:
            try:
                classification = self._governance.classify_risk(tool_name, arguments)
                decision = self._governance.request_approval(
                    classification, plan_id=plan_id, session_id=session_id
                )
                action_id = decision.action_id

                if decision.status == "pending":
                    # HIGH/CRITICAL: requires human approval
                    logger.info(f"Tool {tool_name}: governance PENDING (risk={classification.level.value}, "
                                f"action_id={action_id})")
                    return ToolResult(
                        success=False,
                        output=f"⏳ Action requires approval (risk: {classification.level.value}). "
                               f"Action ID: {action_id}",
                        tool_name=tool_name,
                        error="governance_pending",
                        metadata={
                            "governance_action_id": action_id,
                            "risk_level": classification.level.value,
                            "reasons": classification.reasons,
                            "requires_approval": True,
                        },
                    )

                if decision.status == "denied":
                    logger.info(f"Tool {tool_name}: governance DENIED (action_id={action_id})")
                    return ToolResult(
                        success=False,
                        output=f"Action denied by governance policy.",
                        tool_name=tool_name,
                        error="governance_denied",
                        metadata={
                            "governance_action_id": action_id,
                            "risk_level": classification.level.value,
                        },
                    )

                # Check constraints before execution
                ok, reason = self._governance.check_constraints(
                    tool_name, arguments, classification.constraints
                )
                if not ok:
                    self._governance.record_execution(action_id, False, f"constraint_violation: {reason}")
                    return ToolResult(
                        success=False,
                        output=f"Action blocked by constraints: {reason}",
                        tool_name=tool_name,
                        error="governance_constraint",
                        metadata={
                            "governance_action_id": action_id,
                            "risk_level": classification.level.value,
                        },
                    )

                logger.debug(f"Tool {tool_name}: governance APPROVED (risk={classification.level.value})")

            except Exception as e:
                logger.warning(f"Governance gate error (allowing execution): {e}")

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

            # Attach governance action_id to every result (v8.5 first-class governance)
            if action_id:
                result.metadata["governance_action_id"] = action_id

            logger.info(f"Tool {tool_name}: success={result.success}, {result.elapsed_ms}ms, output={len(result.output)} chars")

        except Exception as e:
            logger.error(f"Tool {tool_name} error: {e}")
            result = ToolResult(
                success=False,
                output=f"Tool execution error: {str(e)}",
                tool_name=tool_name,
                elapsed_ms=int((time.time() - start) * 1000),
                error=str(e),
                metadata={"governance_action_id": action_id} if action_id else {},
            )

        # Record execution in governance (v8.5)
        if self._governance and action_id:
            try:
                self._governance.record_execution(
                    action_id, result.success, result.output[:200]
                )
            except Exception as e:
                logger.debug(f"Failed to record governance execution: {e}")

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
