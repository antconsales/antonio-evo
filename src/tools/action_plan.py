"""Action Plan tool — create multi-step action plans for user approval."""

import json
import time
from .registry import ToolResult

DEFINITION = {
    "name": "create_action_plan",
    "description": (
        "Create a multi-step action plan that requires user approval before execution. "
        "Use this when the user asks for a complex task that involves multiple tool calls "
        "in sequence (e.g., 'generate a report and save it', 'search and summarize')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title for the action plan",
            },
            "steps": {
                "type": "array",
                "description": "Array of steps, each with tool, arguments, and description",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "description": "Tool name to execute"},
                        "arguments": {"type": "object", "description": "Arguments for the tool"},
                        "description": {"type": "string", "description": "What this step does"},
                    },
                    "required": ["tool", "description"],
                },
            },
        },
        "required": ["title", "steps"],
    },
}


def create_handler(workflow_orchestrator):
    """Create action_plan tool handler bound to a WorkflowOrchestrator instance."""

    def create_action_plan(title: str, steps: list) -> ToolResult:
        if not workflow_orchestrator:
            return ToolResult(
                success=False,
                output="Workflow orchestrator not available.",
            )

        start = time.time()
        try:
            plan = workflow_orchestrator.create_plan(
                title=title,
                steps=steps,
            )
            elapsed = int((time.time() - start) * 1000)

            # Format plan for display
            step_lines = []
            for s in plan.steps:
                step_lines.append(
                    f"  {s.step_number}. [{s.tool_name}] {s.description}"
                )

            output = (
                f"Action Plan Created: {plan.title}\n"
                f"Plan ID: {plan.id}\n"
                f"Status: {plan.status} (requires approval)\n\n"
                f"Steps:\n" + "\n".join(step_lines) + "\n\n"
                f"To execute, the user must approve this plan via the UI or API."
            )

            return ToolResult(
                success=True,
                output=output,
                elapsed_ms=elapsed,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Failed to create action plan: {e}",
            )

    return create_action_plan
