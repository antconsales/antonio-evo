"""
Hook Registry — Lifecycle event system for Antonio Evo (v6.0).

Hooks allow plugins to react to events in the processing pipeline:
- pre_process: Before classification (can modify request)
- post_process: After response generation
- on_error: When an error occurs
- on_tool_call: When a tool is about to be called
- on_startup: When the system starts
- on_shutdown: When the system shuts down
"""

import logging
from typing import Dict, Any, Callable, List, Optional

logger = logging.getLogger(__name__)

# Valid hook events
HOOK_EVENTS = {
    "pre_process",
    "post_process",
    "on_error",
    "on_tool_call",
    "on_startup",
    "on_shutdown",
}


class HookRegistry:
    """
    Registry for lifecycle hooks.

    Hooks are simple callables that receive event data.
    They run synchronously in order of registration.
    """

    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {event: [] for event in HOOK_EVENTS}

    def register(self, event: str, callback: Callable) -> bool:
        """
        Register a hook callback for an event.

        Args:
            event: Event name (must be in HOOK_EVENTS)
            callback: Callable that receives (data: Dict[str, Any])

        Returns:
            True if registered, False if invalid event
        """
        if event not in HOOK_EVENTS:
            logger.warning(f"Invalid hook event: {event}. Valid events: {HOOK_EVENTS}")
            return False

        self._hooks[event].append(callback)
        logger.debug(f"Hook registered: {event} -> {callback.__name__}")
        return True

    def emit(self, event: str, data: Dict[str, Any] = None) -> None:
        """
        Emit an event, calling all registered hooks.

        Args:
            event: Event name
            data: Event data passed to callbacks
        """
        if event not in self._hooks:
            return

        data = data or {}
        for callback in self._hooks[event]:
            try:
                callback(data)
            except Exception as e:
                logger.warning(f"Hook error ({event}/{callback.__name__}): {e}")

    def get_hook_count(self, event: Optional[str] = None) -> int:
        """Get number of registered hooks, optionally for a specific event."""
        if event:
            return len(self._hooks.get(event, []))
        return sum(len(hooks) for hooks in self._hooks.values())

    def clear(self, event: Optional[str] = None) -> None:
        """Clear hooks for an event, or all hooks."""
        if event:
            self._hooks[event] = []
        else:
            self._hooks = {e: [] for e in HOOK_EVENTS}
