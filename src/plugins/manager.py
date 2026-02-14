"""
Plugin Manager — Extensibility framework for Antonio Evo (v6.0).

Plugins are Python files in the `plugins/` directory that export a
`register(plugin_manager)` function. This function receives the
PluginManager and can register tools, hooks, and channels.

Example plugin (plugins/hello.py):
```python
def register(pm):
    pm.register_tool(
        name="hello_world",
        description="Says hello",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: "Hello, World!"
    )
    pm.register_hook("post_process", lambda data: print("Response sent!"))
```
"""

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List

from .hooks import HookRegistry

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Central plugin management for Antonio Evo.

    Provides registration APIs for:
    - Tools (added to ToolRegistry for ReAct loop)
    - Hooks (lifecycle event callbacks)
    - Channels (messaging platform integrations)
    """

    def __init__(self, tool_registry=None, hook_registry: Optional[HookRegistry] = None):
        self._tool_registry = tool_registry
        self._hook_registry = hook_registry or HookRegistry()
        self._channels: Dict[str, Any] = {}
        self._loaded_plugins: List[str] = []

    @property
    def hook_registry(self) -> HookRegistry:
        return self._hook_registry

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
    ) -> bool:
        """
        Register a new tool for the ReAct loop.

        Args:
            name: Tool name (e.g. "weather_check")
            description: What the tool does
            parameters: JSON Schema for tool parameters
            handler: Callable(arguments: dict) -> str

        Returns:
            True if registered successfully
        """
        if not self._tool_registry:
            logger.warning(f"Cannot register tool '{name}': no tool registry available")
            return False

        try:
            self._tool_registry.register(
                name=name,
                description=description,
                parameters=parameters,
                handler=handler,
            )
            logger.info(f"Plugin tool registered: {name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to register tool '{name}': {e}")
            return False

    def register_hook(self, event: str, callback: Callable) -> bool:
        """
        Register a lifecycle hook.

        Args:
            event: Event name (pre_process, post_process, on_error, on_tool_call, on_startup, on_shutdown)
            callback: Callable(data: dict)

        Returns:
            True if registered successfully
        """
        return self._hook_registry.register(event, callback)

    def register_channel(self, name: str, channel) -> bool:
        """
        Register a messaging channel.

        Args:
            name: Channel identifier (e.g. "discord")
            channel: Channel instance with start()/stop() methods

        Returns:
            True if registered successfully
        """
        self._channels[name] = channel
        logger.info(f"Plugin channel registered: {name}")
        return True

    def load_plugins(self, plugins_dir: str = "plugins") -> int:
        """
        Discover and load plugins from a directory.

        Each .py file in the directory should export a `register(plugin_manager)` function.

        Args:
            plugins_dir: Path to plugins directory

        Returns:
            Number of plugins loaded
        """
        plugins_path = Path(plugins_dir)
        if not plugins_path.exists():
            logger.info(f"No plugins directory at {plugins_dir}")
            return 0

        loaded = 0
        for plugin_file in sorted(plugins_path.glob("*.py")):
            if plugin_file.name.startswith("_"):
                continue

            try:
                plugin_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(
                    f"plugins.{plugin_name}", plugin_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, "register"):
                        module.register(self)
                        self._loaded_plugins.append(plugin_name)
                        loaded += 1
                        logger.info(f"Plugin loaded: {plugin_name}")
                    else:
                        logger.warning(f"Plugin {plugin_name} has no register() function")

            except Exception as e:
                logger.warning(f"Failed to load plugin {plugin_file.name}: {e}")

        return loaded

    def get_stats(self) -> Dict[str, Any]:
        """Return plugin system statistics."""
        return {
            "loaded_plugins": self._loaded_plugins,
            "total_plugins": len(self._loaded_plugins),
            "total_hooks": self._hook_registry.get_hook_count(),
            "registered_channels": list(self._channels.keys()),
        }
