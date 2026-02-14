"""
Antonio Evo Plugin System (v6.0).

Provides extensibility through:
- register_tool(): Add new tools to the ReAct loop
- register_hook(): Add lifecycle callbacks
- register_channel(): Add messaging channels
"""

from .manager import PluginManager
from .hooks import HookRegistry

__all__ = ["PluginManager", "HookRegistry"]
