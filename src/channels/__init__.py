"""
Antonio Evo Channels — Multi-platform messaging gateway (v6.0).

Channels receive messages from external platforms (Telegram, Discord, etc.)
and route them through the same pipeline as the Web UI.
"""

from .telegram import TelegramChannel

__all__ = ["TelegramChannel"]
