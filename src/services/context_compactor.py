"""
Context Compactor — Manages context window for long conversations (v6.0).

Qwen3:8b has 32K context but slows significantly beyond 4K tokens.
This service compacts conversation history to keep the context lean.

Strategy (3 levels):
1. Prune old tool results (truncate to 200 chars)
2. Summarize old turns (merge old messages into summary)
3. Drop system bloat (duplicate web search results, etc.)
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ContextCompactor:
    """
    Compacts message history to fit within context window.

    Works on Ollama-format messages: [{"role": "system/user/assistant/tool", "content": "..."}]
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.max_context_tokens = config.get("max_context_tokens", 3000)
        self.compaction_threshold = config.get("compaction_threshold", 4000)
        self.keep_recent_turns = config.get("keep_recent_turns", 4)
        self.tool_result_max_chars = config.get("tool_result_max_chars", 200)
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.ollama_model = config.get("ollama_model", "qwen3:8b")

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Rough token estimate: chars / 4."""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return total_chars // 4

    def needs_compaction(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if messages need compaction."""
        return self.estimate_tokens(messages) > self.compaction_threshold

    def compact(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compact messages to fit within context budget.

        Returns a new list of messages (does not modify input).
        """
        if not messages or len(messages) <= 3:
            return messages

        estimated = self.estimate_tokens(messages)
        if estimated <= self.compaction_threshold:
            return messages

        logger.info(f"Context compaction: {len(messages)} messages, ~{estimated} tokens")

        # Copy messages
        result = list(messages)

        # LEVEL 1: Prune old tool results
        result = self._prune_tool_results(result)

        if self.estimate_tokens(result) <= self.max_context_tokens:
            logger.info(f"Compacted (level 1): ~{self.estimate_tokens(result)} tokens")
            return result

        # LEVEL 2: Summarize old turns (keep system + recent)
        result = self._summarize_old_turns(result)

        if self.estimate_tokens(result) <= self.max_context_tokens:
            logger.info(f"Compacted (level 2): ~{self.estimate_tokens(result)} tokens")
            return result

        # LEVEL 3: Aggressive truncation
        result = self._aggressive_truncate(result)
        logger.info(f"Compacted (level 3): ~{self.estimate_tokens(result)} tokens")

        return result

    def _prune_tool_results(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Level 1: Truncate old tool results to save space."""
        result = []
        # Keep last N messages untouched
        cutoff = max(0, len(messages) - self.keep_recent_turns * 2)

        for i, msg in enumerate(messages):
            if i < cutoff and msg.get("role") == "tool":
                content = msg.get("content", "")
                if len(content) > self.tool_result_max_chars:
                    truncated = content[:self.tool_result_max_chars] + "... [truncated]"
                    result.append({"role": "tool", "content": truncated})
                    continue
            result.append(msg)

        return result

    def _summarize_old_turns(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Level 2: Merge old user/assistant turns into a summary."""
        if len(messages) < 6:
            return messages

        # Keep system message (first) and recent turns
        system_msg = messages[0] if messages[0].get("role") == "system" else None
        keep_from = max(1 if system_msg else 0, len(messages) - self.keep_recent_turns * 2)

        # Collect old messages to summarize
        old_messages = messages[1 if system_msg else 0:keep_from]
        recent_messages = messages[keep_from:]

        if not old_messages:
            return messages

        # Build summary from old messages
        summary_parts = []
        for msg in old_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")[:300]
            if role == "user":
                summary_parts.append(f"User: {content}")
            elif role == "assistant":
                summary_parts.append(f"Assistant: {content}")
            elif role == "tool":
                summary_parts.append(f"[tool result: {content[:100]}]")

        summary_text = (
            "[Conversation summary]\n"
            + "\n".join(summary_parts[:10])  # Max 10 entries in summary
        )

        # Rebuild messages
        result = []
        if system_msg:
            result.append(system_msg)
        result.append({"role": "user", "content": summary_text})
        result.extend(recent_messages)

        return result

    def _aggressive_truncate(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Level 3: Keep only system + last few messages."""
        system_msg = messages[0] if messages[0].get("role") == "system" else None
        recent = messages[-(self.keep_recent_turns * 2):]

        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(recent)

        return result
