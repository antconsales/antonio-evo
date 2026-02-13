"""
Session Manager - Handles conversation session lifecycle.

Extracted from Orchestrator to follow Single Responsibility Principle.
"""

import uuid
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages conversation sessions.

    Responsibilities:
    - Create new sessions with unique IDs
    - Track current active session
    - End sessions and persist to memory storage
    """

    def __init__(self, memory_storage=None):
        """
        Initialize session manager.

        Args:
            memory_storage: Optional MemoryStorage for session persistence
        """
        self._memory_storage = memory_storage
        self._current_session_id: Optional[str] = None

    @property
    def current_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._current_session_id

    @current_session_id.setter
    def current_session_id(self, value: Optional[str]):
        """Set current session ID (for external assignment)."""
        self._current_session_id = value

    def start_session(self) -> str:
        """
        Start a new conversation session.

        Returns:
            New session ID (12-char UUID prefix)
        """
        self._current_session_id = str(uuid.uuid4())[:12]
        if self._memory_storage:
            try:
                self._memory_storage.create_session(self._current_session_id)
            except Exception:
                # Session creation in storage failed, continue without it
                pass
        logger.debug(f"Session started: {self._current_session_id}")
        return self._current_session_id

    def end_session(self):
        """End the current session."""
        if self._current_session_id and self._memory_storage:
            try:
                self._memory_storage.end_session(self._current_session_id)
            except Exception:
                # Session end in storage failed, continue
                pass
        logger.debug(f"Session ended: {self._current_session_id}")
        self._current_session_id = None
