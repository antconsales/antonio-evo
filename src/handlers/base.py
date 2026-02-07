"""
Base Handler - All handlers inherit from this.

RULES:
1. Handlers are STATELESS
2. Handlers cannot call other handlers
3. Handlers cannot make policy decisions
4. Handlers must return Response objects
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from ..models.request import Request
from ..models.response import Response


class BaseHandler(ABC):
    """
    Abstract base for all handlers.

    Every handler processes a request and returns a response.
    No side effects. No state. No calling other handlers.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Override to validate handler-specific config."""
        pass

    @abstractmethod
    def process(self, request: Request) -> Response:
        """
        Process request and return response.

        Must be implemented by all handlers.
        """
        pass

    @property
    def name(self) -> str:
        """Return handler name."""
        return self.__class__.__name__

    def is_available(self) -> bool:
        """
        Check if handler is available.

        Override to check for dependencies (models, binaries, etc.)
        """
        return True
