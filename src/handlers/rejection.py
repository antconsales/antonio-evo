"""
Rejection Handler - For blocked requests

This handler is used when:
- Rate limits exceeded
- Blocked content detected
- Unsupported operations
- Policy violations
"""

from typing import Dict, Any

from .base import BaseHandler
from ..models.request import Request
from ..models.response import Response, ResponseMeta


class RejectionHandler(BaseHandler):
    """
    Handler for rejected requests.

    Returns structured rejection response with reason.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})

    def process(self, request: Request) -> Response:
        """
        Return rejection response.

        The reason for rejection is in the request metadata,
        set by the policy engine.
        """

        reject_reason = request.metadata.get("reject_reason", "unknown")
        reject_message = request.metadata.get("reject_message", "Request rejected by policy")

        return Response(
            success=False,
            error=reject_message,
            error_code=f"REJECTED_{reject_reason.upper()}",
            output={
                "rejected": True,
                "reason": reject_reason,
                "message": reject_message,
                "suggestion": self._get_suggestion(reject_reason)
            },
            meta=ResponseMeta()
        )

    def _get_suggestion(self, reason: str) -> str:
        """Provide helpful suggestion based on rejection reason."""

        suggestions = {
            "rate_limited": "Wait a moment before retrying",
            "blocked_content": "Modify your request to comply with policies",
            "unsupported": "This operation is not supported locally",
            "too_complex_no_fallback": "Enable external API fallback or simplify request",
            "invalid_input": "Check your input format",
            "handler_unavailable": "Required handler is not configured"
        }

        return suggestions.get(reason, "Review the error and try again")

    def is_available(self) -> bool:
        """Rejection handler is always available."""
        return True
