"""
Response data model.
All handler output gets wrapped in this structure.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time


@dataclass
class ResponseMeta:
    """Metadata attached to every response."""

    request_id: str = ""
    handler: str = ""
    handler_reason: str = ""
    elapsed_ms: int = 0
    timestamp: float = field(default_factory=time.time)

    # Audit trail
    classification: Dict[str, Any] = field(default_factory=dict)
    policy_decision: Dict[str, Any] = field(default_factory=dict)

    # External API tracking
    used_external: bool = False
    external_justification: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "handler": self.handler,
            "handler_reason": self.handler_reason,
            "elapsed_ms": self.elapsed_ms,
            "timestamp": self.timestamp,
            "classification": self.classification,
            "policy_decision": self.policy_decision,
            "used_external": self.used_external,
            "external_justification": self.external_justification
        }


@dataclass
class Response:
    """
    Standard response structure.

    Every handler returns this (or data that gets wrapped in this).
    """

    success: bool = True

    # Main output
    output: Any = None
    text: Optional[str] = None

    # For audio responses
    audio_path: Optional[str] = None

    # For image responses
    image_path: Optional[str] = None

    # Error info
    error: Optional[str] = None
    error_code: Optional[str] = None

    # Warnings (non-fatal issues)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    meta: ResponseMeta = field(default_factory=ResponseMeta)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "success": self.success,
            "_meta": self.meta.to_dict()
        }

        if self.output is not None:
            result["output"] = self.output

        if self.text is not None:
            result["text"] = self.text

        if self.audio_path is not None:
            result["audio_path"] = self.audio_path

        if self.image_path is not None:
            result["image_path"] = self.image_path

        if self.error is not None:
            result["error"] = self.error
            result["error_code"] = self.error_code

        if self.warnings:
            result["warnings"] = self.warnings

        return result

    @classmethod
    def error_response(cls, error: str, code: str, meta: Optional[ResponseMeta] = None) -> "Response":
        """Create an error response."""
        return cls(
            success=False,
            error=error,
            error_code=code,
            meta=meta or ResponseMeta()
        )

    @classmethod
    def success_response(cls, output: Any, meta: Optional[ResponseMeta] = None) -> "Response":
        """Create a success response."""
        return cls(
            success=True,
            output=output,
            meta=meta or ResponseMeta()
        )
