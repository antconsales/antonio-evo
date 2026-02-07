"""
Policy data models.
Defines handlers, decisions, personas, and rejection reasons.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class Persona(Enum):
    """
    Dual-model personas for response generation.

    SOCIAL: Conversational, empathetic, uses analogies
    LOGIC: Analytical, precise, structured reasoning
    """
    SOCIAL = "social"
    LOGIC = "logic"
    AUTO = "auto"  # Let policy engine decide


class Handler(Enum):
    """Available handlers in the system."""

    # Text handlers with persona support
    TEXT_LOCAL = "text_local"        # Generic Mistral (legacy)
    TEXT_SOCIAL = "text_social"      # SOCIAL persona handler
    TEXT_LOGIC = "text_logic"        # LOGIC persona handler

    # Audio handlers
    AUDIO_IN = "audio_in"            # Whisper (STT)
    AUDIO_OUT = "audio_out"          # Piper TTS

    # Image handlers
    IMAGE_CAPTION = "image_caption"  # CLIP
    IMAGE_GEN = "image_gen"          # Stable Diffusion (limited)

    # Fallback and rejection
    EXTERNAL_LLM = "external_llm"    # Claude/GPT fallback
    REJECT = "reject"                # Blocked requests


@dataclass
class MemoryOperation:
    """
    Record of a memory operation for audit logging.

    Stores references, not content (to avoid duplication with EvoMemory).
    """
    operation_type: str  # "retrieve", "store", "update", "delete", "none"
    neuron_ids: List[str] = field(default_factory=list)

    # For retrieve operations
    retrieval_scores: List[float] = field(default_factory=list)
    query_hash: Optional[str] = None

    # For store operations
    stored_neuron_id: Optional[str] = None
    confidence_assigned: Optional[float] = None

    # For update operations
    confidence_before: Optional[float] = None
    confidence_after: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for audit log."""
        result = {"operation_type": self.operation_type}

        if self.neuron_ids:
            result["neuron_ids"] = self.neuron_ids
        if self.retrieval_scores:
            result["retrieval_scores"] = self.retrieval_scores
        if self.query_hash:
            result["query_hash"] = self.query_hash
        if self.stored_neuron_id:
            result["stored_neuron_id"] = self.stored_neuron_id
        if self.confidence_assigned is not None:
            result["confidence_assigned"] = self.confidence_assigned
        if self.confidence_before is not None:
            result["confidence_before"] = self.confidence_before
        if self.confidence_after is not None:
            result["confidence_after"] = self.confidence_after

        return result


class RejectReason(Enum):
    """Reasons for rejecting a request."""

    RATE_LIMITED = "rate_limited"
    BLOCKED_CONTENT = "blocked_content"
    UNSUPPORTED = "unsupported"
    TOO_COMPLEX_NO_FALLBACK = "too_complex_no_fallback"
    INVALID_INPUT = "invalid_input"
    HANDLER_UNAVAILABLE = "handler_unavailable"


@dataclass
class PolicyDecision:
    """
    The output of the policy engine.

    This is what the router uses to dispatch requests.
    """

    handler: Handler
    reason: str

    # Persona selection (for text handlers)
    persona: Persona = Persona.AUTO

    # External API flags
    allow_external: bool = False
    external_justification: Optional[str] = None

    # Rejection info
    reject_reason: Optional[RejectReason] = None

    # Memory operation to perform
    memory_operation: Optional[MemoryOperation] = None

    # Should store response as neuron?
    store_as_neuron: bool = True

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        result = {
            "handler": self.handler.value,
            "reason": self.reason,
            "persona": self.persona.value,
            "allow_external": self.allow_external,
            "store_as_neuron": self.store_as_neuron,
        }

        if self.external_justification:
            result["external_justification"] = self.external_justification

        if self.reject_reason:
            result["reject_reason"] = self.reject_reason.value

        if self.memory_operation:
            result["memory_operation"] = self.memory_operation.to_dict()

        if self.metadata:
            result["metadata"] = self.metadata

        return result


@dataclass
class Classification:
    """
    The output of the classifier.

    This is used by the policy engine to make routing decisions.
    """

    intent: str = "unknown"  # question, command, generation, analysis, conversation, unknown
    domain: str = "general"  # text, code, math, image, audio, general
    complexity: str = "simple"  # simple, moderate, complex
    requires_external: bool = False
    confidence: float = 0.5
    reasoning: str = ""

    # Persona suggestion (classifier's recommendation)
    suggested_persona: Persona = Persona.AUTO

    # Memory-informed classification
    memory_informed: bool = False  # Was memory context used?
    memory_confidence: float = 0.0  # Confidence from similar past queries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "domain": self.domain,
            "complexity": self.complexity,
            "requires_external": self.requires_external,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "suggested_persona": self.suggested_persona.value,
            "memory_informed": self.memory_informed,
            "memory_confidence": self.memory_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Classification":
        """Create Classification from dictionary (e.g., LLM output)."""
        # Parse persona if present
        persona_str = data.get("suggested_persona", "auto")
        try:
            persona = Persona(persona_str)
        except ValueError:
            persona = Persona.AUTO

        return cls(
            intent=data.get("intent", "unknown"),
            domain=data.get("domain", "general"),
            complexity=data.get("complexity", "simple"),
            requires_external=data.get("requires_external", False),
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", ""),
            suggested_persona=persona,
            memory_informed=data.get("memory_informed", False),
            memory_confidence=data.get("memory_confidence", 0.0),
        )
