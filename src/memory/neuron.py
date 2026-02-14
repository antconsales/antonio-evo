"""
Neuron data structures for EvoMemory.

Per Antonio Evo Unified Spec (v3.1):
Memory is OBSERVATIONAL, not authoritative.

A Neuron represents a single learned interaction:
- input: The user's query/request
- output: Antonio's response
- confidence: How confident we are in this response (0.0-1.0)
- mood: Emotional context (neutral, helpful, curious, etc.)
- persona: Which persona generated this (SOCIAL/LOGIC)
- decay_eligible: Whether this memory can decay over time

Memory influences tone/verbosity/style, never overrides user intent.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import time
import uuid
import hashlib


class Mood(Enum):
    """Mood annotation for neurons."""
    NEUTRAL = "neutral"
    HELPFUL = "helpful"
    CURIOUS = "curious"
    ANALYTICAL = "analytical"
    FRIENDLY = "friendly"
    CAUTIOUS = "cautious"


class Persona(Enum):
    """Persona that generated the response."""
    SOCIAL = "social"      # Conversational, empathetic
    LOGIC = "logic"        # Analytical, precise
    UNKNOWN = "unknown"    # Legacy or unclassified


@dataclass
class Neuron:
    """
    A single memory unit in EvoMemory.

    Per Antonio Evo Unified Spec (v3.1), each neuron includes:
    - Timestamp (created_at, updated_at)
    - Context (session_id, classification_domain)
    - Confidence score (0.0-1.0)
    - Decay eligibility (can this memory be pruned over time)

    Neurons are immutable once created - confidence and access_count
    are updated via separate operations.
    """

    id: str
    input: str
    input_hash: str
    output: str
    confidence: float
    mood: Mood
    handler: str
    persona: Persona
    created_at: float
    updated_at: float
    access_count: int = 0
    last_accessed: Optional[float] = None

    # Decay eligibility per Antonio Evo spec
    # True = memory can decay over time if not accessed
    # False = memory is pinned (explicit user preference, high confidence)
    decay_eligible: bool = True
    decay_factor: float = 1.0  # Multiplier applied to confidence over time

    # Optional metadata
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    classification_domain: Optional[str] = None

    # Multimodal context (v8.0)
    attachment_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "input": self.input,
            "input_hash": self.input_hash,
            "output": self.output,
            "confidence": self.confidence,
            "mood": self.mood.value,
            "handler": self.handler,
            "persona": self.persona.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "decay_eligible": self.decay_eligible,
            "decay_factor": self.decay_factor,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "classification_domain": self.classification_domain,
            "attachment_summary": self.attachment_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Neuron":
        """Create Neuron from dictionary."""
        return cls(
            id=data["id"],
            input=data["input"],
            input_hash=data["input_hash"],
            output=data["output"],
            confidence=data["confidence"],
            mood=Mood(data.get("mood", "neutral")),
            handler=data["handler"],
            persona=Persona(data.get("persona", "unknown")),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
            decay_eligible=data.get("decay_eligible", True),
            decay_factor=data.get("decay_factor", 1.0),
            session_id=data.get("session_id"),
            request_id=data.get("request_id"),
            classification_domain=data.get("classification_domain"),
            attachment_summary=data.get("attachment_summary"),
        )

    def get_effective_confidence(self, current_time: Optional[float] = None) -> float:
        """
        Get confidence adjusted for decay.

        Per Antonio Evo spec: memory confidence decays over time
        unless pinned (decay_eligible=False).
        """
        if not self.decay_eligible:
            return self.confidence

        if current_time is None:
            current_time = time.time()

        # Calculate age in days
        age_days = (current_time - self.created_at) / (24 * 3600)

        # Decay formula: confidence * decay_factor^(age/30)
        # Decays ~10% per month if decay_factor=0.9
        if age_days > 0:
            decay = self.decay_factor ** (age_days / 30)
            return max(0.1, self.confidence * decay)  # Minimum 10%

        return self.confidence

    def should_prune(self, min_confidence: float = 0.2, min_access_days: int = 90) -> bool:
        """
        Check if this neuron should be pruned.

        Per Antonio Evo spec: memory cleanup policies apply.
        """
        if not self.decay_eligible:
            return False  # Pinned memories never pruned

        current_time = time.time()
        effective_conf = self.get_effective_confidence(current_time)

        # Prune if confidence too low
        if effective_conf < min_confidence:
            return True

        # Prune if not accessed in min_access_days
        if self.last_accessed:
            days_since_access = (current_time - self.last_accessed) / (24 * 3600)
            if days_since_access > min_access_days and effective_conf < 0.5:
                return True

        return False


@dataclass
class NeuronCreate:
    """
    Data required to create a new Neuron.

    Used as input to MemoryStorage.store().
    """

    input: str
    output: str
    handler: str
    persona: Persona = Persona.UNKNOWN
    mood: Mood = Mood.NEUTRAL
    confidence: float = 0.5
    decay_eligible: bool = True  # Per spec: memory can decay by default
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    classification_domain: Optional[str] = None
    attachment_summary: Optional[str] = None  # Multimodal context (v8.0)

    def to_neuron(self) -> Neuron:
        """Convert to full Neuron with generated fields."""
        now = time.time()

        # High confidence neurons (explicit preferences) are pinned
        # Per Antonio Evo spec: explicit preferences don't decay
        should_decay = self.decay_eligible
        if self.confidence >= 0.9:
            should_decay = False  # Pin high-confidence memories

        return Neuron(
            id=str(uuid.uuid4())[:12],
            input=self.input,
            input_hash=hashlib.sha256(self.input.encode()).hexdigest()[:16],
            output=self.output,
            confidence=self.confidence,
            mood=self.mood,
            handler=self.handler,
            persona=self.persona,
            created_at=now,
            updated_at=now,
            access_count=0,
            last_accessed=None,
            decay_eligible=should_decay,
            decay_factor=0.95,  # ~5% decay per month
            session_id=self.session_id,
            request_id=self.request_id,
            classification_domain=self.classification_domain,
            attachment_summary=self.attachment_summary,
        )


@dataclass
class RetrievedNeuron:
    """
    A Neuron with retrieval metadata.

    Used when returning search results.
    """

    neuron: Neuron
    relevance_score: float  # BM25 score
    match_type: str  # "exact", "semantic", "partial"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "neuron": self.neuron.to_dict(),
            "relevance_score": self.relevance_score,
            "match_type": self.match_type,
        }


@dataclass
class UserPreference:
    """
    A learned user preference.

    Examples:
    - persona_code: "LOGIC" (user prefers LOGIC for code questions)
    - response_style: "concise" (user prefers short answers)
    """

    key: str
    value: str
    confidence: float
    learned_from: Optional[str] = None  # neuron_id that established this
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "learned_from": self.learned_from,
            "created_at": self.created_at,
        }


@dataclass
class MemoryContext:
    """
    Memory context passed through the pipeline.

    Retrieved before classification, used by:
    - Classifier (to inform intent detection)
    - Policy Engine (to make memory-aware decisions)
    - Handlers (to use past context in responses)
    """

    # Retrieved neurons (most relevant first)
    relevant_neurons: List[RetrievedNeuron] = field(default_factory=list)

    # User preferences (learned over time)
    preferences: Dict[str, UserPreference] = field(default_factory=dict)

    # Session info
    session_id: Optional[str] = None
    session_neuron_count: int = 0

    # Aggregated stats
    avg_confidence: float = 0.0
    dominant_mood: Optional[Mood] = None

    # Flags for pipeline
    has_relevant_memory: bool = False
    has_multimodal_context: bool = False  # v8.0: neurons with attachment summaries
    memory_retrieval_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata attachment."""
        return {
            "relevant_neurons": [n.to_dict() for n in self.relevant_neurons],
            "preferences": {k: v.to_dict() for k, v in self.preferences.items()},
            "session_id": self.session_id,
            "session_neuron_count": self.session_neuron_count,
            "avg_confidence": self.avg_confidence,
            "dominant_mood": self.dominant_mood.value if self.dominant_mood else None,
            "has_relevant_memory": self.has_relevant_memory,
            "has_multimodal_context": self.has_multimodal_context,
            "memory_retrieval_ms": self.memory_retrieval_ms,
        }

    def get_best_neuron(self) -> Optional[Neuron]:
        """Get the most relevant neuron, if any."""
        if self.relevant_neurons:
            return self.relevant_neurons[0].neuron
        return None

    def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a preference value by key."""
        pref = self.preferences.get(key)
        if pref and pref.confidence >= 0.5:
            return pref.value
        return default


@dataclass
class MemoryOperation:
    """
    Record of a memory operation for audit logging.

    Stores references, not content (to avoid duplication).
    """

    operation_type: str  # "retrieve", "store", "update", "delete"
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
        return {
            "operation_type": self.operation_type,
            "neuron_ids": self.neuron_ids,
            "retrieval_scores": self.retrieval_scores if self.retrieval_scores else None,
            "query_hash": self.query_hash,
            "stored_neuron_id": self.stored_neuron_id,
            "confidence_assigned": self.confidence_assigned,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
        }
