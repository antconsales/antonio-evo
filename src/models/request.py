"""
Request data model.
All input gets normalized to this structure.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import time
import uuid

# Avoid circular import
if TYPE_CHECKING:
    from ..memory.neuron import MemoryContext


@dataclass
class Attachment:
    """
    Attachment data structure.

    Represents a file attachment from the user.
    Attachments are treated as UNTRUSTED INERT DATA - never executed.
    """
    name: str
    type: str  # MIME type (e.g., "image/png", "text/plain")
    size: int
    data: str  # Base64 encoded content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "size": self.size,
            "has_data": bool(self.data),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attachment":
        """Create Attachment from dictionary."""
        return cls(
            name=data.get("name", "unknown"),
            type=data.get("type", "application/octet-stream"),
            size=data.get("size", 0),
            data=data.get("data", ""),
        )

    def is_image(self) -> bool:
        """Check if attachment is an image."""
        return self.type.startswith("image/")

    def is_text(self) -> bool:
        """Check if attachment is text-based."""
        text_types = ["text/", "application/json", "application/xml", "application/javascript"]
        return any(self.type.startswith(t) for t in text_types)

    def is_code(self) -> bool:
        """Check if attachment is source code."""
        code_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h',
                          '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala']
        return any(self.name.lower().endswith(ext) for ext in code_extensions)


class Modality(Enum):
    """Supported input/output modalities."""
    TEXT = "text"
    AUDIO_INPUT = "audio_input"
    AUDIO_OUTPUT = "audio_output"
    IMAGE_CAPTION = "image_caption"
    IMAGE_GENERATION = "image_generation"
    VIDEO = "video"


@dataclass
class Request:
    """
    Normalized request structure.

    Every input (HTTP, CLI, audio, file) gets converted to this.
    """

    # Core fields
    text: str = ""
    modality: Modality = Modality.TEXT

    # Optional data
    audio_path: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    image_path: Optional[str] = None
    image_bytes: Optional[bytes] = None

    # Attachments (v2.4 - file uploads from chat)
    # SECURITY: Treated as UNTRUSTED INERT DATA - never executed, only analyzed
    attachments: List[Attachment] = field(default_factory=list)

    # Task specification
    task_type: str = "chat"  # chat, classify, reason, generate, plan

    # Quality hints
    quality: str = "standard"  # low, standard, high

    # Metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"  # cli, http, audio, file

    # Extra data (for extensibility)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Memory context (populated by MemoryRetriever before classification)
    # Contains relevant neurons, user preferences, session info
    memory_context: Optional[Any] = None  # Type: MemoryContext (avoid circular import)

    # Emotional context (v2.1 - populated by EmotionalMemory before classification)
    # Contains user emotional state, trends, and tone recommendations
    emotional_context: Optional[Any] = None  # Type: EmotionalContext (avoid circular import)

    # Session tracking
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "request_id": self.request_id,
            "text": self.text,
            "modality": self.modality.value,
            "task_type": self.task_type,
            "quality": self.quality,
            "timestamp": self.timestamp,
            "source": self.source,
            "has_audio": self.audio_path is not None or self.audio_bytes is not None,
            "has_image": self.image_path is not None or self.image_bytes is not None,
            "has_attachments": len(self.attachments) > 0,
            "attachments": [a.to_dict() for a in self.attachments] if self.attachments else [],
            "metadata": self.metadata,
            "session_id": self.session_id,
        }

        # Include memory context summary if available
        if self.memory_context is not None:
            result["memory_context"] = self.memory_context.to_dict()

        # Include emotional context if available (v2.1)
        if self.emotional_context is not None:
            result["emotional_context"] = self.emotional_context.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Request":
        """Create Request from dictionary."""
        modality_str = data.get("modality", "text")
        try:
            modality = Modality(modality_str)
        except ValueError:
            modality = Modality.TEXT

        # Parse attachments if present
        attachments = []
        raw_attachments = data.get("attachments", [])
        for att_data in raw_attachments:
            if isinstance(att_data, dict):
                attachments.append(Attachment.from_dict(att_data))

        return cls(
            text=data.get("text", ""),
            modality=modality,
            audio_path=data.get("audio_path"),
            image_path=data.get("image_path"),
            task_type=data.get("task_type", "chat"),
            quality=data.get("quality", "standard"),
            source=data.get("source", "unknown"),
            metadata=data.get("metadata", {}),
            session_id=data.get("session_id"),
            attachments=attachments,
        )
