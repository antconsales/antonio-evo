"""
Validation Schema - Defines constraints for input validation.

This module defines:
- Field constraints (lengths, allowed values)
- Error types for validation failures
- Result structures for validation outcomes

Design principles:
- Deterministic validation
- Stateless (no side effects)
- No external dependencies
- No sanitization (that is handled separately)
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# Constants - Validation Limits
# =============================================================================

# Maximum length for text input (characters)
MAX_TEXT_LENGTH = 10000

# Maximum serialized size for metadata (characters)
MAX_METADATA_SIZE = 1000

# Maximum length for source field (characters)
MAX_SOURCE_LENGTH = 50

# Maximum length for task_type field (characters)
MAX_TASK_TYPE_LENGTH = 50

# Maximum length for quality field (characters)
MAX_QUALITY_LENGTH = 20

# Allowed modality values (must match Modality enum)
ALLOWED_MODALITIES: Set[str] = {
    "text",
    "audio_input",
    "audio_output",
    "image_caption",
    "image_generation",
    "video",
}

# Allowed task type values
ALLOWED_TASK_TYPES: Set[str] = {
    "classify",
    "reason",
    "generate",
    "plan",
    "translate",
    "summarize",
    "code",
    "analyze",
}

# Allowed quality levels
ALLOWED_QUALITY_LEVELS: Set[str] = {
    "low",
    "standard",
    "high",
}

# Allowed source values
ALLOWED_SOURCES: Set[str] = {
    "cli",
    "http",
    "audio",
    "file",
    "string",
    "dict",
    "unknown",
    "api",
    "mcp",
    "websocket",
}


# =============================================================================
# Error Types
# =============================================================================

class ValidationErrorType(Enum):
    """Types of validation errors."""

    # Type errors
    INVALID_TYPE = "invalid_type"
    NOT_A_STRING = "not_a_string"
    NOT_A_DICT = "not_a_dict"
    NOT_A_LIST = "not_a_list"
    NOT_A_NUMBER = "not_a_number"
    NOT_A_BOOLEAN = "not_a_boolean"

    # Required field errors
    MISSING_FIELD = "missing_field"
    EMPTY_VALUE = "empty_value"
    NULL_VALUE = "null_value"

    # Length errors
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"

    # Value errors
    INVALID_VALUE = "invalid_value"
    NOT_IN_ALLOWED_VALUES = "not_in_allowed_values"

    # Format errors
    CONTAINS_NULL_BYTES = "contains_null_bytes"
    INVALID_FORMAT = "invalid_format"

    # Metadata errors
    METADATA_TOO_LARGE = "metadata_too_large"
    METADATA_NOT_SERIALIZABLE = "metadata_not_serializable"


@dataclass
class FieldError:
    """
    Error for a specific field.

    Attributes:
        field: Name of the field that failed validation
        error_type: Type of validation error
        message: Human-readable error message
        value: The invalid value (truncated if too long)
        constraint: The constraint that was violated (e.g., max length)
    """
    field: str
    error_type: ValidationErrorType
    message: str
    value: Optional[Any] = None
    constraint: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "field": self.field,
            "error_type": self.error_type.value,
            "message": self.message,
        }
        if self.value is not None:
            # Truncate long values for readability
            value_str = str(self.value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            result["value"] = value_str
        if self.constraint is not None:
            result["constraint"] = self.constraint
        return result


class ValidationError(Exception):
    """
    Exception raised when validation fails.

    Contains a list of all field errors encountered.
    """

    def __init__(self, errors: List[FieldError]):
        self.errors = errors
        message = f"Validation failed with {len(errors)} error(s)"
        if errors:
            message += f": {errors[0].message}"
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "valid": False,
            "error_count": len(self.errors),
            "errors": [e.to_dict() for e in self.errors],
        }


@dataclass
class ValidationResult:
    """
    Result of validation.

    Attributes:
        valid: True if validation passed
        errors: List of field errors (empty if valid)
        data: The validated data (only set if valid)
    """
    valid: bool
    errors: List[FieldError] = field(default_factory=list)
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "valid": self.valid,
            "error_count": len(self.errors),
        }
        if self.errors:
            result["errors"] = [e.to_dict() for e in self.errors]
        return result


# =============================================================================
# Schema Definition
# =============================================================================

@dataclass
class FieldSchema:
    """
    Schema for a single field.

    Attributes:
        name: Field name
        field_type: Expected Python type
        required: Whether field is required
        default: Default value if not provided
        max_length: Maximum length (for strings)
        min_length: Minimum length (for strings)
        allowed_values: Set of allowed values (if constrained)
        allow_empty: Whether empty string is allowed
        allow_null: Whether None is allowed
    """
    name: str
    field_type: type
    required: bool = False
    default: Any = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    allowed_values: Optional[Set[str]] = None
    allow_empty: bool = True
    allow_null: bool = True


class RequestSchema:
    """
    Schema definition for Request validation.

    Defines all fields and their constraints for input requests.
    This schema is used by the Validator to check input data.
    """

    # Field definitions
    FIELDS: Dict[str, FieldSchema] = {
        "text": FieldSchema(
            name="text",
            field_type=str,
            required=False,
            default="",
            max_length=MAX_TEXT_LENGTH,
            min_length=0,
            allow_empty=True,
            allow_null=False,
        ),
        "modality": FieldSchema(
            name="modality",
            field_type=str,
            required=False,
            default="text",
            allowed_values=ALLOWED_MODALITIES,
            allow_empty=False,
            allow_null=False,
        ),
        "task_type": FieldSchema(
            name="task_type",
            field_type=str,
            required=False,
            default="chat",
            max_length=MAX_TASK_TYPE_LENGTH,
            allowed_values=ALLOWED_TASK_TYPES,
            allow_empty=False,
            allow_null=False,
        ),
        "quality": FieldSchema(
            name="quality",
            field_type=str,
            required=False,
            default="standard",
            max_length=MAX_QUALITY_LENGTH,
            allowed_values=ALLOWED_QUALITY_LEVELS,
            allow_empty=False,
            allow_null=False,
        ),
        "source": FieldSchema(
            name="source",
            field_type=str,
            required=False,
            default="unknown",
            max_length=MAX_SOURCE_LENGTH,
            allowed_values=ALLOWED_SOURCES,
            allow_empty=False,
            allow_null=False,
        ),
        "metadata": FieldSchema(
            name="metadata",
            field_type=dict,
            required=False,
            default=None,
            allow_null=True,
        ),
        "audio_path": FieldSchema(
            name="audio_path",
            field_type=str,
            required=False,
            default=None,
            max_length=500,
            allow_null=True,
        ),
        "image_path": FieldSchema(
            name="image_path",
            field_type=str,
            required=False,
            default=None,
            max_length=500,
            allow_null=True,
        ),
    }

    @classmethod
    def get_field(cls, name: str) -> Optional[FieldSchema]:
        """Get field schema by name."""
        return cls.FIELDS.get(name)

    @classmethod
    def get_required_fields(cls) -> List[str]:
        """Get list of required field names."""
        return [name for name, schema in cls.FIELDS.items() if schema.required]

    @classmethod
    def get_all_fields(cls) -> List[str]:
        """Get list of all field names."""
        return list(cls.FIELDS.keys())

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Get default values for all fields."""
        return {
            name: schema.default
            for name, schema in cls.FIELDS.items()
            if schema.default is not None
        }
