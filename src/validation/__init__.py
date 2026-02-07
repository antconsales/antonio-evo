"""
Validation module for input validation and sanitization.

Provides schema-based validation and sanitization for all input before processing.
"""

from .schema import (
    ValidationError,
    ValidationResult,
    FieldError,
    RequestSchema,
    ALLOWED_MODALITIES,
    ALLOWED_TASK_TYPES,
    ALLOWED_QUALITY_LEVELS,
    MAX_TEXT_LENGTH,
    MAX_METADATA_SIZE,
    MAX_SOURCE_LENGTH,
)

from .validator import (
    Validator,
)

from .sanitizer import (
    Sanitizer,
    sanitize_text,
    sanitize_dict,
)

__all__ = [
    # Schema
    "ValidationError",
    "ValidationResult",
    "FieldError",
    "RequestSchema",
    "ALLOWED_MODALITIES",
    "ALLOWED_TASK_TYPES",
    "ALLOWED_QUALITY_LEVELS",
    "MAX_TEXT_LENGTH",
    "MAX_METADATA_SIZE",
    "MAX_SOURCE_LENGTH",
    # Validator
    "Validator",
    # Sanitizer
    "Sanitizer",
    "sanitize_text",
    "sanitize_dict",
]
