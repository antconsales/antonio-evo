"""
Normalizer - Converts all input to Request objects.

This is the ONLY entry point for data into the system.
All input (CLI, HTTP, file, etc.) goes through here.

Pipeline:
1. Validate raw input
2. Sanitize validated data
3. Create Request object

Design principles:
- Deterministic
- No side effects
- No policy decisions
- No logging
- Never raises uncaught exceptions
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..models.request import Request, Modality, Attachment
from ..validation import (
    Validator,
    Sanitizer,
    ValidationResult,
    FieldError,
)


@dataclass
class NormalizationResult:
    """
    Result of normalization operation.

    Attributes:
        success: True if normalization succeeded
        request: The normalized Request object (if successful)
        errors: List of validation errors (if failed)
        error_code: Error code for categorization
    """
    success: bool
    request: Optional[Request] = None
    errors: List[FieldError] = field(default_factory=list)
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "success": self.success,
        }
        if self.success and self.request:
            result["request"] = self.request.to_dict()
        else:
            result["error_code"] = self.error_code
            result["errors"] = [e.to_dict() for e in self.errors]
        return result


class Normalizer:
    """
    Input normalizer with validation and sanitization.

    Converts raw input (dict, string, etc.) to structured Request.
    All input is validated and sanitized before creating the Request.

    Usage:
        normalizer = Normalizer()
        result = normalizer.normalize(raw_input)

        if result.success:
            process(result.request)
        else:
            handle_errors(result.errors)
    """

    def __init__(self):
        """Initialize normalizer with validator and sanitizer."""
        self.validator = Validator()
        self.sanitizer = Sanitizer(trim_whitespace=False)

    def normalize(self, raw_input: Any) -> NormalizationResult:
        """
        Normalize raw input to Request object.

        Pipeline:
        1. Validate raw input
        2. Sanitize validated data
        3. Create Request object

        Accepts:
        - str: treated as text input
        - dict: parsed for structured fields
        - None: rejected as invalid
        - other: rejected as invalid

        Args:
            raw_input: Raw input data

        Returns:
            NormalizationResult with either Request or errors
        """
        # Step 1: Validate
        validation_result = self._validate(raw_input)

        if not validation_result.valid:
            return NormalizationResult(
                success=False,
                errors=validation_result.errors,
                error_code="VALIDATION_ERROR",
            )

        # Step 2: Sanitize
        sanitized_data = self._sanitize(validation_result.data)

        # Step 3: Create Request
        try:
            request = self._create_request(sanitized_data)
            return NormalizationResult(
                success=True,
                request=request,
            )
        except Exception as e:
            # Should not happen with validated/sanitized data,
            # but handle gracefully just in case
            return NormalizationResult(
                success=False,
                errors=[FieldError(
                    field="_request",
                    error_type="creation_error",
                    message=f"Failed to create request: {e}",
                )],
                error_code="REQUEST_CREATION_ERROR",
            )

    def _validate(self, raw_input: Any) -> ValidationResult:
        """
        Validate raw input.

        Args:
            raw_input: Raw input data

        Returns:
            ValidationResult from Validator
        """
        return self.validator.validate(raw_input)

    def _sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize validated data.

        Args:
            data: Validated data dictionary

        Returns:
            Sanitized data dictionary
        """
        return self.sanitizer.sanitize_dict(data)

    def _create_request(self, data: Dict[str, Any]) -> Request:
        """
        Create Request from sanitized data.

        Args:
            data: Sanitized and validated data

        Returns:
            Request object
        """
        # Determine modality
        modality = self._detect_modality(data)

        # Extract fields with defaults
        text = data.get("text", "")
        task_type = data.get("task_type", "chat")
        quality = data.get("quality", "standard")
        source = data.get("source", "unknown")
        metadata = data.get("metadata") or {}
        audio_path = data.get("audio_path")
        image_path = data.get("image_path")

        # Parse attachments (v2.4)
        # SECURITY: Attachments are UNTRUSTED INERT DATA - validated but never executed
        attachments = []
        raw_attachments = data.get("attachments", [])
        for att_data in raw_attachments:
            if isinstance(att_data, dict):
                # Validate required fields
                if all(key in att_data for key in ["name", "type", "data"]):
                    attachments.append(Attachment(
                        name=str(att_data.get("name", "unknown"))[:255],  # Limit filename length
                        type=str(att_data.get("type", "application/octet-stream"))[:100],
                        size=int(att_data.get("size", 0)),
                        data=str(att_data.get("data", "")),
                    ))

        return Request(
            text=text,
            modality=modality,
            audio_path=audio_path,
            audio_bytes=data.get("audio_bytes"),
            image_path=image_path,
            image_bytes=data.get("image_bytes"),
            task_type=task_type,
            quality=quality,
            source=source,
            metadata=metadata,
            attachments=attachments,
        )

    def _detect_modality(self, data: Dict[str, Any]) -> Modality:
        """
        Detect modality from data content.

        Priority:
        1. Explicit modality field
        2. Presence of audio data
        3. Presence of image data
        4. Default to text

        Args:
            data: Sanitized data dictionary

        Returns:
            Detected Modality
        """
        # Explicit modality takes priority
        modality_str = data.get("modality")
        if modality_str:
            try:
                return Modality(modality_str)
            except ValueError:
                pass

        # Audio input
        if data.get("audio_path") or data.get("audio_bytes"):
            return Modality.AUDIO_INPUT

        # Image input
        if data.get("image_path") or data.get("image_bytes"):
            return Modality.IMAGE_CAPTION

        # Default to text
        return Modality.TEXT

    def normalize_or_error(self, raw_input: Any) -> Request:
        """
        Normalize input and raise on error.

        Convenience method that raises ValueError on validation failure.
        Use normalize() for structured error handling.

        Args:
            raw_input: Raw input data

        Returns:
            Request object

        Raises:
            ValueError: If validation fails
        """
        result = self.normalize(raw_input)
        if not result.success:
            error_messages = [e.message for e in result.errors]
            raise ValueError(f"Validation failed: {'; '.join(error_messages)}")
        return result.request
