"""
Response Formatter - User-safe response formatting.

This module transforms internal handler responses into user-facing output.
It ensures that internal details (stack traces, class names, config values)
are never leaked to the user.

DESIGN PRINCIPLES:
- Deterministic: Same input always produces same output
- No side effects: Pure transformation, no I/O or state changes
- Never raises: All errors handled, always returns valid output
- Security-first: Internal details are hidden from users

RESPONSIBILITIES:
- Map internal error codes to user-friendly messages
- Sanitize error messages to remove sensitive information
- Support both TEXT and JSON output modes
- Provide consistent response structure
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class OutputMode(Enum):
    """Supported output modes for formatting."""
    TEXT = "text"
    JSON = "json"


class ErrorCategory(Enum):
    """
    Categories of errors for user messaging.

    Each category has a different user-facing tone and level of detail.
    """
    VALIDATION = "validation"
    SANDBOX = "sandbox"
    LLM = "llm"
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


# =============================================================================
# Error Code Mapping
# =============================================================================

# Map internal error codes to categories
ERROR_CODE_TO_CATEGORY: Dict[str, ErrorCategory] = {
    # Validation errors
    "VALIDATION_ERROR": ErrorCategory.VALIDATION,

    # Sandbox errors
    "SANDBOX_TIMEOUT": ErrorCategory.TIMEOUT,
    "SANDBOX_CPU_EXCEEDED": ErrorCategory.SANDBOX,
    "SANDBOX_MEMORY_EXCEEDED": ErrorCategory.SANDBOX,
    "SANDBOX_EXCEPTION": ErrorCategory.INTERNAL,

    # LLM errors (from LLMLocalHandler)
    "LLM_MISSING_TEXT": ErrorCategory.VALIDATION,
    "LLM_CONNECTION_ERROR": ErrorCategory.CONNECTION,
    "LLM_TIMEOUT": ErrorCategory.TIMEOUT,
    "LLM_OLLAMA_ERROR": ErrorCategory.LLM,
    "LLM_MALFORMED_RESPONSE": ErrorCategory.LLM,
    "LLM_JSON_PARSE_ERROR": ErrorCategory.LLM,

    # Legacy handler errors
    "MISSING_TEXT": ErrorCategory.VALIDATION,
    "TIMEOUT": ErrorCategory.TIMEOUT,
    "CONNECTION_ERROR": ErrorCategory.CONNECTION,
    "OLLAMA_ERROR": ErrorCategory.LLM,

    # External handler errors
    "EXTERNAL_ERROR": ErrorCategory.INTERNAL,
    "API_ERROR": ErrorCategory.CONNECTION,

    # Rejection
    "REJECTED": ErrorCategory.VALIDATION,
}

# User-safe messages for each error code
# These messages are shown to users and must not contain internal details
USER_ERROR_MESSAGES: Dict[str, str] = {
    # Validation errors
    "VALIDATION_ERROR": "Your request could not be processed due to invalid input.",
    "LLM_MISSING_TEXT": "Please provide some text for the assistant to process.",
    "MISSING_TEXT": "Please provide some text for the assistant to process.",
    "REJECTED": "This request cannot be processed.",

    # Timeout errors
    "SANDBOX_TIMEOUT": "The request took too long to process. Please try a simpler request.",
    "LLM_TIMEOUT": "The assistant took too long to respond. Please try again.",
    "TIMEOUT": "The request timed out. Please try again.",

    # Sandbox resource errors
    "SANDBOX_CPU_EXCEEDED": "The request required too many resources. Please try a simpler request.",
    "SANDBOX_MEMORY_EXCEEDED": "The request required too much memory. Please try a simpler request.",
    "SANDBOX_EXCEPTION": "An error occurred while processing your request.",

    # Connection errors
    "LLM_CONNECTION_ERROR": "The assistant is currently unavailable. Please try again later.",
    "CONNECTION_ERROR": "A service is currently unavailable. Please try again later.",
    "API_ERROR": "An external service is unavailable. Please try again later.",

    # LLM errors
    "LLM_OLLAMA_ERROR": "The assistant encountered an issue. Please try again.",
    "LLM_MALFORMED_RESPONSE": "The assistant produced an unexpected response. Please try again.",
    "LLM_JSON_PARSE_ERROR": "The response could not be processed correctly.",
    "OLLAMA_ERROR": "The assistant encountered an issue. Please try again.",

    # External handler errors
    "EXTERNAL_ERROR": "An error occurred with an external service.",
}

# Category-level fallback messages
CATEGORY_FALLBACK_MESSAGES: Dict[ErrorCategory, str] = {
    ErrorCategory.VALIDATION: "Your input could not be validated. Please check and try again.",
    ErrorCategory.SANDBOX: "Processing limits were exceeded. Please try a simpler request.",
    ErrorCategory.LLM: "The assistant encountered an issue. Please try again.",
    ErrorCategory.CONNECTION: "A service is temporarily unavailable. Please try again later.",
    ErrorCategory.TIMEOUT: "The request timed out. Please try again.",
    ErrorCategory.INTERNAL: "An internal error occurred. Please try again.",
    ErrorCategory.UNKNOWN: "An unexpected error occurred. Please try again.",
}

# Generic fallback for completely unknown errors
GENERIC_ERROR_MESSAGE = "Something went wrong. Please try again."


# =============================================================================
# Formatted Response
# =============================================================================

@dataclass
class FormattedResponse:
    """
    User-facing formatted response.

    This is the final output shown to users, with all internal
    details removed and error messages made user-friendly.

    Attributes:
        success: Whether the request succeeded
        message: User-facing message (success or error)
        data: Response data (for successful responses)
        error_category: Category of error (for failed responses)
    """
    success: bool
    message: str
    data: Optional[Any] = None
    error_category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: Dict[str, Any] = {
            "success": self.success,
            "message": self.message,
        }

        if self.data is not None:
            result["data"] = self.data

        if self.error_category is not None:
            result["error_category"] = self.error_category

        return result

    def to_text(self) -> str:
        """Convert to plain text output."""
        if self.success:
            if self.data is not None:
                if isinstance(self.data, str):
                    return self.data
                elif isinstance(self.data, dict):
                    # Format dict nicely for text output
                    return self._format_dict_as_text(self.data)
                else:
                    return str(self.data)
            return self.message
        else:
            return f"Error: {self.message}"

    def _format_dict_as_text(self, data: Dict[str, Any]) -> str:
        """Format dictionary as readable text."""
        lines = []
        for key, value in data.items():
            if key.startswith("_"):
                continue  # Skip internal fields
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    if not k.startswith("_"):
                        lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines) if lines else str(data)

    def to_json(self) -> str:
        """Convert to JSON string output."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# =============================================================================
# Response Formatter
# =============================================================================

class ResponseFormatter:
    """
    Transforms internal responses into user-safe formatted output.

    This formatter ensures that:
    - Internal error codes are mapped to user-friendly messages
    - Stack traces and internal details are never exposed
    - Output is consistent regardless of error type
    - Both TEXT and JSON output modes are supported

    Usage:
        formatter = ResponseFormatter()
        formatted = formatter.format(internal_response)
        print(formatted.to_text())  # or formatted.to_json()
    """

    def __init__(self, default_mode: OutputMode = OutputMode.TEXT):
        """
        Initialize the formatter.

        Args:
            default_mode: Default output mode when not specified
        """
        self._default_mode = default_mode

    def format(
        self,
        response: Dict[str, Any],
        mode: Optional[OutputMode] = None,
    ) -> FormattedResponse:
        """
        Format an internal response for user output.

        Args:
            response: Internal response dictionary from handler
            mode: Output mode (TEXT or JSON), uses default if not specified

        Returns:
            FormattedResponse ready for user output

        Note:
            This method NEVER raises exceptions. Invalid input
            produces a generic error response.
        """
        try:
            return self._format_internal(response, mode or self._default_mode)
        except Exception:
            # Never raise - return generic error
            return FormattedResponse(
                success=False,
                message=GENERIC_ERROR_MESSAGE,
                error_category=ErrorCategory.UNKNOWN.value,
            )

    def _format_internal(
        self,
        response: Dict[str, Any],
        mode: OutputMode,
    ) -> FormattedResponse:
        """
        Internal formatting logic.

        Args:
            response: Internal response dictionary
            mode: Output mode

        Returns:
            FormattedResponse
        """
        # Handle None or non-dict input
        if response is None:
            return FormattedResponse(
                success=False,
                message=GENERIC_ERROR_MESSAGE,
                error_category=ErrorCategory.UNKNOWN.value,
            )

        if not isinstance(response, dict):
            return FormattedResponse(
                success=False,
                message=GENERIC_ERROR_MESSAGE,
                error_category=ErrorCategory.UNKNOWN.value,
            )

        # Check success flag
        success = response.get("success", False)

        if success:
            return self._format_success(response, mode)
        else:
            return self._format_error(response)

    def _format_success(
        self,
        response: Dict[str, Any],
        mode: OutputMode,
    ) -> FormattedResponse:
        """
        Format a successful response.

        Args:
            response: Successful internal response
            mode: Output mode

        Returns:
            FormattedResponse with data
        """
        # Extract the main output
        data = self._extract_output(response)

        # Build success message
        if data is not None:
            if isinstance(data, str):
                message = data[:100] + "..." if len(data) > 100 else data
            else:
                message = "Request completed successfully."
        else:
            message = "Request completed successfully."

        return FormattedResponse(
            success=True,
            message=message,
            data=data,
        )

    def _extract_output(self, response: Dict[str, Any]) -> Optional[Any]:
        """
        Extract the main output from response.

        Looks for output in order: text, output, audio_path, image_path

        Args:
            response: Internal response

        Returns:
            Extracted output or None
        """
        # Priority order for output extraction
        if "text" in response and response["text"]:
            return response["text"]

        if "output" in response and response["output"] is not None:
            output = response["output"]
            # If output is a dict with internal keys, clean it
            if isinstance(output, dict):
                return self._clean_output_dict(output)
            return output

        if "audio_path" in response and response["audio_path"]:
            return {"audio_file": response["audio_path"]}

        if "image_path" in response and response["image_path"]:
            return {"image_file": response["image_path"]}

        return None

    def _clean_output_dict(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a dictionary output by removing internal fields.

        Args:
            output: Dictionary to clean

        Returns:
            Cleaned dictionary
        """
        # Remove keys that start with underscore (internal)
        cleaned = {}
        for key, value in output.items():
            if not key.startswith("_"):
                if isinstance(value, dict):
                    cleaned[key] = self._clean_output_dict(value)
                else:
                    cleaned[key] = value
        return cleaned

    def _format_error(self, response: Dict[str, Any]) -> FormattedResponse:
        """
        Format an error response with user-safe message.

        Args:
            response: Failed internal response

        Returns:
            FormattedResponse with user-safe error message
        """
        error_code = response.get("error_code", "")
        internal_error = response.get("error", "")

        # Get user message for this error code
        user_message = self._get_user_message(error_code, internal_error)

        # Get error category
        category = self._get_error_category(error_code)

        return FormattedResponse(
            success=False,
            message=user_message,
            error_category=category.value,
        )

    def _get_user_message(self, error_code: str, internal_error: str) -> str:
        """
        Get user-safe message for an error.

        Args:
            error_code: Internal error code
            internal_error: Internal error message (not shown to user)

        Returns:
            User-safe error message
        """
        # Try exact error code match
        if error_code and error_code in USER_ERROR_MESSAGES:
            return USER_ERROR_MESSAGES[error_code]

        # Try category fallback
        category = self._get_error_category(error_code)
        if category in CATEGORY_FALLBACK_MESSAGES:
            return CATEGORY_FALLBACK_MESSAGES[category]

        # Generic fallback
        return GENERIC_ERROR_MESSAGE

    def _get_error_category(self, error_code: str) -> ErrorCategory:
        """
        Determine error category from error code.

        Args:
            error_code: Internal error code

        Returns:
            ErrorCategory for the error
        """
        if not error_code:
            return ErrorCategory.UNKNOWN

        # Try exact match
        if error_code in ERROR_CODE_TO_CATEGORY:
            return ERROR_CODE_TO_CATEGORY[error_code]

        # Try prefix matching for unknown codes
        error_upper = error_code.upper()

        if "VALIDATION" in error_upper or "INVALID" in error_upper:
            return ErrorCategory.VALIDATION

        if "TIMEOUT" in error_upper:
            return ErrorCategory.TIMEOUT

        if "SANDBOX" in error_upper:
            return ErrorCategory.SANDBOX

        if "CONNECTION" in error_upper or "CONNECT" in error_upper:
            return ErrorCategory.CONNECTION

        if "LLM" in error_upper or "OLLAMA" in error_upper:
            return ErrorCategory.LLM

        return ErrorCategory.UNKNOWN

    def format_validation_errors(
        self,
        errors: List[Dict[str, Any]],
    ) -> FormattedResponse:
        """
        Format validation errors into user-friendly message.

        Args:
            errors: List of validation error dictionaries

        Returns:
            FormattedResponse with combined validation message
        """
        if not errors:
            return FormattedResponse(
                success=False,
                message=USER_ERROR_MESSAGES["VALIDATION_ERROR"],
                error_category=ErrorCategory.VALIDATION.value,
            )

        # Build user-friendly list of issues
        issues = []
        for error in errors:
            field = error.get("field", "input")
            # Don't expose the raw error message, just the field
            issues.append(f"Invalid {field}")

        if len(issues) == 1:
            message = f"Please check your input: {issues[0].lower()}."
        else:
            message = f"Please check your input. Issues found: {', '.join(issues).lower()}."

        return FormattedResponse(
            success=False,
            message=message,
            error_category=ErrorCategory.VALIDATION.value,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def format_response(
    response: Dict[str, Any],
    mode: OutputMode = OutputMode.TEXT,
) -> FormattedResponse:
    """
    Format a response using the default formatter.

    Args:
        response: Internal response dictionary
        mode: Output mode (TEXT or JSON)

    Returns:
        FormattedResponse ready for output
    """
    formatter = ResponseFormatter(default_mode=mode)
    return formatter.format(response)


def format_as_text(response: Dict[str, Any]) -> str:
    """
    Format a response as plain text.

    Args:
        response: Internal response dictionary

    Returns:
        Plain text output string
    """
    formatted = format_response(response, OutputMode.TEXT)
    return formatted.to_text()


def format_as_json(response: Dict[str, Any]) -> str:
    """
    Format a response as JSON string.

    Args:
        response: Internal response dictionary

    Returns:
        JSON string output
    """
    formatted = format_response(response, OutputMode.JSON)
    return formatted.to_json()
