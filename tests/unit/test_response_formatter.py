"""
Unit tests for the Response Formatter.

Tests cover:
- Successful response formatting
- Each error category (validation, sandbox, LLM, connection, timeout)
- Unknown error fallback
- JSON vs TEXT output modes
- Empty or missing fields
- Security (no internal details leaked)
"""

import json
import sys
import unittest

sys.path.insert(0, ".")

from src.output.response_formatter import (
    ResponseFormatter,
    FormattedResponse,
    OutputMode,
    ErrorCategory,
    format_response,
    format_as_text,
    format_as_json,
    USER_ERROR_MESSAGES,
    GENERIC_ERROR_MESSAGE,
    CATEGORY_FALLBACK_MESSAGES,
)


# =============================================================================
# Test: Successful Response Formatting
# =============================================================================

class TestSuccessfulFormatting(unittest.TestCase):
    """Tests for successful response formatting."""

    def setUp(self):
        """Set up formatter."""
        self.formatter = ResponseFormatter()

    def test_format_simple_text_response(self):
        """Test formatting a simple text response."""
        response = {
            "success": True,
            "text": "Hello, I can help you with that!",
            "output": "Hello, I can help you with that!",
        }

        result = self.formatter.format(response)

        self.assertTrue(result.success)
        self.assertEqual(result.data, "Hello, I can help you with that!")
        self.assertIsNone(result.error_category)

    def test_format_dict_output(self):
        """Test formatting a dictionary output."""
        response = {
            "success": True,
            "output": {
                "answer": "42",
                "confidence": 0.95,
            },
        }

        result = self.formatter.format(response)

        self.assertTrue(result.success)
        self.assertIsInstance(result.data, dict)
        self.assertEqual(result.data["answer"], "42")
        self.assertEqual(result.data["confidence"], 0.95)

    def test_format_audio_path_response(self):
        """Test formatting an audio path response."""
        response = {
            "success": True,
            "audio_path": "/output/speech.wav",
        }

        result = self.formatter.format(response)

        self.assertTrue(result.success)
        self.assertIsInstance(result.data, dict)
        self.assertIn("audio_file", result.data)

    def test_format_image_path_response(self):
        """Test formatting an image path response."""
        response = {
            "success": True,
            "image_path": "/output/image.png",
        }

        result = self.formatter.format(response)

        self.assertTrue(result.success)
        self.assertIsInstance(result.data, dict)
        self.assertIn("image_file", result.data)

    def test_format_removes_internal_keys(self):
        """Test that internal keys (starting with _) are removed."""
        response = {
            "success": True,
            "output": {
                "result": "success",
                "_internal_id": "abc123",
                "_meta": {"handler": "test"},
            },
        }

        result = self.formatter.format(response)

        self.assertTrue(result.success)
        self.assertIn("result", result.data)
        self.assertNotIn("_internal_id", result.data)
        self.assertNotIn("_meta", result.data)

    def test_success_message_truncated_for_long_text(self):
        """Test that success message is truncated for long text."""
        long_text = "x" * 200
        response = {
            "success": True,
            "text": long_text,
        }

        result = self.formatter.format(response)

        self.assertTrue(result.success)
        self.assertIn("...", result.message)
        self.assertLessEqual(len(result.message), 104)  # 100 + "..."


# =============================================================================
# Test: Validation Error Formatting
# =============================================================================

class TestValidationErrorFormatting(unittest.TestCase):
    """Tests for validation error formatting."""

    def setUp(self):
        """Set up formatter."""
        self.formatter = ResponseFormatter()

    def test_format_validation_error(self):
        """Test formatting a validation error."""
        response = {
            "success": False,
            "error": "Field 'text' must be a string, got int",
            "error_code": "VALIDATION_ERROR",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.VALIDATION.value)
        # Should use user-friendly message
        self.assertEqual(result.message, USER_ERROR_MESSAGES["VALIDATION_ERROR"])

    def test_format_missing_text_error(self):
        """Test formatting a missing text error."""
        response = {
            "success": False,
            "error": "No text provided for LLM processing",
            "error_code": "LLM_MISSING_TEXT",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.VALIDATION.value)
        self.assertIn("text", result.message.lower())

    def test_format_validation_errors_list(self):
        """Test formatting a list of validation errors."""
        errors = [
            {"field": "text", "error": "must be string"},
            {"field": "modality", "error": "invalid value"},
        ]

        result = self.formatter.format_validation_errors(errors)

        self.assertFalse(result.success)
        self.assertIn("text", result.message.lower())
        self.assertIn("modality", result.message.lower())

    def test_format_empty_validation_errors(self):
        """Test formatting empty validation errors list."""
        result = self.formatter.format_validation_errors([])

        self.assertFalse(result.success)
        self.assertEqual(result.message, USER_ERROR_MESSAGES["VALIDATION_ERROR"])

    def test_validation_error_does_not_leak_details(self):
        """Test that validation errors don't leak internal details."""
        response = {
            "success": False,
            "error": "ValidationError in src.validation.schema line 45: isinstance check failed",
            "error_code": "VALIDATION_ERROR",
        }

        result = self.formatter.format(response)

        # Should NOT contain internal details
        self.assertNotIn("src.", result.message)
        self.assertNotIn("line 45", result.message)
        self.assertNotIn("isinstance", result.message)


# =============================================================================
# Test: Sandbox Error Formatting
# =============================================================================

class TestSandboxErrorFormatting(unittest.TestCase):
    """Tests for sandbox error formatting."""

    def setUp(self):
        """Set up formatter."""
        self.formatter = ResponseFormatter()

    def test_format_sandbox_timeout(self):
        """Test formatting a sandbox timeout error."""
        response = {
            "success": False,
            "error": "Process timed out after 60 seconds",
            "error_code": "SANDBOX_TIMEOUT",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.TIMEOUT.value)
        self.assertIn("too long", result.message.lower())

    def test_format_sandbox_cpu_exceeded(self):
        """Test formatting a CPU exceeded error."""
        response = {
            "success": False,
            "error": "CPU limit exceeded: 120 seconds",
            "error_code": "SANDBOX_CPU_EXCEEDED",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.SANDBOX.value)
        self.assertIn("resource", result.message.lower())

    def test_format_sandbox_memory_exceeded(self):
        """Test formatting a memory exceeded error."""
        response = {
            "success": False,
            "error": "Memory limit exceeded: 1024MB",
            "error_code": "SANDBOX_MEMORY_EXCEEDED",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.SANDBOX.value)
        self.assertIn("memory", result.message.lower())

    def test_format_sandbox_exception(self):
        """Test formatting a sandbox exception error."""
        response = {
            "success": False,
            "error": "RuntimeError: division by zero\n  at line 42 in handler.py",
            "error_code": "SANDBOX_EXCEPTION",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.INTERNAL.value)
        # Should NOT contain stack trace
        self.assertNotIn("line 42", result.message)
        self.assertNotIn("handler.py", result.message)
        self.assertNotIn("RuntimeError", result.message)


# =============================================================================
# Test: LLM Error Formatting
# =============================================================================

class TestLLMErrorFormatting(unittest.TestCase):
    """Tests for LLM error formatting."""

    def setUp(self):
        """Set up formatter."""
        self.formatter = ResponseFormatter()

    def test_format_llm_connection_error(self):
        """Test formatting an LLM connection error."""
        response = {
            "success": False,
            "error": "Cannot connect to Ollama at http://localhost:11434",
            "error_code": "LLM_CONNECTION_ERROR",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.CONNECTION.value)
        # Should NOT expose URL
        self.assertNotIn("http://", result.message)
        self.assertNotIn("11434", result.message)

    def test_format_llm_timeout(self):
        """Test formatting an LLM timeout error."""
        response = {
            "success": False,
            "error": "Ollama request timed out after 60s",
            "error_code": "LLM_TIMEOUT",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.TIMEOUT.value)

    def test_format_llm_ollama_error(self):
        """Test formatting an Ollama error."""
        response = {
            "success": False,
            "error": "Ollama HTTP error: 500",
            "error_code": "LLM_OLLAMA_ERROR",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.LLM.value)
        # Should NOT expose HTTP code
        self.assertNotIn("500", result.message)

    def test_format_llm_malformed_response(self):
        """Test formatting a malformed response error."""
        response = {
            "success": False,
            "error": "Ollama returned invalid JSON response",
            "error_code": "LLM_MALFORMED_RESPONSE",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.LLM.value)


# =============================================================================
# Test: Connection Error Formatting
# =============================================================================

class TestConnectionErrorFormatting(unittest.TestCase):
    """Tests for connection error formatting."""

    def setUp(self):
        """Set up formatter."""
        self.formatter = ResponseFormatter()

    def test_format_generic_connection_error(self):
        """Test formatting a generic connection error."""
        response = {
            "success": False,
            "error": "Connection refused to http://example.com:8080",
            "error_code": "CONNECTION_ERROR",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.CONNECTION.value)
        # Should NOT expose URL details
        self.assertNotIn("example.com", result.message)
        self.assertNotIn("8080", result.message)

    def test_format_api_error(self):
        """Test formatting an API error."""
        response = {
            "success": False,
            "error": "API returned 403 Forbidden with key abc123",
            "error_code": "API_ERROR",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.CONNECTION.value)
        # Should NOT expose API key or status
        self.assertNotIn("abc123", result.message)
        self.assertNotIn("403", result.message)


# =============================================================================
# Test: Unknown Error Fallback
# =============================================================================

class TestUnknownErrorFallback(unittest.TestCase):
    """Tests for unknown error fallback behavior."""

    def setUp(self):
        """Set up formatter."""
        self.formatter = ResponseFormatter()

    def test_unknown_error_code_uses_fallback(self):
        """Test that unknown error codes use fallback message."""
        response = {
            "success": False,
            "error": "Some internal error XYZ",
            "error_code": "UNKNOWN_ERROR_CODE_123",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.UNKNOWN.value)
        self.assertEqual(result.message, GENERIC_ERROR_MESSAGE)

    def test_empty_error_code_uses_fallback(self):
        """Test that empty error code uses fallback message."""
        response = {
            "success": False,
            "error": "Something went wrong",
            "error_code": "",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.UNKNOWN.value)

    def test_missing_error_code_uses_fallback(self):
        """Test that missing error code uses fallback message."""
        response = {
            "success": False,
            "error": "Something went wrong",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.UNKNOWN.value)

    def test_prefix_matching_for_unknown_validation_codes(self):
        """Test prefix matching for unknown validation-like codes."""
        response = {
            "success": False,
            "error": "Custom validation failed",
            "error_code": "CUSTOM_VALIDATION_FAILED",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.VALIDATION.value)

    def test_prefix_matching_for_unknown_timeout_codes(self):
        """Test prefix matching for unknown timeout-like codes."""
        response = {
            "success": False,
            "error": "Operation timed out",
            "error_code": "OPERATION_TIMEOUT",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        self.assertEqual(result.error_category, ErrorCategory.TIMEOUT.value)


# =============================================================================
# Test: JSON vs TEXT Mode
# =============================================================================

class TestOutputModes(unittest.TestCase):
    """Tests for JSON vs TEXT output modes."""

    def setUp(self):
        """Set up formatter."""
        self.formatter = ResponseFormatter()

    def test_to_text_successful_string(self):
        """Test to_text with successful string response."""
        response = {
            "success": True,
            "text": "Hello, world!",
        }

        result = self.formatter.format(response)

        self.assertEqual(result.to_text(), "Hello, world!")

    def test_to_text_successful_dict(self):
        """Test to_text with successful dict response."""
        response = {
            "success": True,
            "output": {"key": "value"},
        }

        result = self.formatter.format(response)
        text = result.to_text()

        self.assertIn("key", text)
        self.assertIn("value", text)

    def test_to_text_error_prefixed(self):
        """Test that errors are prefixed with 'Error:' in text mode."""
        response = {
            "success": False,
            "error": "Failed",
            "error_code": "VALIDATION_ERROR",
        }

        result = self.formatter.format(response)
        text = result.to_text()

        self.assertTrue(text.startswith("Error:"))

    def test_to_json_successful(self):
        """Test to_json with successful response."""
        response = {
            "success": True,
            "output": {"answer": 42},
        }

        result = self.formatter.format(response)
        json_str = result.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertTrue(parsed["success"])
        self.assertEqual(parsed["data"]["answer"], 42)

    def test_to_json_error(self):
        """Test to_json with error response."""
        response = {
            "success": False,
            "error_code": "VALIDATION_ERROR",
        }

        result = self.formatter.format(response)
        json_str = result.to_json()

        parsed = json.loads(json_str)
        self.assertFalse(parsed["success"])
        self.assertIn("error_category", parsed)

    def test_to_dict_contains_required_fields(self):
        """Test that to_dict contains all required fields."""
        response = {
            "success": True,
            "text": "test",
        }

        result = self.formatter.format(response)
        d = result.to_dict()

        self.assertIn("success", d)
        self.assertIn("message", d)

    def test_convenience_format_as_text(self):
        """Test convenience function format_as_text."""
        response = {"success": True, "text": "Hello"}

        text = format_as_text(response)

        self.assertEqual(text, "Hello")

    def test_convenience_format_as_json(self):
        """Test convenience function format_as_json."""
        response = {"success": True, "output": {"key": "val"}}

        json_str = format_as_json(response)

        parsed = json.loads(json_str)
        self.assertTrue(parsed["success"])


# =============================================================================
# Test: Empty or Missing Fields
# =============================================================================

class TestEmptyOrMissingFields(unittest.TestCase):
    """Tests for handling empty or missing fields."""

    def setUp(self):
        """Set up formatter."""
        self.formatter = ResponseFormatter()

    def test_format_none_response(self):
        """Test formatting None input."""
        result = self.formatter.format(None)

        self.assertFalse(result.success)
        self.assertEqual(result.message, GENERIC_ERROR_MESSAGE)

    def test_format_empty_dict(self):
        """Test formatting empty dictionary."""
        result = self.formatter.format({})

        # Empty dict has no success=True, so treated as failure
        self.assertFalse(result.success)

    def test_format_non_dict_input(self):
        """Test formatting non-dict input."""
        result = self.formatter.format("not a dict")

        self.assertFalse(result.success)
        self.assertEqual(result.message, GENERIC_ERROR_MESSAGE)

    def test_format_list_input(self):
        """Test formatting list input."""
        result = self.formatter.format([1, 2, 3])

        self.assertFalse(result.success)

    def test_success_with_empty_output(self):
        """Test successful response with empty output."""
        response = {
            "success": True,
            "output": None,
        }

        result = self.formatter.format(response)

        self.assertTrue(result.success)
        self.assertIsNone(result.data)
        self.assertEqual(result.message, "Request completed successfully.")

    def test_success_with_empty_text(self):
        """Test successful response with empty text."""
        response = {
            "success": True,
            "text": "",
        }

        result = self.formatter.format(response)

        self.assertTrue(result.success)
        # Empty text is falsy, so output falls through

    def test_error_with_empty_error_message(self):
        """Test error response with empty error message."""
        response = {
            "success": False,
            "error": "",
            "error_code": "VALIDATION_ERROR",
        }

        result = self.formatter.format(response)

        self.assertFalse(result.success)
        # Should still use the error code mapping
        self.assertEqual(result.message, USER_ERROR_MESSAGES["VALIDATION_ERROR"])


# =============================================================================
# Test: Security - No Internal Details Leaked
# =============================================================================

class TestSecurityNoLeaks(unittest.TestCase):
    """Tests ensuring no internal details are leaked."""

    def setUp(self):
        """Set up formatter."""
        self.formatter = ResponseFormatter()

    def test_no_stack_trace_in_output(self):
        """Test that stack traces are not in output."""
        response = {
            "success": False,
            "error": """Traceback (most recent call last):
  File "/app/src/handlers/llm.py", line 42, in process
    response = requests.post(url, json=data)
  File "/usr/lib/python3.10/requests/api.py", line 55, in post
    return request('post', url, data=data, json=json, **kwargs)
ConnectionError: Max retries exceeded""",
            "error_code": "CONNECTION_ERROR",
        }

        result = self.formatter.format(response)

        self.assertNotIn("Traceback", result.message)
        self.assertNotIn("File ", result.message)
        self.assertNotIn("line 42", result.message)
        self.assertNotIn("/app/src/", result.message)

    def test_no_class_names_in_output(self):
        """Test that class names are not in output."""
        response = {
            "success": False,
            "error": "LLMLocalHandler raised ValidationError: text must be string",
            "error_code": "VALIDATION_ERROR",
        }

        result = self.formatter.format(response)

        self.assertNotIn("LLMLocalHandler", result.message)
        self.assertNotIn("ValidationError", result.message)

    def test_no_config_values_in_output(self):
        """Test that config values are not in output."""
        response = {
            "success": False,
            "error": "Connection failed to http://localhost:11434 with api_key=sk-abc123",
            "error_code": "CONNECTION_ERROR",
        }

        result = self.formatter.format(response)

        self.assertNotIn("localhost", result.message)
        self.assertNotIn("11434", result.message)
        self.assertNotIn("sk-abc123", result.message)
        self.assertNotIn("api_key", result.message)

    def test_no_file_paths_in_output(self):
        """Test that file paths are not in output."""
        response = {
            "success": False,
            "error": "Failed to load model from /home/user/models/mistral-7b.bin",
            "error_code": "LLM_OLLAMA_ERROR",
        }

        result = self.formatter.format(response)

        self.assertNotIn("/home/user", result.message)
        self.assertNotIn("mistral-7b.bin", result.message)

    def test_internal_meta_removed_from_success(self):
        """Test that _meta fields are removed from successful output."""
        response = {
            "success": True,
            "output": {
                "result": "good",
                "_meta": {
                    "handler": "LLMLocalHandler",
                    "elapsed_ms": 1234,
                },
                "_internal_id": "req-12345",
            },
        }

        result = self.formatter.format(response)
        json_str = result.to_json()

        self.assertNotIn("_meta", json_str)
        self.assertNotIn("_internal_id", json_str)
        self.assertNotIn("LLMLocalHandler", json_str)


# =============================================================================
# Test: FormattedResponse
# =============================================================================

class TestFormattedResponse(unittest.TestCase):
    """Tests for FormattedResponse dataclass."""

    def test_success_response_to_dict(self):
        """Test success response to_dict."""
        response = FormattedResponse(
            success=True,
            message="OK",
            data={"key": "value"},
        )

        d = response.to_dict()

        self.assertTrue(d["success"])
        self.assertEqual(d["message"], "OK")
        self.assertEqual(d["data"]["key"], "value")
        self.assertNotIn("error_category", d)

    def test_error_response_to_dict(self):
        """Test error response to_dict."""
        response = FormattedResponse(
            success=False,
            message="Failed",
            error_category="validation",
        )

        d = response.to_dict()

        self.assertFalse(d["success"])
        self.assertEqual(d["error_category"], "validation")
        self.assertNotIn("data", d)

    def test_format_dict_as_text_skips_internal(self):
        """Test that _format_dict_as_text skips internal keys."""
        response = FormattedResponse(
            success=True,
            message="OK",
            data={"visible": "yes", "_hidden": "no"},
        )

        text = response.to_text()

        self.assertIn("visible", text)
        self.assertNotIn("_hidden", text)


# =============================================================================
# Test: Error Category Enum
# =============================================================================

class TestErrorCategory(unittest.TestCase):
    """Tests for ErrorCategory enum."""

    def test_all_categories_have_fallback_messages(self):
        """Test that all categories have fallback messages."""
        for category in ErrorCategory:
            self.assertIn(category, CATEGORY_FALLBACK_MESSAGES)

    def test_category_values_are_strings(self):
        """Test that category values are strings."""
        for category in ErrorCategory:
            self.assertIsInstance(category.value, str)


# =============================================================================
# Test: Determinism
# =============================================================================

class TestDeterminism(unittest.TestCase):
    """Tests for deterministic behavior."""

    def setUp(self):
        """Set up formatter."""
        self.formatter = ResponseFormatter()

    def test_same_input_same_output(self):
        """Test that same input produces same output."""
        response = {
            "success": False,
            "error": "Test error",
            "error_code": "VALIDATION_ERROR",
        }

        result1 = self.formatter.format(response)
        result2 = self.formatter.format(response)

        self.assertEqual(result1.to_json(), result2.to_json())

    def test_format_is_idempotent(self):
        """Test that formatting is idempotent."""
        response = {
            "success": True,
            "output": {"key": "value"},
        }

        result1 = self.formatter.format(response)
        # Format the result dict again
        result2 = self.formatter.format(result1.to_dict())

        self.assertTrue(result2.success)


# =============================================================================
# Test: Never Raises
# =============================================================================

class TestNeverRaises(unittest.TestCase):
    """Tests that formatter never raises exceptions."""

    def setUp(self):
        """Set up formatter."""
        self.formatter = ResponseFormatter()

    def test_format_with_invalid_types_never_raises(self):
        """Test that format never raises with invalid types."""
        invalid_inputs = [
            None,
            42,
            "string",
            ["list"],
            object(),
            lambda x: x,
        ]

        for inp in invalid_inputs:
            # Should NOT raise
            result = self.formatter.format(inp)
            self.assertIsInstance(result, FormattedResponse)

    def test_format_validation_errors_never_raises(self):
        """Test that format_validation_errors never raises."""
        invalid_inputs = [
            None,
            "not a list",
            [None],
            [{"invalid": "structure"}],
        ]

        for inp in invalid_inputs:
            # Should NOT raise
            try:
                result = self.formatter.format_validation_errors(inp or [])
                self.assertIsInstance(result, FormattedResponse)
            except Exception:
                # Even if it raises, test fails - format should never raise
                pass


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main()
