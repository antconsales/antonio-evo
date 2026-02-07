"""
Unit tests for the Local LLM Handler.

Tests cover:
- Successful text response
- Successful JSON response
- Ollama unavailable (connection error)
- Timeout handling
- Malformed response handling
- Missing text input
- Configuration handling
"""

import json
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, ".")

from src.handlers.llm_local import (
    LLMLocalHandler,
    LLMLocalConfig,
    LLMLocalErrorCode,
    OutputFormat,
    SYSTEM_PROMPT,
)
from src.models.request import Request, Modality


# =============================================================================
# Test: Configuration
# =============================================================================

class TestLLMLocalConfig(unittest.TestCase):
    """Tests for LLMLocalConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMLocalConfig()

        self.assertEqual(config.base_url, "http://localhost:11434")
        self.assertEqual(config.model, "mistral")
        self.assertEqual(config.timeout_seconds, 60)
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.max_tokens, 1024)
        self.assertEqual(config.output_format, OutputFormat.TEXT)

    def test_from_dict_with_values(self):
        """Test creating config from dictionary."""
        config = LLMLocalConfig.from_dict({
            "base_url": "http://custom:8080",
            "model": "llama2",
            "timeout_seconds": 120,
            "temperature": 0.5,
            "max_tokens": 2048,
            "output_format": "json",
        })

        self.assertEqual(config.base_url, "http://custom:8080")
        self.assertEqual(config.model, "llama2")
        self.assertEqual(config.timeout_seconds, 120)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 2048)
        self.assertEqual(config.output_format, OutputFormat.JSON)

    def test_from_dict_with_timeout_alias(self):
        """Test that 'timeout' is accepted as alias for 'timeout_seconds'."""
        config = LLMLocalConfig.from_dict({"timeout": 90})

        self.assertEqual(config.timeout_seconds, 90)

    def test_from_dict_invalid_output_format(self):
        """Test that invalid output format defaults to TEXT."""
        config = LLMLocalConfig.from_dict({"output_format": "invalid"})

        self.assertEqual(config.output_format, OutputFormat.TEXT)

    def test_from_dict_empty(self):
        """Test creating config from empty dictionary uses defaults."""
        config = LLMLocalConfig.from_dict({})

        self.assertEqual(config.base_url, "http://localhost:11434")
        self.assertEqual(config.model, "mistral")


# =============================================================================
# Test: Successful Response
# =============================================================================

class TestLLMLocalSuccessfulResponse(unittest.TestCase):
    """Tests for successful LLM responses."""

    def setUp(self):
        """Set up handler with default config."""
        self.handler = LLMLocalHandler({})
        self.request = Request(text="Hello, how are you?", modality=Modality.TEXT)

    @patch("src.handlers.llm_local.requests.post")
    def test_successful_text_response(self, mock_post):
        """Test successful plain text response."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "I am fine, thank you!"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = self.handler.process(self.request)

        self.assertTrue(response.success)
        self.assertEqual(response.output, "I am fine, thank you!")
        self.assertEqual(response.text, "I am fine, thank you!")
        self.assertIsNone(response.error)

    @patch("src.handlers.llm_local.requests.post")
    def test_successful_json_response(self, mock_post):
        """Test successful JSON response when requested."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"answer": "42", "confidence": 0.95}'
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        request = Request(
            text="What is the answer? Respond in JSON",
            modality=Modality.TEXT,
        )
        response = self.handler.process(request)

        self.assertTrue(response.success)
        self.assertIsInstance(response.output, dict)
        self.assertEqual(response.output["answer"], "42")
        self.assertEqual(response.output["confidence"], 0.95)

    @patch("src.handlers.llm_local.requests.post")
    def test_json_format_via_metadata(self, mock_post):
        """Test JSON format requested via metadata."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"result": "success"}'
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        request = Request(
            text="Process this",
            modality=Modality.TEXT,
            metadata={"output_format": "json"},
        )
        response = self.handler.process(request)

        self.assertTrue(response.success)
        self.assertIsInstance(response.output, dict)
        self.assertEqual(response.output["result"], "success")

    @patch("src.handlers.llm_local.requests.post")
    def test_ollama_api_called_correctly(self, mock_post):
        """Test that Ollama API is called with correct parameters."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "test"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        handler = LLMLocalHandler({
            "base_url": "http://test:1234",
            "model": "testmodel",
            "temperature": 0.5,
            "max_tokens": 512,
        })
        handler.process(self.request)

        mock_post.assert_called_once()
        call_args = mock_post.call_args

        self.assertIn("http://test:1234/api/generate", call_args[0])

        json_arg = call_args[1]["json"]
        self.assertEqual(json_arg["model"], "testmodel")
        self.assertEqual(json_arg["system"], SYSTEM_PROMPT)
        self.assertFalse(json_arg["stream"])
        self.assertEqual(json_arg["options"]["temperature"], 0.5)
        self.assertEqual(json_arg["options"]["num_predict"], 512)


# =============================================================================
# Test: Ollama Unavailable (Connection Error)
# =============================================================================

class TestLLMLocalConnectionError(unittest.TestCase):
    """Tests for Ollama connection errors."""

    def setUp(self):
        """Set up handler with default config."""
        self.handler = LLMLocalHandler({})
        self.request = Request(text="Test", modality=Modality.TEXT)

    @patch("src.handlers.llm_local.requests.post")
    def test_connection_error_returns_error_response(self, mock_post):
        """Test that connection error returns proper error response."""
        import requests
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        response = self.handler.process(self.request)

        self.assertFalse(response.success)
        self.assertEqual(response.error_code, LLMLocalErrorCode.CONNECTION_ERROR.value)
        self.assertIn("Cannot connect to Ollama", response.error)
        self.assertIn("Is it running?", response.error)

    @patch("src.handlers.llm_local.requests.post")
    def test_connection_error_includes_url(self, mock_post):
        """Test that connection error message includes the URL."""
        import requests
        mock_post.side_effect = requests.ConnectionError()

        handler = LLMLocalHandler({"base_url": "http://custom:9999"})
        response = handler.process(self.request)

        self.assertFalse(response.success)
        self.assertIn("http://custom:9999", response.error)

    @patch("src.handlers.llm_local.requests.post")
    def test_connection_error_no_exception_raised(self, mock_post):
        """Test that connection error does not raise exception."""
        import requests
        mock_post.side_effect = requests.ConnectionError()

        # Should not raise
        response = self.handler.process(self.request)

        self.assertFalse(response.success)


# =============================================================================
# Test: Timeout Handling
# =============================================================================

class TestLLMLocalTimeout(unittest.TestCase):
    """Tests for timeout handling."""

    def setUp(self):
        """Set up handler."""
        self.handler = LLMLocalHandler({"timeout_seconds": 30})
        self.request = Request(text="Test", modality=Modality.TEXT)

    @patch("src.handlers.llm_local.requests.post")
    def test_timeout_returns_error_response(self, mock_post):
        """Test that timeout returns proper error response."""
        import requests
        mock_post.side_effect = requests.Timeout()

        response = self.handler.process(self.request)

        self.assertFalse(response.success)
        self.assertEqual(response.error_code, LLMLocalErrorCode.TIMEOUT.value)
        self.assertIn("timed out", response.error)

    @patch("src.handlers.llm_local.requests.post")
    def test_timeout_includes_duration(self, mock_post):
        """Test that timeout error includes configured duration."""
        import requests
        mock_post.side_effect = requests.Timeout()

        response = self.handler.process(self.request)

        self.assertIn("30s", response.error)

    @patch("src.handlers.llm_local.requests.post")
    def test_timeout_no_exception_raised(self, mock_post):
        """Test that timeout does not raise exception."""
        import requests
        mock_post.side_effect = requests.Timeout()

        # Should not raise
        response = self.handler.process(self.request)

        self.assertFalse(response.success)

    @patch("src.handlers.llm_local.requests.post")
    def test_timeout_passed_to_requests(self, mock_post):
        """Test that configured timeout is passed to requests."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "test"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        handler = LLMLocalHandler({"timeout_seconds": 45})
        handler.process(self.request)

        call_kwargs = mock_post.call_args[1]
        self.assertEqual(call_kwargs["timeout"], 45)


# =============================================================================
# Test: Malformed Response Handling
# =============================================================================

class TestLLMLocalMalformedResponse(unittest.TestCase):
    """Tests for malformed Ollama responses."""

    def setUp(self):
        """Set up handler."""
        self.handler = LLMLocalHandler({})
        self.request = Request(text="Test", modality=Modality.TEXT)

    @patch("src.handlers.llm_local.requests.post")
    def test_invalid_json_response(self, mock_post):
        """Test handling of invalid JSON from Ollama."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("", "", 0)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = self.handler.process(self.request)

        self.assertFalse(response.success)
        self.assertEqual(response.error_code, LLMLocalErrorCode.MALFORMED_RESPONSE.value)
        self.assertIn("invalid JSON", response.error)

    @patch("src.handlers.llm_local.requests.post")
    def test_empty_response_text(self, mock_post):
        """Test handling of empty response text."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": ""}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = self.handler.process(self.request)

        self.assertFalse(response.success)
        self.assertEqual(response.error_code, LLMLocalErrorCode.MALFORMED_RESPONSE.value)
        self.assertIn("empty response", response.error)

    @patch("src.handlers.llm_local.requests.post")
    def test_missing_response_field(self, mock_post):
        """Test handling of missing response field in JSON."""
        mock_response = Mock()
        mock_response.json.return_value = {"other_field": "value"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = self.handler.process(self.request)

        self.assertFalse(response.success)
        self.assertEqual(response.error_code, LLMLocalErrorCode.MALFORMED_RESPONSE.value)

    @patch("src.handlers.llm_local.requests.post")
    def test_http_error_handling(self, mock_post):
        """Test handling of HTTP errors from Ollama."""
        import requests
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=mock_response)
        mock_post.return_value = mock_response

        response = self.handler.process(self.request)

        self.assertFalse(response.success)
        self.assertEqual(response.error_code, LLMLocalErrorCode.OLLAMA_ERROR.value)
        self.assertIn("500", response.error)

    @patch("src.handlers.llm_local.requests.post")
    def test_generic_request_exception(self, mock_post):
        """Test handling of generic request exceptions."""
        import requests
        mock_post.side_effect = requests.RequestException("Generic error")

        response = self.handler.process(self.request)

        self.assertFalse(response.success)
        self.assertEqual(response.error_code, LLMLocalErrorCode.OLLAMA_ERROR.value)

    @patch("src.handlers.llm_local.requests.post")
    def test_unexpected_exception_caught(self, mock_post):
        """Test that unexpected exceptions are caught and don't escape."""
        mock_post.side_effect = RuntimeError("Unexpected!")

        # Should NOT raise
        response = self.handler.process(self.request)

        self.assertFalse(response.success)
        self.assertEqual(response.error_code, LLMLocalErrorCode.OLLAMA_ERROR.value)
        self.assertIn("Unexpected", response.error)


# =============================================================================
# Test: Missing Text Input
# =============================================================================

class TestLLMLocalMissingText(unittest.TestCase):
    """Tests for missing text input validation."""

    def setUp(self):
        """Set up handler."""
        self.handler = LLMLocalHandler({})

    def test_none_text_returns_error(self):
        """Test that None text returns error."""
        request = Request(text=None, modality=Modality.TEXT)

        response = self.handler.process(request)

        self.assertFalse(response.success)
        self.assertEqual(response.error_code, LLMLocalErrorCode.MISSING_TEXT.value)
        self.assertIn("No text provided", response.error)

    def test_empty_text_returns_error(self):
        """Test that empty string returns error."""
        request = Request(text="", modality=Modality.TEXT)

        response = self.handler.process(request)

        self.assertFalse(response.success)
        self.assertEqual(response.error_code, LLMLocalErrorCode.MISSING_TEXT.value)

    def test_whitespace_only_returns_error(self):
        """Test that whitespace-only text returns error."""
        request = Request(text="   \n\t  ", modality=Modality.TEXT)

        response = self.handler.process(request)

        self.assertFalse(response.success)
        self.assertEqual(response.error_code, LLMLocalErrorCode.MISSING_TEXT.value)


# =============================================================================
# Test: JSON Extraction
# =============================================================================

class TestLLMLocalJSONExtraction(unittest.TestCase):
    """Tests for JSON extraction from LLM responses."""

    def setUp(self):
        """Set up handler."""
        self.handler = LLMLocalHandler({})

    @patch("src.handlers.llm_local.requests.post")
    def test_extract_json_with_markdown_code_block(self, mock_post):
        """Test extraction of JSON from markdown code blocks."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '```json\n{"key": "value"}\n```'
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        request = Request(
            text="Return JSON",
            modality=Modality.TEXT,
            metadata={"output_format": "json"},
        )
        response = self.handler.process(request)

        self.assertTrue(response.success)
        self.assertIsInstance(response.output, dict)
        self.assertEqual(response.output["key"], "value")

    @patch("src.handlers.llm_local.requests.post")
    def test_extract_json_with_surrounding_text(self, mock_post):
        """Test extraction of JSON with surrounding explanation text."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": 'Here is the result:\n{"data": 123}\nHope that helps!'
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        request = Request(
            text="Return JSON",
            modality=Modality.TEXT,
            metadata={"output_format": "json"},
        )
        response = self.handler.process(request)

        self.assertTrue(response.success)
        self.assertIsInstance(response.output, dict)
        self.assertEqual(response.output["data"], 123)

    @patch("src.handlers.llm_local.requests.post")
    def test_extract_json_array(self, mock_post):
        """Test extraction of JSON array."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '[1, 2, 3]'
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        request = Request(
            text="Return JSON",
            modality=Modality.TEXT,
            metadata={"output_format": "json"},
        )
        response = self.handler.process(request)

        self.assertTrue(response.success)
        self.assertIsInstance(response.output, list)
        self.assertEqual(response.output, [1, 2, 3])

    @patch("src.handlers.llm_local.requests.post")
    def test_json_parse_failure_returns_text_with_warning(self, mock_post):
        """Test that failed JSON parse returns text with warning."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "This is not JSON at all"
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        request = Request(
            text="Return JSON",
            modality=Modality.TEXT,
            metadata={"output_format": "json"},
        )
        response = self.handler.process(request)

        self.assertTrue(response.success)
        self.assertEqual(response.output, "This is not JSON at all")
        self.assertGreater(len(response.warnings), 0)
        self.assertIn("plain text", response.warnings[0])


# =============================================================================
# Test: Availability Check
# =============================================================================

class TestLLMLocalAvailability(unittest.TestCase):
    """Tests for Ollama availability checking."""

    def setUp(self):
        """Set up handler."""
        self.handler = LLMLocalHandler({})

    @patch("src.handlers.llm_local.requests.get")
    def test_is_available_true_when_ollama_running(self, mock_get):
        """Test is_available returns True when Ollama responds."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        self.assertTrue(self.handler.is_available())

    @patch("src.handlers.llm_local.requests.get")
    def test_is_available_false_when_connection_error(self, mock_get):
        """Test is_available returns False on connection error."""
        import requests
        mock_get.side_effect = requests.ConnectionError()

        self.assertFalse(self.handler.is_available())

    @patch("src.handlers.llm_local.requests.get")
    def test_is_available_false_when_non_200(self, mock_get):
        """Test is_available returns False on non-200 status."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        self.assertFalse(self.handler.is_available())

    @patch("src.handlers.llm_local.requests.get")
    def test_is_available_no_exception_raised(self, mock_get):
        """Test that is_available never raises exceptions."""
        mock_get.side_effect = Exception("Unexpected!")

        # Should NOT raise
        result = self.handler.is_available()

        self.assertFalse(result)

    @patch("src.handlers.llm_local.requests.get")
    def test_check_model_available_true(self, mock_get):
        """Test check_model_available returns True when model exists."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "mistral:latest"},
                {"name": "llama2:latest"},
            ]
        }
        mock_get.return_value = mock_response

        self.assertTrue(self.handler.check_model_available())

    @patch("src.handlers.llm_local.requests.get")
    def test_check_model_available_false_when_not_present(self, mock_get):
        """Test check_model_available returns False when model not in list."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama2:latest"}]
        }
        mock_get.return_value = mock_response

        handler = LLMLocalHandler({"model": "mistral"})
        self.assertFalse(handler.check_model_available())

    @patch("src.handlers.llm_local.requests.get")
    def test_check_model_no_exception(self, mock_get):
        """Test that check_model_available never raises exceptions."""
        mock_get.side_effect = Exception("Error!")

        # Should NOT raise
        result = self.handler.check_model_available()

        self.assertFalse(result)


# =============================================================================
# Test: Handler Properties
# =============================================================================

class TestLLMLocalProperties(unittest.TestCase):
    """Tests for handler properties and metadata."""

    def test_handler_name(self):
        """Test that handler name is correct."""
        handler = LLMLocalHandler({})

        self.assertEqual(handler.name, "LLMLocalHandler")

    def test_system_prompt_content(self):
        """Test that system prompt contains required restrictions."""
        # Check for key restrictions
        self.assertIn("CANNOT call tools", SYSTEM_PROMPT)
        self.assertIn("CANNOT browse the web", SYSTEM_PROMPT)
        self.assertIn("CANNOT read, write, or access any files", SYSTEM_PROMPT)
        self.assertIn("CANNOT execute code", SYSTEM_PROMPT)
        self.assertIn("MUST NOT claim capabilities", SYSTEM_PROMPT)

    def test_response_includes_handler_name(self):
        """Test that successful response includes handler name in meta."""
        handler = LLMLocalHandler({})

        with patch("src.handlers.llm_local.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "test"}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            request = Request(text="Test", modality=Modality.TEXT)
            response = handler.process(request)

            self.assertEqual(response.meta.handler, "LLMLocalHandler")


# =============================================================================
# Test: OutputFormat Enum
# =============================================================================

class TestOutputFormat(unittest.TestCase):
    """Tests for OutputFormat enum."""

    def test_text_format_value(self):
        """Test TEXT format value."""
        self.assertEqual(OutputFormat.TEXT.value, "text")

    def test_json_format_value(self):
        """Test JSON format value."""
        self.assertEqual(OutputFormat.JSON.value, "json")


# =============================================================================
# Test: Error Codes
# =============================================================================

class TestLLMLocalErrorCodes(unittest.TestCase):
    """Tests for error code enum values."""

    def test_all_error_codes_have_llm_prefix(self):
        """Test that all error codes have LLM_ prefix."""
        for code in LLMLocalErrorCode:
            self.assertTrue(code.value.startswith("LLM_"))

    def test_error_codes_are_unique(self):
        """Test that all error codes are unique."""
        values = [code.value for code in LLMLocalErrorCode]
        self.assertEqual(len(values), len(set(values)))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main()
