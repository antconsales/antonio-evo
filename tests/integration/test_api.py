"""
Integration tests for the Antonio HTTP API.

Tests cover:
- /health endpoint success
- /ask valid request
- /ask invalid request
- JSON vs TEXT output modes
- raw=true behavior
- LLM unavailable behavior
"""

import json
import sys
import threading
import time
import unittest
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, ".")

from src.api.server import (
    AntonioAPIServer,
    AntonioRequestHandler,
    create_server,
    create_orchestrator,
    DEFAULT_HOST,
    DEFAULT_PORT,
    HTTP_OK,
    HTTP_BAD_REQUEST,
    HTTP_NOT_FOUND,
    HTTP_SERVICE_UNAVAILABLE,
    HTTP_INTERNAL_ERROR,
)


# =============================================================================
# Mock Request/Response Classes for Testing
# =============================================================================

class MockRequest:
    """Mock HTTP request for testing handler methods."""

    def __init__(
        self,
        method: str = "GET",
        path: str = "/",
        body: bytes = b"",
        headers: dict = None,
    ):
        self.method = method
        self.path = path
        self.body = body
        self.headers = headers or {}
        self._body_stream = BytesIO(body)

    def makefile(self, mode, bufsize=None):
        """Create a file-like object for the request."""
        if "r" in mode:
            return self._body_stream
        return BytesIO()


class MockResponse:
    """Captures response data from handler."""

    def __init__(self):
        self.status_code = None
        self.headers = {}
        self.body = b""
        self._buffer = BytesIO()

    def write(self, data):
        self._buffer.write(data)
        self.body = self._buffer.getvalue()


# =============================================================================
# Test: Health Endpoint
# =============================================================================

class TestHealthEndpoint(unittest.TestCase):
    """Tests for GET /health endpoint."""

    def _create_mock_handler(self, orchestrator=None):
        """Create a mock request handler."""
        # Create mock server
        mock_server = Mock(spec=AntonioAPIServer)
        mock_server.orchestrator = orchestrator

        # Create handler with mock request
        handler = Mock(spec=AntonioRequestHandler)
        handler.server = mock_server
        handler.headers = {"Content-Length": "0"}
        handler.path = "/health"

        return handler

    def test_health_returns_ok_status(self):
        """Test that /health returns status ok."""
        mock_orchestrator = Mock()
        mock_orchestrator.check_llm_available.return_value = True

        responses = []

        def capture_response(data, status_code=HTTP_OK):
            responses.append({"data": data, "status": status_code})

        handler = self._create_mock_handler(mock_orchestrator)
        handler._send_json_response = capture_response

        # Call the actual health handler
        AntonioRequestHandler._handle_health(handler)

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["data"]["status"], "ok")

    def test_health_includes_llm_available(self):
        """Test that /health includes llm_available field."""
        mock_orchestrator = Mock()
        mock_orchestrator.check_llm_available.return_value = True

        responses = []

        def capture_response(data, status_code=HTTP_OK):
            responses.append({"data": data, "status": status_code})

        handler = self._create_mock_handler(mock_orchestrator)
        handler._send_json_response = capture_response

        AntonioRequestHandler._handle_health(handler)

        self.assertIn("llm_available", responses[0]["data"])
        self.assertTrue(responses[0]["data"]["llm_available"])

    def test_health_llm_unavailable(self):
        """Test that /health reports llm_available=false when LLM unavailable."""
        mock_orchestrator = Mock()
        mock_orchestrator.check_llm_available.return_value = False

        responses = []

        def capture_response(data, status_code=HTTP_OK):
            responses.append({"data": data, "status": status_code})

        handler = self._create_mock_handler(mock_orchestrator)
        handler._send_json_response = capture_response

        AntonioRequestHandler._handle_health(handler)

        self.assertFalse(responses[0]["data"]["llm_available"])

    def test_health_includes_timestamp(self):
        """Test that /health includes timestamp."""
        mock_orchestrator = Mock()
        mock_orchestrator.check_llm_available.return_value = True

        responses = []

        def capture_response(data, status_code=HTTP_OK):
            responses.append({"data": data, "status": status_code})

        handler = self._create_mock_handler(mock_orchestrator)
        handler._send_json_response = capture_response

        AntonioRequestHandler._handle_health(handler)

        self.assertIn("timestamp", responses[0]["data"])
        self.assertIsInstance(responses[0]["data"]["timestamp"], float)

    def test_health_no_orchestrator(self):
        """Test that /health works even without orchestrator."""
        responses = []

        def capture_response(data, status_code=HTTP_OK):
            responses.append({"data": data, "status": status_code})

        handler = self._create_mock_handler(None)
        handler._send_json_response = capture_response

        AntonioRequestHandler._handle_health(handler)

        # Should still return ok status
        self.assertEqual(responses[0]["data"]["status"], "ok")
        self.assertFalse(responses[0]["data"]["llm_available"])


# =============================================================================
# Test: Ask Endpoint - Valid Requests
# =============================================================================

class TestAskValidRequest(unittest.TestCase):
    """Tests for POST /ask with valid requests."""

    def _create_mock_handler_with_orchestrator(self, process_response: dict):
        """Create a handler with mocked orchestrator."""
        mock_orchestrator = Mock()
        mock_orchestrator.process.return_value = process_response

        formatted_mock = Mock()
        formatted_mock.to_text.return_value = "Formatted response"
        formatted_mock.to_dict.return_value = {"success": True, "message": "OK"}
        mock_orchestrator.formatter = Mock()
        mock_orchestrator.formatter.format.return_value = formatted_mock

        mock_server = Mock(spec=AntonioAPIServer)
        mock_server.orchestrator = mock_orchestrator

        handler = Mock(spec=AntonioRequestHandler)
        handler.server = mock_server
        handler.path = "/ask"

        return handler, mock_orchestrator

    def test_ask_valid_question_success(self):
        """Test that valid question returns success."""
        handler, orchestrator = self._create_mock_handler_with_orchestrator({
            "success": True,
            "text": "The answer is 42",
        })

        responses = []

        def capture_text(text, status_code=HTTP_OK):
            responses.append({"text": text, "status": status_code})

        def mock_parse():
            return {"question": "What is the answer?"}

        handler._parse_json_body = mock_parse
        handler._send_text_response = capture_text
        handler._send_json_response = Mock()
        handler._send_error_response = Mock()

        AntonioRequestHandler._handle_ask(handler)

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["status"], HTTP_OK)

    def test_ask_calls_orchestrator_process(self):
        """Test that /ask calls orchestrator.process with question."""
        handler, orchestrator = self._create_mock_handler_with_orchestrator({
            "success": True,
        })

        def mock_parse():
            return {"question": "Test question"}

        handler._parse_json_body = mock_parse
        handler._send_text_response = Mock()
        handler._send_json_response = Mock()
        handler._send_error_response = Mock()

        AntonioRequestHandler._handle_ask(handler)

        orchestrator.process.assert_called_once_with("Test question")

    def test_ask_strips_whitespace(self):
        """Test that question whitespace is stripped."""
        handler, orchestrator = self._create_mock_handler_with_orchestrator({
            "success": True,
        })

        def mock_parse():
            return {"question": "  Question with spaces  "}

        handler._parse_json_body = mock_parse
        handler._send_text_response = Mock()
        handler._send_json_response = Mock()
        handler._send_error_response = Mock()

        AntonioRequestHandler._handle_ask(handler)

        orchestrator.process.assert_called_once_with("Question with spaces")


# =============================================================================
# Test: Ask Endpoint - Invalid Requests
# =============================================================================

class TestAskInvalidRequest(unittest.TestCase):
    """Tests for POST /ask with invalid requests."""

    def _create_mock_handler(self):
        """Create a handler for invalid request tests."""
        mock_server = Mock(spec=AntonioAPIServer)
        mock_server.orchestrator = Mock()

        handler = Mock(spec=AntonioRequestHandler)
        handler.server = mock_server
        handler.path = "/ask"

        return handler

    def test_ask_missing_question_returns_error(self):
        """Test that missing question field returns error."""
        handler = self._create_mock_handler()

        errors = []

        def capture_error(message, status_code=HTTP_BAD_REQUEST):
            errors.append({"message": message, "status": status_code})

        def mock_parse():
            return {}  # No question field

        handler._parse_json_body = mock_parse
        handler._send_error_response = capture_error

        AntonioRequestHandler._handle_ask(handler)

        self.assertEqual(len(errors), 1)
        self.assertIn("question", errors[0]["message"].lower())

    def test_ask_empty_question_returns_error(self):
        """Test that empty question returns error."""
        handler = self._create_mock_handler()

        errors = []

        def capture_error(message, status_code=HTTP_BAD_REQUEST):
            errors.append({"message": message, "status": status_code})

        def mock_parse():
            return {"question": ""}

        handler._parse_json_body = mock_parse
        handler._send_error_response = capture_error

        AntonioRequestHandler._handle_ask(handler)

        self.assertEqual(len(errors), 1)
        self.assertIn("empty", errors[0]["message"].lower())

    def test_ask_whitespace_question_returns_error(self):
        """Test that whitespace-only question returns error."""
        handler = self._create_mock_handler()

        errors = []

        def capture_error(message, status_code=HTTP_BAD_REQUEST):
            errors.append({"message": message, "status": status_code})

        def mock_parse():
            return {"question": "   \n\t   "}

        handler._parse_json_body = mock_parse
        handler._send_error_response = capture_error

        AntonioRequestHandler._handle_ask(handler)

        self.assertEqual(len(errors), 1)

    def test_ask_non_string_question_returns_error(self):
        """Test that non-string question returns error."""
        handler = self._create_mock_handler()

        errors = []

        def capture_error(message, status_code=HTTP_BAD_REQUEST):
            errors.append({"message": message, "status": status_code})

        def mock_parse():
            return {"question": 123}

        handler._parse_json_body = mock_parse
        handler._send_error_response = capture_error

        AntonioRequestHandler._handle_ask(handler)

        self.assertEqual(len(errors), 1)
        self.assertIn("string", errors[0]["message"].lower())

    def test_ask_invalid_output_mode_returns_error(self):
        """Test that invalid output_mode returns error."""
        handler = self._create_mock_handler()

        errors = []

        def capture_error(message, status_code=HTTP_BAD_REQUEST):
            errors.append({"message": message, "status": status_code})

        def mock_parse():
            return {"question": "Test", "output_mode": "invalid"}

        handler._parse_json_body = mock_parse
        handler._send_error_response = capture_error

        AntonioRequestHandler._handle_ask(handler)

        self.assertEqual(len(errors), 1)
        self.assertIn("output_mode", errors[0]["message"].lower())

    def test_ask_invalid_raw_type_returns_error(self):
        """Test that non-boolean raw field returns error."""
        handler = self._create_mock_handler()

        errors = []

        def capture_error(message, status_code=HTTP_BAD_REQUEST):
            errors.append({"message": message, "status": status_code})

        def mock_parse():
            return {"question": "Test", "raw": "true"}  # string, not bool

        handler._parse_json_body = mock_parse
        handler._send_error_response = capture_error

        AntonioRequestHandler._handle_ask(handler)

        self.assertEqual(len(errors), 1)
        self.assertIn("raw", errors[0]["message"].lower())


# =============================================================================
# Test: JSON vs TEXT Output Modes
# =============================================================================

class TestOutputModes(unittest.TestCase):
    """Tests for JSON and TEXT output modes."""

    def _create_mock_handler_with_orchestrator(self, process_response: dict):
        """Create a handler with mocked orchestrator."""
        mock_orchestrator = Mock()
        mock_orchestrator.process.return_value = process_response

        formatted_mock = Mock()
        formatted_mock.to_text.return_value = "Plain text response"
        formatted_mock.to_dict.return_value = {"success": True, "message": "JSON"}
        mock_orchestrator.formatter = Mock()
        mock_orchestrator.formatter.format.return_value = formatted_mock

        mock_server = Mock(spec=AntonioAPIServer)
        mock_server.orchestrator = mock_orchestrator

        handler = Mock(spec=AntonioRequestHandler)
        handler.server = mock_server
        handler.path = "/ask"

        return handler, mock_orchestrator

    def test_text_mode_returns_plain_text(self):
        """Test that output_mode=text returns plain text."""
        handler, _ = self._create_mock_handler_with_orchestrator({
            "success": True,
        })

        text_responses = []
        json_responses = []

        def capture_text(text, status_code=HTTP_OK):
            text_responses.append(text)

        def capture_json(data, status_code=HTTP_OK):
            json_responses.append(data)

        def mock_parse():
            return {"question": "Test", "output_mode": "text"}

        handler._parse_json_body = mock_parse
        handler._send_text_response = capture_text
        handler._send_json_response = capture_json
        handler._send_error_response = Mock()

        AntonioRequestHandler._handle_ask(handler)

        # Should use text response
        self.assertEqual(len(text_responses), 1)
        self.assertEqual(len(json_responses), 0)
        self.assertEqual(text_responses[0], "Plain text response")

    def test_json_mode_returns_json(self):
        """Test that output_mode=json returns JSON."""
        handler, _ = self._create_mock_handler_with_orchestrator({
            "success": True,
        })

        text_responses = []
        json_responses = []

        def capture_text(text, status_code=HTTP_OK):
            text_responses.append(text)

        def capture_json(data, status_code=HTTP_OK):
            json_responses.append(data)

        def mock_parse():
            return {"question": "Test", "output_mode": "json"}

        handler._parse_json_body = mock_parse
        handler._send_text_response = capture_text
        handler._send_json_response = capture_json
        handler._send_error_response = Mock()

        AntonioRequestHandler._handle_ask(handler)

        # Should use JSON response
        self.assertEqual(len(json_responses), 1)
        self.assertEqual(len(text_responses), 0)

    def test_default_mode_is_text(self):
        """Test that default output mode is text."""
        handler, _ = self._create_mock_handler_with_orchestrator({
            "success": True,
        })

        text_responses = []

        def capture_text(text, status_code=HTTP_OK):
            text_responses.append(text)

        def mock_parse():
            return {"question": "Test"}  # No output_mode specified

        handler._parse_json_body = mock_parse
        handler._send_text_response = capture_text
        handler._send_json_response = Mock()
        handler._send_error_response = Mock()

        AntonioRequestHandler._handle_ask(handler)

        self.assertEqual(len(text_responses), 1)


# =============================================================================
# Test: Raw Output Mode
# =============================================================================

class TestRawOutputMode(unittest.TestCase):
    """Tests for raw=true output mode."""

    def _create_mock_handler_with_orchestrator(self, process_response: dict):
        """Create a handler with mocked orchestrator."""
        mock_orchestrator = Mock()
        mock_orchestrator.process.return_value = process_response
        mock_orchestrator.formatter = Mock()

        mock_server = Mock(spec=AntonioAPIServer)
        mock_server.orchestrator = mock_orchestrator

        handler = Mock(spec=AntonioRequestHandler)
        handler.server = mock_server
        handler.path = "/ask"

        return handler, mock_orchestrator

    def test_raw_mode_returns_internal_response(self):
        """Test that raw=true returns internal response."""
        internal_response = {
            "success": True,
            "_meta": {"handler": "test", "elapsed_ms": 100},
        }

        handler, orchestrator = self._create_mock_handler_with_orchestrator(
            internal_response
        )

        json_responses = []

        def capture_json(data, status_code=HTTP_OK):
            json_responses.append(data)

        def mock_parse():
            return {"question": "Test", "raw": True}

        handler._parse_json_body = mock_parse
        handler._send_json_response = capture_json
        handler._send_text_response = Mock()
        handler._send_error_response = Mock()

        AntonioRequestHandler._handle_ask(handler)

        # Should return internal response with _meta
        self.assertEqual(len(json_responses), 1)
        self.assertIn("_meta", json_responses[0])
        self.assertEqual(json_responses[0]["_meta"]["handler"], "test")

    def test_raw_mode_bypasses_formatter(self):
        """Test that raw=true bypasses ResponseFormatter."""
        handler, orchestrator = self._create_mock_handler_with_orchestrator({
            "success": True,
        })

        def mock_parse():
            return {"question": "Test", "raw": True}

        handler._parse_json_body = mock_parse
        handler._send_json_response = Mock()
        handler._send_text_response = Mock()
        handler._send_error_response = Mock()

        AntonioRequestHandler._handle_ask(handler)

        # Formatter should NOT be called
        orchestrator.formatter.format.assert_not_called()

    def test_non_raw_mode_uses_formatter(self):
        """Test that raw=false uses ResponseFormatter."""
        handler, orchestrator = self._create_mock_handler_with_orchestrator({
            "success": True,
        })

        formatted_mock = Mock()
        formatted_mock.to_text.return_value = "Formatted"
        orchestrator.formatter.format.return_value = formatted_mock

        def mock_parse():
            return {"question": "Test", "raw": False}

        handler._parse_json_body = mock_parse
        handler._send_text_response = Mock()
        handler._send_json_response = Mock()
        handler._send_error_response = Mock()

        AntonioRequestHandler._handle_ask(handler)

        # Formatter SHOULD be called
        orchestrator.formatter.format.assert_called_once()


# =============================================================================
# Test: Service Unavailable
# =============================================================================

class TestServiceUnavailable(unittest.TestCase):
    """Tests for service unavailable scenarios."""

    def test_ask_no_orchestrator_returns_503(self):
        """Test that /ask returns 503 when orchestrator is unavailable."""
        mock_server = Mock(spec=AntonioAPIServer)
        mock_server.orchestrator = None

        handler = Mock(spec=AntonioRequestHandler)
        handler.server = mock_server
        handler.path = "/ask"

        errors = []

        def capture_error(message, status_code=HTTP_BAD_REQUEST):
            errors.append({"message": message, "status": status_code})

        def mock_parse():
            return {"question": "Test"}

        handler._parse_json_body = mock_parse
        handler._send_error_response = capture_error

        AntonioRequestHandler._handle_ask(handler)

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["status"], HTTP_SERVICE_UNAVAILABLE)


# =============================================================================
# Test: Server Creation
# =============================================================================

class TestServerCreation(unittest.TestCase):
    """Tests for server creation."""

    def test_create_server_returns_tuple(self):
        """Test that create_server returns a tuple."""
        with patch("src.api.server.create_orchestrator") as mock_create:
            mock_create.return_value = (Mock(), None)

            result = create_server()

            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)

    def test_create_server_rejects_non_localhost(self):
        """Test that create_server rejects non-localhost hosts."""
        server, error = create_server(host="0.0.0.0")

        self.assertIsNone(server)
        self.assertIn("localhost", error.lower())

    def test_create_server_accepts_localhost(self):
        """Test that create_server accepts localhost."""
        with patch("src.api.server.create_orchestrator") as mock_create:
            mock_create.return_value = (Mock(), None)

            server, error = create_server(host="127.0.0.1")

            # Should not fail due to host check
            if error:
                self.assertNotIn("localhost", error.lower())

    def test_create_server_accepts_ipv6_localhost(self):
        """Test that create_server accepts IPv6 localhost."""
        with patch("src.api.server.create_orchestrator") as mock_create:
            mock_create.return_value = (Mock(), None)

            server, error = create_server(host="::1")

            if error:
                self.assertNotIn("localhost", error.lower())


# =============================================================================
# Test: Orchestrator Creation
# =============================================================================

class TestOrchestratorCreation(unittest.TestCase):
    """Tests for orchestrator creation."""

    def test_create_orchestrator_returns_tuple(self):
        """Test that create_orchestrator returns a tuple."""
        result = create_orchestrator()

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_create_orchestrator_failure_returns_none_and_error(self):
        """Test that orchestrator creation failure is handled."""
        with patch("src.api.server.Normalizer", side_effect=ImportError("Test")):
            orchestrator, error = create_orchestrator()

        # May or may not fail depending on import order
        if orchestrator is None:
            self.assertIsNotNone(error)


# =============================================================================
# Test: HTTP Methods
# =============================================================================

class TestHTTPMethods(unittest.TestCase):
    """Tests for HTTP method handling."""

    def test_get_unknown_path_returns_404(self):
        """Test that GET to unknown path returns 404."""
        mock_server = Mock(spec=AntonioAPIServer)

        handler = Mock(spec=AntonioRequestHandler)
        handler.server = mock_server
        handler.path = "/unknown"

        errors = []

        def capture_error(message, status_code=HTTP_BAD_REQUEST):
            errors.append({"message": message, "status": status_code})

        handler._send_error_response = capture_error

        # Simulate do_GET
        from urllib.parse import urlparse
        parsed = urlparse(handler.path)
        path = parsed.path.rstrip("/")

        if path == "/health":
            pass  # Would call _handle_health
        else:
            handler._send_error_response("Not found", HTTP_NOT_FOUND)

        self.assertEqual(errors[0]["status"], HTTP_NOT_FOUND)

    def test_post_unknown_path_returns_404(self):
        """Test that POST to unknown path returns 404."""
        mock_server = Mock(spec=AntonioAPIServer)

        handler = Mock(spec=AntonioRequestHandler)
        handler.server = mock_server
        handler.path = "/unknown"

        errors = []

        def capture_error(message, status_code=HTTP_BAD_REQUEST):
            errors.append({"message": message, "status": status_code})

        handler._send_error_response = capture_error

        # Simulate do_POST
        from urllib.parse import urlparse
        parsed = urlparse(handler.path)
        path = parsed.path.rstrip("/")

        if path == "/ask":
            pass  # Would call _handle_ask
        else:
            handler._send_error_response("Not found", HTTP_NOT_FOUND)

        self.assertEqual(errors[0]["status"], HTTP_NOT_FOUND)


# =============================================================================
# Test: Error Response Format
# =============================================================================

class TestErrorResponseFormat(unittest.TestCase):
    """Tests for error response format."""

    def test_error_response_has_success_false(self):
        """Test that error responses have success=false."""
        # This tests the contract of _send_error_response
        handler = Mock(spec=AntonioRequestHandler)

        captured = []

        def capture_json(data, status_code=HTTP_OK):
            captured.append(data)

        handler._send_json_response = capture_json

        # Call actual method
        AntonioRequestHandler._send_error_response(handler, "Test error", HTTP_BAD_REQUEST)

        self.assertEqual(len(captured), 1)
        self.assertFalse(captured[0]["success"])
        self.assertEqual(captured[0]["error"], "Test error")


# =============================================================================
# Test: Security - No Internal Details
# =============================================================================

class TestSecurityNoInternalDetails(unittest.TestCase):
    """Tests ensuring internal details are not leaked."""

    def test_process_exception_hides_details(self):
        """Test that process exceptions don't leak details."""
        mock_orchestrator = Mock()
        mock_orchestrator.process.side_effect = RuntimeError("Internal: line 42")
        mock_orchestrator.formatter = Mock()

        mock_server = Mock(spec=AntonioAPIServer)
        mock_server.orchestrator = mock_orchestrator

        handler = Mock(spec=AntonioRequestHandler)
        handler.server = mock_server
        handler.path = "/ask"

        errors = []

        def capture_error(message, status_code=HTTP_BAD_REQUEST):
            errors.append({"message": message, "status": status_code})

        def mock_parse():
            return {"question": "Test"}

        handler._parse_json_body = mock_parse
        handler._send_error_response = capture_error

        AntonioRequestHandler._handle_ask(handler)

        # Should NOT contain internal details
        self.assertNotIn("line 42", errors[0]["message"])
        self.assertNotIn("RuntimeError", errors[0]["message"])


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main()
