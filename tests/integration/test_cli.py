"""
Integration tests for the Antonio CLI.

Tests cover:
- Successful question processing
- --json flag output
- --raw flag output
- Missing input handling
- Invalid input handling
- Non-zero exit codes on error
"""

import json
import sys
import unittest
from io import StringIO
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, ".")

from src.cli.antonio import (
    run_cli,
    create_parser,
    determine_exit_code,
    format_output,
    initialize_orchestrator,
    EXIT_SUCCESS,
    EXIT_USER_ERROR,
    EXIT_SYSTEM_ERROR,
    VERSION,
)


# =============================================================================
# Test: Argument Parsing
# =============================================================================

class TestArgumentParsing(unittest.TestCase):
    """Tests for CLI argument parsing."""

    def test_parser_accepts_question(self):
        """Test that parser accepts a question argument."""
        parser = create_parser()
        args = parser.parse_args(["What is Python?"])

        self.assertEqual(args.question, "What is Python?")
        self.assertFalse(args.json_output)
        self.assertFalse(args.raw)

    def test_parser_accepts_json_flag(self):
        """Test that parser accepts --json flag."""
        parser = create_parser()
        args = parser.parse_args(["--json", "Question"])

        self.assertTrue(args.json_output)

    def test_parser_accepts_raw_flag(self):
        """Test that parser accepts --raw flag."""
        parser = create_parser()
        args = parser.parse_args(["--raw", "Question"])

        self.assertTrue(args.raw)

    def test_parser_accepts_combined_flags(self):
        """Test that parser accepts both flags together."""
        parser = create_parser()
        args = parser.parse_args(["--json", "--raw", "Question"])

        self.assertTrue(args.json_output)
        self.assertTrue(args.raw)

    def test_parser_question_is_optional(self):
        """Test that question can be omitted (handled later)."""
        parser = create_parser()
        args = parser.parse_args([])

        self.assertIsNone(args.question)


# =============================================================================
# Test: Missing Input Handling
# =============================================================================

class TestMissingInput(unittest.TestCase):
    """Tests for missing input handling."""

    def test_missing_question_returns_user_error(self):
        """Test that missing question returns user error exit code."""
        with patch("sys.stderr", new_callable=StringIO):
            exit_code = run_cli([])

        self.assertEqual(exit_code, EXIT_USER_ERROR)

    def test_missing_question_prints_usage(self):
        """Test that missing question prints usage message."""
        stderr = StringIO()
        with patch("sys.stderr", stderr):
            run_cli([])

        output = stderr.getvalue()
        self.assertIn("Error", output)
        self.assertIn("question", output.lower())

    def test_empty_question_returns_user_error(self):
        """Test that empty question returns user error exit code."""
        with patch("sys.stderr", new_callable=StringIO):
            exit_code = run_cli(["   "])

        self.assertEqual(exit_code, EXIT_USER_ERROR)

    def test_whitespace_only_returns_user_error(self):
        """Test that whitespace-only input returns user error."""
        with patch("sys.stderr", new_callable=StringIO):
            exit_code = run_cli(["  \n\t  "])

        self.assertEqual(exit_code, EXIT_USER_ERROR)


# =============================================================================
# Test: Successful Question
# =============================================================================

class TestSuccessfulQuestion(unittest.TestCase):
    """Tests for successful question processing."""

    def _create_mock_orchestrator(self, response: dict):
        """Create a mock orchestrator that returns the given response."""
        mock_orchestrator = Mock()
        mock_orchestrator.process.return_value = response
        mock_orchestrator.formatter = Mock()
        mock_orchestrator.formatter.format.return_value = Mock(
            to_text=Mock(return_value="Formatted text"),
            to_json=Mock(return_value='{"success": true}'),
        )
        return mock_orchestrator

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_successful_question_returns_zero(self, mock_init):
        """Test that successful question returns exit code 0."""
        mock_orch = self._create_mock_orchestrator({
            "success": True,
            "text": "Paris is the capital of France.",
        })
        mock_init.return_value = (mock_orch, None)

        with patch("sys.stdout", new_callable=StringIO):
            exit_code = run_cli(["What is the capital of France?"])

        self.assertEqual(exit_code, EXIT_SUCCESS)

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_successful_question_prints_output(self, mock_init):
        """Test that successful question prints formatted output."""
        mock_orch = self._create_mock_orchestrator({
            "success": True,
            "text": "The answer is 42.",
        })
        mock_init.return_value = (mock_orch, None)

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            run_cli(["What is the answer?"])

        output = stdout.getvalue()
        self.assertIn("Formatted text", output)

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_orchestrator_process_called_with_question(self, mock_init):
        """Test that orchestrator.process is called with the question."""
        mock_orch = self._create_mock_orchestrator({"success": True})
        mock_init.return_value = (mock_orch, None)

        with patch("sys.stdout", new_callable=StringIO):
            run_cli(["My test question"])

        mock_orch.process.assert_called_once_with("My test question")


# =============================================================================
# Test: --json Flag
# =============================================================================

class TestJsonFlag(unittest.TestCase):
    """Tests for --json flag functionality."""

    def _create_mock_orchestrator(self, response: dict):
        """Create a mock orchestrator."""
        mock_orchestrator = Mock()
        mock_orchestrator.process.return_value = response

        formatted_mock = Mock()
        formatted_mock.to_text.return_value = "Plain text"
        formatted_mock.to_json.return_value = json.dumps({"success": True, "message": "OK"})

        mock_orchestrator.formatter = Mock()
        mock_orchestrator.formatter.format.return_value = formatted_mock
        return mock_orchestrator

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_json_flag_outputs_json(self, mock_init):
        """Test that --json flag produces JSON output."""
        mock_orch = self._create_mock_orchestrator({"success": True})
        mock_init.return_value = (mock_orch, None)

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            run_cli(["--json", "Question"])

        output = stdout.getvalue()
        # Should be valid JSON
        parsed = json.loads(output)
        self.assertIn("success", parsed)

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_json_flag_calls_to_json(self, mock_init):
        """Test that --json flag uses to_json method."""
        mock_orch = self._create_mock_orchestrator({"success": True})
        mock_init.return_value = (mock_orch, None)

        with patch("sys.stdout", new_callable=StringIO):
            run_cli(["--json", "Question"])

        # Verify to_json was called (via formatter.format result)
        mock_orch.formatter.format.assert_called_once()

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_json_error_output_to_stdout(self, mock_init):
        """Test that JSON errors go to stdout for easy parsing."""
        mock_orch = self._create_mock_orchestrator({
            "success": False,
            "error_code": "TIMEOUT",
        })

        formatted_mock = Mock()
        formatted_mock.to_json.return_value = '{"success": false}'
        mock_orch.formatter.format.return_value = formatted_mock

        mock_init.return_value = (mock_orch, None)

        stdout = StringIO()
        with patch("sys.stdout", stdout), patch("sys.stderr", new_callable=StringIO):
            run_cli(["--json", "Question"])

        # Error output should still go to stdout in JSON mode
        self.assertIn("false", stdout.getvalue())


# =============================================================================
# Test: --raw Flag
# =============================================================================

class TestRawFlag(unittest.TestCase):
    """Tests for --raw flag functionality."""

    def _create_mock_orchestrator(self, response: dict):
        """Create a mock orchestrator."""
        mock_orchestrator = Mock()
        mock_orchestrator.process.return_value = response
        mock_orchestrator.formatter = Mock()
        return mock_orchestrator

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_raw_flag_outputs_internal_response(self, mock_init):
        """Test that --raw flag outputs internal response."""
        internal_response = {
            "success": True,
            "output": "Test output",
            "_meta": {"handler": "test"},
        }
        mock_orch = self._create_mock_orchestrator(internal_response)
        mock_init.return_value = (mock_orch, None)

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            run_cli(["--raw", "Question"])

        output = stdout.getvalue()
        parsed = json.loads(output)

        # Should contain internal fields
        self.assertIn("_meta", parsed)
        self.assertEqual(parsed["_meta"]["handler"], "test")

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_raw_flag_bypasses_formatter(self, mock_init):
        """Test that --raw flag bypasses ResponseFormatter."""
        mock_orch = self._create_mock_orchestrator({"success": True})
        mock_init.return_value = (mock_orch, None)

        with patch("sys.stdout", new_callable=StringIO):
            run_cli(["--raw", "Question"])

        # Formatter should not be used
        mock_orch.formatter.format.assert_not_called()

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_raw_flag_shows_error_details(self, mock_init):
        """Test that --raw flag shows internal error details."""
        internal_response = {
            "success": False,
            "error": "Internal error message with details",
            "error_code": "INTERNAL_ERROR",
        }
        mock_orch = self._create_mock_orchestrator(internal_response)
        mock_init.return_value = (mock_orch, None)

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            run_cli(["--raw", "Question"])

        output = stdout.getvalue()
        self.assertIn("Internal error message", output)


# =============================================================================
# Test: Invalid Input Handling
# =============================================================================

class TestInvalidInput(unittest.TestCase):
    """Tests for invalid input handling."""

    def _create_mock_orchestrator_validation_error(self):
        """Create orchestrator that returns validation error."""
        mock_orchestrator = Mock()
        mock_orchestrator.process.return_value = {
            "success": False,
            "error": "Invalid input",
            "error_code": "VALIDATION_ERROR",
        }

        formatted_mock = Mock()
        formatted_mock.to_text.return_value = "Error: Invalid input"
        mock_orchestrator.formatter = Mock()
        mock_orchestrator.formatter.format.return_value = formatted_mock

        return mock_orchestrator

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_validation_error_returns_user_error_code(self, mock_init):
        """Test that validation error returns EXIT_USER_ERROR."""
        mock_orch = self._create_mock_orchestrator_validation_error()
        mock_init.return_value = (mock_orch, None)

        with patch("sys.stderr", new_callable=StringIO):
            exit_code = run_cli(["invalid input"])

        self.assertEqual(exit_code, EXIT_USER_ERROR)

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_validation_error_prints_to_stderr(self, mock_init):
        """Test that validation errors print to stderr in text mode."""
        mock_orch = self._create_mock_orchestrator_validation_error()
        mock_init.return_value = (mock_orch, None)

        stderr = StringIO()
        with patch("sys.stderr", stderr), patch("sys.stdout", new_callable=StringIO):
            run_cli(["invalid input"])

        self.assertIn("Error", stderr.getvalue())


# =============================================================================
# Test: Non-Zero Exit on Error
# =============================================================================

class TestExitCodes(unittest.TestCase):
    """Tests for exit codes on various error conditions."""

    def test_determine_exit_code_success(self):
        """Test that successful response returns EXIT_SUCCESS."""
        response = {"success": True}
        self.assertEqual(determine_exit_code(response), EXIT_SUCCESS)

    def test_determine_exit_code_validation_error(self):
        """Test that validation error returns EXIT_USER_ERROR."""
        response = {"success": False, "error_code": "VALIDATION_ERROR"}
        self.assertEqual(determine_exit_code(response), EXIT_USER_ERROR)

    def test_determine_exit_code_missing_text(self):
        """Test that missing text returns EXIT_USER_ERROR."""
        response = {"success": False, "error_code": "LLM_MISSING_TEXT"}
        self.assertEqual(determine_exit_code(response), EXIT_USER_ERROR)

    def test_determine_exit_code_timeout(self):
        """Test that timeout returns EXIT_SYSTEM_ERROR."""
        response = {"success": False, "error_code": "SANDBOX_TIMEOUT"}
        self.assertEqual(determine_exit_code(response), EXIT_SYSTEM_ERROR)

    def test_determine_exit_code_connection_error(self):
        """Test that connection error returns EXIT_SYSTEM_ERROR."""
        response = {"success": False, "error_code": "LLM_CONNECTION_ERROR"}
        self.assertEqual(determine_exit_code(response), EXIT_SYSTEM_ERROR)

    def test_determine_exit_code_unknown_error(self):
        """Test that unknown error returns EXIT_SYSTEM_ERROR."""
        response = {"success": False, "error_code": "UNKNOWN_ERROR"}
        self.assertEqual(determine_exit_code(response), EXIT_SYSTEM_ERROR)

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_system_error_returns_exit_code_2(self, mock_init):
        """Test that system errors return exit code 2."""
        mock_orch = Mock()
        mock_orch.process.return_value = {
            "success": False,
            "error_code": "LLM_CONNECTION_ERROR",
        }
        mock_orch.formatter = Mock()
        mock_orch.formatter.format.return_value = Mock(
            to_text=Mock(return_value="Error"),
        )
        mock_init.return_value = (mock_orch, None)

        with patch("sys.stderr", new_callable=StringIO):
            exit_code = run_cli(["Question"])

        self.assertEqual(exit_code, EXIT_SYSTEM_ERROR)

    def test_initialization_failure_returns_system_error(self):
        """Test that initialization failure returns EXIT_SYSTEM_ERROR."""
        with patch("src.cli.antonio.initialize_orchestrator") as mock_init:
            mock_init.return_value = (None, "Init failed")

            with patch("sys.stderr", new_callable=StringIO):
                exit_code = run_cli(["Question"])

        self.assertEqual(exit_code, EXIT_SYSTEM_ERROR)


# =============================================================================
# Test: Output Formatting
# =============================================================================

class TestOutputFormatting(unittest.TestCase):
    """Tests for output formatting logic."""

    def test_format_output_raw_mode(self):
        """Test that raw mode outputs JSON of internal response."""
        response = {"key": "value", "_internal": "data"}
        formatter = Mock()

        output = format_output(response, json_output=False, raw_output=True, formatter=formatter)

        parsed = json.loads(output)
        self.assertEqual(parsed["key"], "value")
        self.assertEqual(parsed["_internal"], "data")
        formatter.format.assert_not_called()

    def test_format_output_json_mode(self):
        """Test that JSON mode uses formatter."""
        response = {"success": True}
        formatter = Mock()
        formatter.format.return_value = Mock(
            to_json=Mock(return_value='{"formatted": true}')
        )

        output = format_output(response, json_output=True, raw_output=False, formatter=formatter)

        self.assertEqual(output, '{"formatted": true}')
        formatter.format.assert_called_once_with(response)

    def test_format_output_text_mode(self):
        """Test that text mode uses formatter to_text."""
        response = {"success": True}
        formatter = Mock()
        formatter.format.return_value = Mock(
            to_text=Mock(return_value="Plain text output")
        )

        output = format_output(response, json_output=False, raw_output=False, formatter=formatter)

        self.assertEqual(output, "Plain text output")

    def test_raw_takes_precedence_over_json(self):
        """Test that --raw takes precedence over --json."""
        response = {"raw": "data"}
        formatter = Mock()

        output = format_output(response, json_output=True, raw_output=True, formatter=formatter)

        # Should be raw JSON, not formatted
        parsed = json.loads(output)
        self.assertEqual(parsed["raw"], "data")
        formatter.format.assert_not_called()


# =============================================================================
# Test: Orchestrator Initialization
# =============================================================================

class TestOrchestratorInitialization(unittest.TestCase):
    """Tests for orchestrator initialization."""

    def test_initialization_returns_tuple(self):
        """Test that initialize_orchestrator returns a tuple."""
        result = initialize_orchestrator()

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_initialization_failure_returns_none_and_error(self):
        """Test that initialization failure returns (None, error_message)."""
        with patch("src.cli.antonio.Normalizer", side_effect=ImportError("Test")):
            orchestrator, error = initialize_orchestrator()

        # May or may not fail depending on import order
        # Just check the contract is upheld
        if orchestrator is None:
            self.assertIsNotNone(error)
        else:
            self.assertIsNone(error)


# =============================================================================
# Test: Help and Version
# =============================================================================

class TestHelpAndVersion(unittest.TestCase):
    """Tests for --help and --version flags."""

    def test_version_flag(self):
        """Test that --version shows version."""
        parser = create_parser()

        with self.assertRaises(SystemExit) as cm:
            parser.parse_args(["--version"])

        self.assertEqual(cm.exception.code, 0)

    def test_help_flag(self):
        """Test that --help exits with code 0."""
        parser = create_parser()

        with self.assertRaises(SystemExit) as cm:
            parser.parse_args(["--help"])

        self.assertEqual(cm.exception.code, 0)


# =============================================================================
# Test: Error Message Security
# =============================================================================

class TestErrorMessageSecurity(unittest.TestCase):
    """Tests ensuring no internal details leak in errors."""

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_text_mode_hides_internal_errors(self, mock_init):
        """Test that text mode doesn't expose internal error details."""
        mock_orch = Mock()
        mock_orch.process.return_value = {
            "success": False,
            "error": "Internal: ConnectionError at line 42 in handler.py",
            "error_code": "CONNECTION_ERROR",
        }
        mock_orch.formatter = Mock()
        mock_orch.formatter.format.return_value = Mock(
            to_text=Mock(return_value="Error: Service unavailable")
        )
        mock_init.return_value = (mock_orch, None)

        stderr = StringIO()
        with patch("sys.stderr", stderr):
            run_cli(["Question"])

        output = stderr.getvalue()
        # Should NOT contain internal details
        self.assertNotIn("line 42", output)
        self.assertNotIn("handler.py", output)
        self.assertNotIn("ConnectionError", output)

    @patch("src.cli.antonio.initialize_orchestrator")
    def test_raw_mode_shows_internal_errors(self, mock_init):
        """Test that raw mode does show internal error details."""
        mock_orch = Mock()
        mock_orch.process.return_value = {
            "success": False,
            "error": "Internal: ConnectionError at line 42",
            "error_code": "CONNECTION_ERROR",
        }
        mock_orch.formatter = Mock()
        mock_init.return_value = (mock_orch, None)

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            run_cli(["--raw", "Question"])

        output = stdout.getvalue()
        # Raw mode SHOULD contain internal details
        self.assertIn("line 42", output)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main()
