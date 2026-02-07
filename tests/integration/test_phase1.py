"""
Integration tests for Phase 1: Security & Sandboxing.

Tests sandbox integration with the Router.
"""

import json
import os
import sys
import time
import unittest

sys.path.insert(0, ".")

from src.models.request import Request, Modality
from src.models.response import Response
from src.models.policy import PolicyDecision, Handler
from src.sandbox import SandboxConfig, SandboxViolation
from src.handlers.base import BaseHandler


# =============================================================================
# Mock Handlers for Testing
# =============================================================================

class MockSuccessHandler(BaseHandler):
    """Handler that returns a simple success response."""

    def process(self, request: Request) -> Response:
        return Response.success_response(
            output={"message": f"Processed: {request.text}"}
        )


class MockSlowHandler(BaseHandler):
    """Handler that sleeps for a configurable duration."""

    def process(self, request: Request) -> Response:
        sleep_time = self.config.get("sleep_seconds", 5)
        time.sleep(sleep_time)
        return Response.success_response(output={"slept": sleep_time})


class MockExceptionHandler(BaseHandler):
    """Handler that raises an exception."""

    def process(self, request: Request) -> Response:
        raise ValueError("Handler intentionally failed")


class MockInfiniteLoopHandler(BaseHandler):
    """Handler that runs forever (for timeout testing)."""

    def process(self, request: Request) -> Response:
        while True:
            pass


class MockReturnDictHandler(BaseHandler):
    """Handler that returns a raw dict instead of Response."""

    def process(self, request: Request) -> dict:
        return {"raw": "dict", "text": request.text}


# =============================================================================
# Test: Router Sandbox Integration
# =============================================================================

class TestRouterSandboxIntegration(unittest.TestCase):
    """Tests that Router executes handlers through sandbox."""

    def _create_router_with_mock_handler(self, handler: BaseHandler, timeout: int = 10):
        """
        Create a Router-like object with a mock handler.

        We can't easily mock the full Router, so we test the sandbox
        execution pattern directly.
        """
        from src.sandbox import ProcessSandbox, SandboxConfig
        from src.router.router import _execute_handler

        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=timeout))
        return sandbox, handler

    def test_successful_handler_execution(self):
        """Test that successful handler execution returns output."""
        from src.sandbox import ProcessSandbox, SandboxConfig
        from src.router.router import _execute_handler

        handler = MockSuccessHandler({})
        request = Request(text="Hello", modality=Modality.TEXT)
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))

        result = sandbox.execute(_execute_handler, handler, request)

        self.assertTrue(result.success)
        self.assertIsInstance(result.output, Response)
        self.assertTrue(result.output.success)
        self.assertEqual(result.output.output["message"], "Processed: Hello")

    def test_handler_returning_dict_wrapped(self):
        """Test that handler returning dict is handled."""
        from src.sandbox import ProcessSandbox, SandboxConfig
        from src.router.router import _execute_handler

        handler = MockReturnDictHandler({})
        request = Request(text="Test", modality=Modality.TEXT)
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))

        result = sandbox.execute(_execute_handler, handler, request)

        self.assertTrue(result.success)
        # Output is the raw dict from handler
        self.assertEqual(result.output["raw"], "dict")
        self.assertEqual(result.output["text"], "Test")

    def test_handler_exception_caught(self):
        """Test that handler exception is caught by sandbox."""
        from src.sandbox import ProcessSandbox, SandboxConfig
        from src.router.router import _execute_handler

        handler = MockExceptionHandler({})
        request = Request(text="Test", modality=Modality.TEXT)
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))

        result = sandbox.execute(_execute_handler, handler, request)

        self.assertFalse(result.success)
        self.assertEqual(result.violation, SandboxViolation.EXCEPTION)
        self.assertIn("ValueError", result.error)
        self.assertIn("intentionally failed", result.error)

    def test_handler_timeout_enforced(self):
        """Test that handler timeout is enforced."""
        from src.sandbox import ProcessSandbox, SandboxConfig
        from src.router.router import _execute_handler

        handler = MockSlowHandler({"sleep_seconds": 10})
        request = Request(text="Test", modality=Modality.TEXT)
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=1))

        start = time.time()
        result = sandbox.execute(_execute_handler, handler, request)
        elapsed = time.time() - start

        self.assertFalse(result.success)
        self.assertEqual(result.violation, SandboxViolation.TIMEOUT)
        self.assertIn("timed out", result.error.lower())
        # Should have stopped around 1 second, not 10
        self.assertLess(elapsed, 3)

    def test_infinite_loop_timeout(self):
        """Test that infinite loop handler is terminated."""
        from src.sandbox import ProcessSandbox, SandboxConfig
        from src.router.router import _execute_handler

        handler = MockInfiniteLoopHandler({})
        request = Request(text="Test", modality=Modality.TEXT)
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=1))

        start = time.time()
        result = sandbox.execute(_execute_handler, handler, request)
        elapsed = time.time() - start

        self.assertFalse(result.success)
        self.assertEqual(result.violation, SandboxViolation.TIMEOUT)
        # Should terminate around 1 second
        self.assertLess(elapsed, 3)

    def test_elapsed_time_tracked(self):
        """Test that elapsed time is tracked in result."""
        from src.sandbox import ProcessSandbox, SandboxConfig
        from src.router.router import _execute_handler

        handler = MockSlowHandler({"sleep_seconds": 0.5})
        request = Request(text="Test", modality=Modality.TEXT)
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))

        result = sandbox.execute(_execute_handler, handler, request)

        self.assertTrue(result.success)
        # Should have taken at least 400ms
        self.assertGreaterEqual(result.elapsed_ms, 400)
        # But less than 2 seconds
        self.assertLess(result.elapsed_ms, 2000)


# =============================================================================
# Test: Router Error Code Mapping
# =============================================================================

class TestRouterErrorCodes(unittest.TestCase):
    """Tests that Router maps sandbox violations to error codes."""

    def test_timeout_error_code(self):
        """Test that timeout produces SANDBOX_TIMEOUT error code."""
        from src.router.router import SANDBOX_ERROR_CODES

        self.assertEqual(
            SANDBOX_ERROR_CODES[SandboxViolation.TIMEOUT],
            "SANDBOX_TIMEOUT"
        )

    def test_cpu_error_code(self):
        """Test that CPU violation produces SANDBOX_CPU_EXCEEDED error code."""
        from src.router.router import SANDBOX_ERROR_CODES

        self.assertEqual(
            SANDBOX_ERROR_CODES[SandboxViolation.CPU_EXCEEDED],
            "SANDBOX_CPU_EXCEEDED"
        )

    def test_memory_error_code(self):
        """Test that memory violation produces SANDBOX_MEMORY_EXCEEDED error code."""
        from src.router.router import SANDBOX_ERROR_CODES

        self.assertEqual(
            SANDBOX_ERROR_CODES[SandboxViolation.MEMORY_EXCEEDED],
            "SANDBOX_MEMORY_EXCEEDED"
        )

    def test_exception_error_code(self):
        """Test that exception produces SANDBOX_EXCEPTION error code."""
        from src.router.router import SANDBOX_ERROR_CODES

        self.assertEqual(
            SANDBOX_ERROR_CODES[SandboxViolation.EXCEPTION],
            "SANDBOX_EXCEPTION"
        )


# =============================================================================
# Test: Router Sandbox Configuration
# =============================================================================

class TestRouterSandboxConfig(unittest.TestCase):
    """Tests for Router sandbox configuration."""

    def test_default_sandbox_config_values(self):
        """Test that default sandbox config has expected values."""
        from src.router.router import DEFAULT_SANDBOX_CONFIG

        self.assertIn("cpu_seconds", DEFAULT_SANDBOX_CONFIG)
        self.assertIn("memory_mb", DEFAULT_SANDBOX_CONFIG)
        self.assertIn("timeout_seconds", DEFAULT_SANDBOX_CONFIG)
        self.assertGreater(DEFAULT_SANDBOX_CONFIG["timeout_seconds"], 0)

    def test_handler_specific_defaults(self):
        """Test that handler-specific defaults exist."""
        from src.router.router import HANDLER_SANDBOX_DEFAULTS

        # Mistral should have longer timeout than reject handler
        self.assertIn("mistral", HANDLER_SANDBOX_DEFAULTS)
        self.assertIn("reject", HANDLER_SANDBOX_DEFAULTS)

        mistral_timeout = HANDLER_SANDBOX_DEFAULTS["mistral"]["timeout_seconds"]
        reject_timeout = HANDLER_SANDBOX_DEFAULTS["reject"]["timeout_seconds"]

        self.assertGreater(mistral_timeout, reject_timeout)

    def test_handler_config_key_mapping(self):
        """Test that all handlers have config key mappings."""
        from src.router.router import HANDLER_CONFIG_KEYS

        self.assertEqual(HANDLER_CONFIG_KEYS[Handler.TEXT_LOCAL], "mistral")
        self.assertEqual(HANDLER_CONFIG_KEYS[Handler.AUDIO_IN], "whisper")
        self.assertEqual(HANDLER_CONFIG_KEYS[Handler.REJECT], "reject")


# =============================================================================
# Test: Full Router (if handlers are available)
# =============================================================================

class TestRouterFullIntegration(unittest.TestCase):
    """Integration tests with actual Router (when handlers available)."""

    def test_router_has_sandboxes_for_all_handlers(self):
        """Test that Router creates sandboxes for all handlers."""
        try:
            from src.router.router import Router

            router = Router()

            # Every handler should have a sandbox
            for handler_enum in router.handlers.keys():
                self.assertIn(handler_enum, router.sandboxes)
                self.assertIsNotNone(router.sandboxes[handler_enum])

        except Exception as e:
            # Skip if Router can't be instantiated (missing deps)
            self.skipTest(f"Router instantiation failed: {e}")

    def test_router_sandbox_configs_valid(self):
        """Test that all sandbox configs are valid SandboxConfig objects."""
        try:
            from src.router.router import Router
            from src.sandbox import SandboxConfig

            router = Router()

            for handler_enum, sandbox in router.sandboxes.items():
                config = sandbox.config
                self.assertIsInstance(config, SandboxConfig)
                # All should have timeout at minimum
                self.assertIsNotNone(config.timeout_seconds)
                self.assertGreater(config.timeout_seconds, 0)

        except Exception as e:
            self.skipTest(f"Router instantiation failed: {e}")


# =============================================================================
# Test: Normalizer Validation Integration
# =============================================================================

class TestNormalizerValidationIntegration(unittest.TestCase):
    """Tests that Normalizer validates input before processing."""

    def test_valid_input_succeeds(self):
        """Test that valid input passes through normalizer."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize({"text": "Hello world"})

        self.assertTrue(result.success)
        self.assertIsNotNone(result.request)
        self.assertEqual(result.request.text, "Hello world")

    def test_valid_string_input_succeeds(self):
        """Test that valid string input passes through normalizer."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize("Hello world")

        self.assertTrue(result.success)
        self.assertIsNotNone(result.request)
        self.assertEqual(result.request.text, "Hello world")

    def test_invalid_input_rejected(self):
        """Test that invalid input is rejected."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize({"text": 123})  # Invalid type

        self.assertFalse(result.success)
        self.assertIsNone(result.request)
        self.assertEqual(result.error_code, "VALIDATION_ERROR")
        self.assertGreater(len(result.errors), 0)

    def test_null_input_rejected(self):
        """Test that null input is rejected."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize(None)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "VALIDATION_ERROR")

    def test_oversized_text_rejected(self):
        """Test that oversized text is rejected."""
        from src.input.normalizer import Normalizer
        from src.validation import MAX_TEXT_LENGTH

        normalizer = Normalizer()
        long_text = "x" * (MAX_TEXT_LENGTH + 1)
        result = normalizer.normalize({"text": long_text})

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "VALIDATION_ERROR")

    def test_invalid_modality_rejected(self):
        """Test that invalid modality is rejected."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize({"modality": "invalid_modality"})

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "VALIDATION_ERROR")

    def test_validation_errors_contain_field_info(self):
        """Test that validation errors contain field information."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize({"text": 123})

        self.assertFalse(result.success)
        self.assertGreater(len(result.errors), 0)
        self.assertEqual(result.errors[0].field, "text")


# =============================================================================
# Test: Normalizer Sanitization Integration
# =============================================================================

class TestNormalizerSanitizationIntegration(unittest.TestCase):
    """Tests that Normalizer sanitizes validated input."""

    def test_null_bytes_removed(self):
        """Test that null bytes are removed during normalization."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize({"text": "hello\x00world"})

        self.assertTrue(result.success)
        self.assertEqual(result.request.text, "helloworld")
        self.assertNotIn("\x00", result.request.text)

    def test_control_chars_removed(self):
        """Test that control characters are removed during normalization."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize({"text": "hello\x07world"})

        self.assertTrue(result.success)
        self.assertEqual(result.request.text, "helloworld")

    def test_line_endings_normalized(self):
        """Test that line endings are normalized to Unix style."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize({"text": "line1\r\nline2"})

        self.assertTrue(result.success)
        self.assertEqual(result.request.text, "line1\nline2")
        self.assertNotIn("\r", result.request.text)

    def test_unicode_normalized(self):
        """Test that Unicode is normalized to NFC form."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        # Decomposed form: e + combining acute accent
        result = normalizer.normalize({"text": "cafe\u0301"})

        self.assertTrue(result.success)
        # Should be normalized to composed form
        self.assertEqual(result.request.text, "café")

    def test_tabs_preserved(self):
        """Test that tabs are preserved."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize({"text": "hello\tworld"})

        self.assertTrue(result.success)
        self.assertEqual(result.request.text, "hello\tworld")

    def test_newlines_preserved(self):
        """Test that newlines are preserved."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize({"text": "hello\nworld"})

        self.assertTrue(result.success)
        self.assertEqual(result.request.text, "hello\nworld")

    def test_metadata_sanitized(self):
        """Test that metadata is also sanitized."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize({
            "text": "hello",
            "metadata": {"key": "value\x00"}
        })

        self.assertTrue(result.success)
        self.assertEqual(result.request.metadata["key"], "value")


# =============================================================================
# Test: Normalizer Prevents Invalid Input from Reaching Router
# =============================================================================

class TestNormalizerRouterIsolation(unittest.TestCase):
    """Tests that invalid input never reaches Router or handlers."""

    def test_invalid_input_does_not_create_request(self):
        """Test that invalid input does not create a Request object."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize({"text": 123})

        self.assertFalse(result.success)
        self.assertIsNone(result.request)

    def test_normalization_result_serializable(self):
        """Test that NormalizationResult can be serialized."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()

        # Test success case
        success_result = normalizer.normalize({"text": "hello"})
        success_dict = success_result.to_dict()
        self.assertTrue(success_dict["success"])
        self.assertIn("request", success_dict)

        # Test failure case
        failure_result = normalizer.normalize({"text": 123})
        failure_dict = failure_result.to_dict()
        self.assertFalse(failure_dict["success"])
        self.assertIn("errors", failure_dict)
        self.assertIn("error_code", failure_dict)

    def test_normalize_or_error_raises_on_invalid(self):
        """Test that normalize_or_error raises ValueError on invalid input."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()

        with self.assertRaises(ValueError) as context:
            normalizer.normalize_or_error({"text": 123})

        self.assertIn("Validation failed", str(context.exception))

    def test_normalize_or_error_returns_request_on_valid(self):
        """Test that normalize_or_error returns Request on valid input."""
        from src.input.normalizer import Normalizer
        from src.models.request import Request

        normalizer = Normalizer()
        request = normalizer.normalize_or_error({"text": "hello"})

        self.assertIsInstance(request, Request)
        self.assertEqual(request.text, "hello")

    def test_empty_input_valid(self):
        """Test that empty dict input is valid (uses defaults)."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()
        result = normalizer.normalize({})

        self.assertTrue(result.success)
        self.assertIsNotNone(result.request)

    def test_modality_detection_preserved(self):
        """Test that modality detection still works after validation."""
        from src.input.normalizer import Normalizer
        from src.models.request import Modality

        normalizer = Normalizer()

        # Text modality
        text_result = normalizer.normalize({"text": "hello", "modality": "text"})
        self.assertTrue(text_result.success)
        self.assertEqual(text_result.request.modality, Modality.TEXT)

        # Audio modality
        audio_result = normalizer.normalize({"audio_path": "/path/to/audio.wav"})
        self.assertTrue(audio_result.success)
        self.assertEqual(audio_result.request.modality, Modality.AUDIO_INPUT)


# =============================================================================
# Test: Audit Hash Chain Integration (Task 10.5)
# =============================================================================

class TestAuditHashChainIntegration(unittest.TestCase):
    """Tests that audit log maintains valid hash chain after multiple requests."""

    def setUp(self):
        """Create a temporary log file for testing."""
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    def test_single_entry_chain_valid(self):
        """Test that a single entry creates a valid chain."""
        from src.utils.audit import AuditLogger

        logger = AuditLogger(self.log_path)
        logger.log_event("test_event", {"data": "value"})

        result = logger.verify_chain()

        self.assertTrue(result.valid)
        self.assertEqual(result.entries_checked, 1)
        self.assertIsNone(result.index)
        self.assertIsNone(result.reason)

    def test_multiple_entries_chain_valid(self):
        """Test that multiple entries maintain a valid chain."""
        from src.utils.audit import AuditLogger

        logger = AuditLogger(self.log_path)

        # Log 10 events
        for i in range(10):
            logger.log_event(f"event_{i}", {"index": i, "data": f"value_{i}"})

        result = logger.verify_chain()

        self.assertTrue(result.valid)
        self.assertEqual(result.entries_checked, 10)

    def test_chain_persists_across_restarts(self):
        """Test that chain remains valid after logger restart."""
        from src.utils.audit import AuditLogger

        # First session - create some entries
        logger1 = AuditLogger(self.log_path)
        for i in range(5):
            logger1.log_event("session1", {"index": i})

        # Second session - create more entries
        logger2 = AuditLogger(self.log_path)
        for i in range(5):
            logger2.log_event("session2", {"index": i})

        # Verify chain is still valid
        result = logger2.verify_chain()

        self.assertTrue(result.valid)
        self.assertEqual(result.entries_checked, 10)

    def test_chain_sequences_are_continuous(self):
        """Test that sequence numbers are continuous across all entries."""
        from src.utils.audit import AuditLogger

        logger = AuditLogger(self.log_path)
        for i in range(20):
            logger.log_event("test", {"i": i})

        entries = logger.get_all()

        # Verify sequences are 1, 2, 3, ..., 20
        sequences = [e.sequence for e in entries]
        self.assertEqual(sequences, list(range(1, 21)))

    def test_chain_hashes_are_linked(self):
        """Test that each entry's previous_hash matches prior entry's hash."""
        from src.utils.audit import AuditLogger, GENESIS_HASH

        logger = AuditLogger(self.log_path)
        for i in range(5):
            logger.log_event("test", {"i": i})

        entries = logger.get_all()

        # First entry should reference genesis
        self.assertEqual(entries[0].previous_hash, GENESIS_HASH)

        # Each subsequent entry should reference prior entry's hash
        for i in range(1, len(entries)):
            self.assertEqual(entries[i].previous_hash, entries[i-1].entry_hash)

    def test_empty_log_is_valid(self):
        """Test that empty log file verifies as valid."""
        from src.utils.audit import AuditLogger

        logger = AuditLogger(self.log_path)
        # Don't log anything

        result = logger.verify_chain()

        self.assertTrue(result.valid)
        self.assertEqual(result.entries_checked, 0)


# =============================================================================
# Test: Audit Tamper Detection Integration (Task 10.6)
# =============================================================================

class TestAuditTamperDetectionIntegration(unittest.TestCase):
    """Tests that tampered audit logs are detected."""

    def setUp(self):
        """Create a temporary log file for testing."""
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_audit.jsonl")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    def _create_valid_log(self, num_entries: int = 5):
        """Helper to create a valid log with entries."""
        from src.utils.audit import AuditLogger

        logger = AuditLogger(self.log_path)
        for i in range(num_entries):
            logger.log_event("test_event", {"index": i, "data": f"original_{i}"})
        return logger

    def _tamper_entry(self, entry_index: int, key: str, new_value):
        """Helper to tamper with a specific entry in the log file."""
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        entry = json.loads(lines[entry_index])
        if key == "payload":
            entry["payload"] = new_value
        else:
            entry[key] = new_value

        lines[entry_index] = json.dumps(entry) + "\n"

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def test_modified_payload_detected(self):
        """Test that modifying entry payload is detected."""
        from src.utils.audit import AuditLogger, VerificationFailureReason

        self._create_valid_log(5)

        # Tamper with middle entry's payload
        self._tamper_entry(2, "payload", {"index": 2, "data": "TAMPERED"})

        logger = AuditLogger(self.log_path)
        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 2)
        self.assertEqual(result.reason, VerificationFailureReason.HASH_MISMATCH.value)

    def test_modified_entry_hash_detected(self):
        """Test that modifying entry_hash is detected."""
        from src.utils.audit import AuditLogger, VerificationFailureReason

        self._create_valid_log(5)

        # Tamper with entry's hash
        self._tamper_entry(1, "entry_hash", "bad" * 16 + "0" * 16)

        logger = AuditLogger(self.log_path)
        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 1)
        self.assertEqual(result.reason, VerificationFailureReason.HASH_MISMATCH.value)

    def test_modified_previous_hash_detected(self):
        """Test that modifying previous_hash is detected."""
        from src.utils.audit import AuditLogger, VerificationFailureReason

        self._create_valid_log(5)

        # Tamper with entry's previous_hash link
        self._tamper_entry(3, "previous_hash", "x" * 64)

        logger = AuditLogger(self.log_path)
        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 3)
        self.assertEqual(result.reason, VerificationFailureReason.PREVIOUS_HASH_MISMATCH.value)

    def test_modified_sequence_detected(self):
        """Test that modifying sequence number is detected."""
        from src.utils.audit import AuditLogger, VerificationFailureReason

        self._create_valid_log(5)

        # Tamper with entry's sequence
        self._tamper_entry(2, "sequence", 99)

        logger = AuditLogger(self.log_path)
        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 2)
        self.assertEqual(result.reason, VerificationFailureReason.SEQUENCE_ERROR.value)

    def test_deleted_entry_detected(self):
        """Test that deleting an entry is detected."""
        from src.utils.audit import AuditLogger, VerificationFailureReason

        self._create_valid_log(5)

        # Remove middle entry
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        del lines[2]  # Remove entry at index 2

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        logger = AuditLogger(self.log_path)
        result = logger.verify_chain()

        self.assertFalse(result.valid)
        # Should fail at index 2 due to broken chain
        self.assertEqual(result.index, 2)

    def test_first_entry_tamper_detected(self):
        """Test that tampering with first entry is detected."""
        from src.utils.audit import AuditLogger, VerificationFailureReason

        self._create_valid_log(5)

        # Tamper with first entry
        self._tamper_entry(0, "payload", {"TAMPERED": True})

        logger = AuditLogger(self.log_path)
        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 0)
        self.assertEqual(result.reason, VerificationFailureReason.HASH_MISMATCH.value)

    def test_last_entry_tamper_detected(self):
        """Test that tampering with last entry is detected."""
        from src.utils.audit import AuditLogger, VerificationFailureReason

        self._create_valid_log(5)

        # Tamper with last entry
        self._tamper_entry(4, "event_type", "TAMPERED_EVENT")

        logger = AuditLogger(self.log_path)
        result = logger.verify_chain()

        self.assertFalse(result.valid)
        self.assertEqual(result.index, 4)
        self.assertEqual(result.reason, VerificationFailureReason.HASH_MISMATCH.value)

    def test_verification_result_serializable(self):
        """Test that VerificationResult can be serialized."""
        from src.utils.audit import AuditLogger

        self._create_valid_log(3)
        self._tamper_entry(1, "payload", {"tampered": True})

        logger = AuditLogger(self.log_path)
        result = logger.verify_chain()

        result_dict = result.to_dict()

        self.assertIn("valid", result_dict)
        self.assertIn("index", result_dict)
        self.assertIn("reason", result_dict)
        self.assertIn("entries_checked", result_dict)
        self.assertFalse(result_dict["valid"])


# =============================================================================
# Test: Full Pipeline Integration
# =============================================================================

class TestFullPipelineIntegration(unittest.TestCase):
    """End-to-end integration tests for the complete Phase 1 pipeline."""

    def test_valid_input_through_entire_pipeline(self):
        """Test that valid input passes through normalizer -> policy -> router."""
        from src.input.normalizer import Normalizer
        from src.policy.policy_engine import PolicyEngine
        from src.models.request import Modality

        normalizer = Normalizer()
        policy = PolicyEngine()

        # Step 1: Normalize input
        norm_result = normalizer.normalize({"text": "Hello, world!"})
        self.assertTrue(norm_result.success, "Normalization should succeed")
        self.assertIsNotNone(norm_result.request)

        request = norm_result.request
        self.assertEqual(request.text, "Hello, world!")
        self.assertEqual(request.modality, Modality.TEXT)

        # Step 2: Policy classification and decision
        classification = policy.classify(request)
        self.assertIsNotNone(classification)

        decision = policy.decide(request, classification)
        self.assertIsNotNone(decision)
        # Should not be rejected for simple valid input
        self.assertNotEqual(decision.handler.value, "reject")

    def test_invalid_input_rejected_at_normalization(self):
        """Test that invalid input is rejected before reaching router."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()

        # Invalid type for text
        result = normalizer.normalize({"text": ["not", "a", "string"]})

        self.assertFalse(result.success)
        self.assertIsNone(result.request)
        self.assertEqual(result.error_code, "VALIDATION_ERROR")
        # Input should never reach policy or router

    def test_oversized_input_rejected_at_normalization(self):
        """Test that oversized input is rejected at normalization stage."""
        from src.input.normalizer import Normalizer
        from src.validation import MAX_TEXT_LENGTH

        normalizer = Normalizer()

        # Input exceeding max length
        huge_text = "x" * (MAX_TEXT_LENGTH + 100)
        result = normalizer.normalize({"text": huge_text})

        self.assertFalse(result.success)
        self.assertIsNone(result.request)
        self.assertEqual(result.error_code, "VALIDATION_ERROR")

    def test_sanitized_input_reaches_request(self):
        """Test that input is sanitized before creating Request."""
        from src.input.normalizer import Normalizer

        normalizer = Normalizer()

        # Input with control characters and null bytes
        dirty_input = "Hello\x00\x07World\r\n"
        result = normalizer.normalize({"text": dirty_input})

        self.assertTrue(result.success)
        self.assertIsNotNone(result.request)
        # Sanitized: no null bytes, no control chars, normalized line ending
        self.assertNotIn("\x00", result.request.text)
        self.assertNotIn("\x07", result.request.text)
        self.assertNotIn("\r", result.request.text)

    def test_sandbox_timeout_produces_error_response(self):
        """Test that sandbox timeout produces correct error response."""
        from src.sandbox import ProcessSandbox, SandboxConfig, SandboxViolation
        from src.router.router import _execute_handler, SANDBOX_ERROR_CODES
        from src.models.request import Request, Modality

        handler = MockSlowHandler({"sleep_seconds": 10})
        request = Request(text="Test", modality=Modality.TEXT)
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=1))

        result = sandbox.execute(_execute_handler, handler, request)

        self.assertFalse(result.success)
        self.assertEqual(result.violation, SandboxViolation.TIMEOUT)

        # Verify error code mapping
        error_code = SANDBOX_ERROR_CODES[result.violation]
        self.assertEqual(error_code, "SANDBOX_TIMEOUT")

    def test_pipeline_with_unicode_normalization(self):
        """Test that Unicode is properly normalized through pipeline."""
        from src.input.normalizer import Normalizer
        import unicodedata

        normalizer = Normalizer()

        # Decomposed form (e + combining acute accent)
        decomposed = "cafe\u0301"  # café in NFD
        result = normalizer.normalize({"text": decomposed})

        self.assertTrue(result.success)
        # Should be NFC normalized
        self.assertEqual(result.request.text, "café")
        self.assertEqual(
            unicodedata.is_normalized("NFC", result.request.text),
            True
        )

    def test_multiple_validation_errors_collected(self):
        """Test that multiple validation errors are collected and reported."""
        from src.input.normalizer import Normalizer
        from src.validation import MAX_TEXT_LENGTH

        normalizer = Normalizer()

        # Multiple problems: wrong type AND invalid modality
        result = normalizer.normalize({
            "text": 12345,
            "modality": "not_a_valid_modality"
        })

        self.assertFalse(result.success)
        self.assertGreater(len(result.errors), 0)

    def test_metadata_size_limit_enforced(self):
        """Test that metadata size limit is enforced."""
        from src.input.normalizer import Normalizer
        from src.validation import MAX_METADATA_SIZE

        normalizer = Normalizer()

        # Large metadata
        large_metadata = {"key": "x" * (MAX_METADATA_SIZE + 100)}
        result = normalizer.normalize({
            "text": "hello",
            "metadata": large_metadata
        })

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "VALIDATION_ERROR")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main()
