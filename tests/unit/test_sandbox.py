"""
Unit tests for the ProcessSandbox module.

Tests cover:
- Successful execution
- Timeout violation
- Memory violation (Unix only)
- CPU time violation (Unix only)
- Unexpected exception in sandboxed process
- Configuration validation
- Result serialization
"""

import platform
import sys
import time
import unittest

# Add src to path for imports
sys.path.insert(0, ".")

from src.sandbox import (
    ProcessSandbox,
    SandboxConfig,
    SandboxResult,
    SandboxViolation,
    SandboxError,
)


# =============================================================================
# Test helper functions (executed in child process)
# =============================================================================

def simple_add(a: int, b: int) -> int:
    """Simple function that returns sum of two numbers."""
    return a + b


def return_none() -> None:
    """Function that returns None."""
    return None


def return_string() -> str:
    """Function that returns a string."""
    return "hello from sandbox"


def return_dict() -> dict:
    """Function that returns a dictionary."""
    return {"key": "value", "number": 42}


def sleep_function(seconds: float) -> str:
    """Function that sleeps for specified duration."""
    time.sleep(seconds)
    return "slept"


def raise_exception() -> None:
    """Function that raises an exception."""
    raise ValueError("This is a test exception")


def raise_runtime_error() -> None:
    """Function that raises a RuntimeError."""
    raise RuntimeError("Runtime error in sandbox")


def divide_by_zero() -> float:
    """Function that causes division by zero."""
    return 1 / 0


def infinite_loop() -> None:
    """Function that runs forever (for timeout testing)."""
    while True:
        pass


def cpu_intensive() -> int:
    """Function that burns CPU (for CPU limit testing)."""
    result = 0
    for i in range(100_000_000):
        result += i * i
    return result


def memory_hog() -> list:
    """Function that allocates lots of memory."""
    # Allocate ~1GB of memory
    data = []
    for _ in range(1000):
        data.append("x" * (1024 * 1024))  # 1MB strings
    return data


def allocate_memory_mb(mb: int) -> str:
    """Allocate specified amount of memory."""
    data = "x" * (mb * 1024 * 1024)
    return f"allocated {len(data)} bytes"


# =============================================================================
# Test: SandboxConfig
# =============================================================================

class TestSandboxConfig(unittest.TestCase):
    """Tests for SandboxConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SandboxConfig()
        self.assertIsNone(config.cpu_seconds)
        self.assertIsNone(config.memory_mb)
        self.assertEqual(config.timeout_seconds, 60)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SandboxConfig(cpu_seconds=10, memory_mb=256, timeout_seconds=30)
        self.assertEqual(config.cpu_seconds, 10)
        self.assertEqual(config.memory_mb, 256)
        self.assertEqual(config.timeout_seconds, 30)

    def test_invalid_cpu_seconds(self):
        """Test that non-positive cpu_seconds raises ValueError."""
        with self.assertRaises(ValueError):
            SandboxConfig(cpu_seconds=0)
        with self.assertRaises(ValueError):
            SandboxConfig(cpu_seconds=-1)

    def test_invalid_memory_mb(self):
        """Test that non-positive memory_mb raises ValueError."""
        with self.assertRaises(ValueError):
            SandboxConfig(memory_mb=0)
        with self.assertRaises(ValueError):
            SandboxConfig(memory_mb=-1)

    def test_invalid_timeout_seconds(self):
        """Test that non-positive timeout_seconds raises ValueError."""
        with self.assertRaises(ValueError):
            SandboxConfig(timeout_seconds=0)
        with self.assertRaises(ValueError):
            SandboxConfig(timeout_seconds=-1)


# =============================================================================
# Test: SandboxResult
# =============================================================================

class TestSandboxResult(unittest.TestCase):
    """Tests for SandboxResult dataclass."""

    def test_success_result(self):
        """Test successful result structure."""
        result = SandboxResult(
            success=True,
            output=42,
            violation=SandboxViolation.NONE,
            elapsed_ms=100,
            exit_code=0,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.output, 42)
        self.assertIsNone(result.error)
        self.assertEqual(result.violation, SandboxViolation.NONE)

    def test_failure_result(self):
        """Test failure result structure."""
        result = SandboxResult(
            success=False,
            error="Timeout exceeded",
            violation=SandboxViolation.TIMEOUT,
            elapsed_ms=5000,
            exit_code=-1,
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.output)
        self.assertEqual(result.error, "Timeout exceeded")
        self.assertEqual(result.violation, SandboxViolation.TIMEOUT)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = SandboxResult(
            success=True,
            output={"key": "value"},
            violation=SandboxViolation.NONE,
            elapsed_ms=50,
            exit_code=0,
        )
        d = result.to_dict()
        self.assertEqual(d["success"], True)
        self.assertEqual(d["output"], {"key": "value"})
        self.assertEqual(d["violation"], "none")
        self.assertEqual(d["elapsed_ms"], 50)

    def test_to_dict_failure(self):
        """Test serialization of failure result."""
        result = SandboxResult(
            success=False,
            output="ignored",  # Should be None in dict
            error="Test error",
            violation=SandboxViolation.EXCEPTION,
            elapsed_ms=10,
            exit_code=1,
        )
        d = result.to_dict()
        self.assertEqual(d["success"], False)
        self.assertIsNone(d["output"])  # Output should be None for failures
        self.assertEqual(d["error"], "Test error")
        self.assertEqual(d["violation"], "exception")


# =============================================================================
# Test: ProcessSandbox - Successful Execution
# =============================================================================

class TestProcessSandboxSuccess(unittest.TestCase):
    """Tests for successful sandbox execution."""

    def test_simple_function(self):
        """Test execution of a simple function."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))
        result = sandbox.execute(simple_add, 2, 3)

        self.assertTrue(result.success)
        self.assertEqual(result.output, 5)
        self.assertEqual(result.violation, SandboxViolation.NONE)
        self.assertEqual(result.exit_code, 0)

    def test_function_returns_none(self):
        """Test function that returns None."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))
        result = sandbox.execute(return_none)

        self.assertTrue(result.success)
        self.assertIsNone(result.output)
        self.assertEqual(result.violation, SandboxViolation.NONE)

    def test_function_returns_string(self):
        """Test function that returns a string."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))
        result = sandbox.execute(return_string)

        self.assertTrue(result.success)
        self.assertEqual(result.output, "hello from sandbox")

    def test_function_returns_dict(self):
        """Test function that returns a dictionary."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))
        result = sandbox.execute(return_dict)

        self.assertTrue(result.success)
        self.assertEqual(result.output, {"key": "value", "number": 42})

    def test_function_with_kwargs(self):
        """Test function called with keyword arguments."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))
        result = sandbox.execute(simple_add, a=10, b=20)

        self.assertTrue(result.success)
        self.assertEqual(result.output, 30)

    def test_function_with_mixed_args(self):
        """Test function called with mixed positional and keyword arguments."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))
        result = sandbox.execute(simple_add, 5, b=15)

        self.assertTrue(result.success)
        self.assertEqual(result.output, 20)

    def test_elapsed_time_tracked(self):
        """Test that elapsed time is tracked correctly."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))
        result = sandbox.execute(sleep_function, 0.5)

        self.assertTrue(result.success)
        # Should take at least 500ms
        self.assertGreaterEqual(result.elapsed_ms, 400)
        # But not too long (allow some overhead)
        self.assertLess(result.elapsed_ms, 2000)


# =============================================================================
# Test: ProcessSandbox - Timeout Violation
# =============================================================================

class TestProcessSandboxTimeout(unittest.TestCase):
    """Tests for timeout violation handling."""

    def test_timeout_exceeded(self):
        """Test that function exceeding timeout is terminated."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=1))
        result = sandbox.execute(infinite_loop)

        self.assertFalse(result.success)
        self.assertEqual(result.violation, SandboxViolation.TIMEOUT)
        self.assertIn("timed out", result.error.lower())
        # Should be approximately 1 second
        self.assertGreaterEqual(result.elapsed_ms, 900)

    def test_sleep_exceeds_timeout(self):
        """Test that sleeping function exceeding timeout is terminated."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=1))
        result = sandbox.execute(sleep_function, 5)

        self.assertFalse(result.success)
        self.assertEqual(result.violation, SandboxViolation.TIMEOUT)

    def test_function_completes_before_timeout(self):
        """Test that function completing before timeout succeeds."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=5))
        result = sandbox.execute(sleep_function, 0.1)

        self.assertTrue(result.success)
        self.assertEqual(result.output, "slept")


# =============================================================================
# Test: ProcessSandbox - Exception Handling
# =============================================================================

class TestProcessSandboxException(unittest.TestCase):
    """Tests for exception handling in sandboxed process."""

    def test_value_error_caught(self):
        """Test that ValueError in sandboxed function is caught."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))
        result = sandbox.execute(raise_exception)

        self.assertFalse(result.success)
        self.assertEqual(result.violation, SandboxViolation.EXCEPTION)
        self.assertIn("ValueError", result.error)
        self.assertIn("test exception", result.error)

    def test_runtime_error_caught(self):
        """Test that RuntimeError in sandboxed function is caught."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))
        result = sandbox.execute(raise_runtime_error)

        self.assertFalse(result.success)
        self.assertEqual(result.violation, SandboxViolation.EXCEPTION)
        self.assertIn("RuntimeError", result.error)

    def test_division_by_zero_caught(self):
        """Test that ZeroDivisionError is caught."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))
        result = sandbox.execute(divide_by_zero)

        self.assertFalse(result.success)
        self.assertEqual(result.violation, SandboxViolation.EXCEPTION)
        self.assertIn("ZeroDivisionError", result.error)

    def test_non_callable_rejected(self):
        """Test that non-callable target is rejected."""
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=10))
        result = sandbox.execute("not a function")

        self.assertFalse(result.success)
        self.assertEqual(result.violation, SandboxViolation.EXCEPTION)
        self.assertIn("callable", result.error.lower())


# =============================================================================
# Test: ProcessSandbox - Memory Violation (Unix only)
# =============================================================================

@unittest.skipIf(platform.system() == "Windows", "Memory limits not supported on Windows")
class TestProcessSandboxMemory(unittest.TestCase):
    """Tests for memory limit enforcement (Unix only)."""

    def test_memory_limit_exceeded(self):
        """Test that function exceeding memory limit is terminated."""
        # Set a small memory limit (50MB)
        sandbox = ProcessSandbox(SandboxConfig(memory_mb=50, timeout_seconds=30))
        result = sandbox.execute(allocate_memory_mb, 100)  # Try to allocate 100MB

        self.assertFalse(result.success)
        self.assertEqual(result.violation, SandboxViolation.MEMORY_EXCEEDED)

    def test_memory_within_limit(self):
        """Test that function within memory limit succeeds."""
        # Set reasonable memory limit
        sandbox = ProcessSandbox(SandboxConfig(memory_mb=256, timeout_seconds=10))
        result = sandbox.execute(allocate_memory_mb, 10)  # Allocate 10MB

        self.assertTrue(result.success)
        self.assertIn("allocated", result.output)


# =============================================================================
# Test: ProcessSandbox - CPU Violation (Unix only)
# =============================================================================

@unittest.skipIf(platform.system() == "Windows", "CPU limits not supported on Windows")
class TestProcessSandboxCPU(unittest.TestCase):
    """Tests for CPU time limit enforcement (Unix only)."""

    def test_cpu_limit_exceeded(self):
        """Test that function exceeding CPU limit is terminated."""
        # Set a very short CPU limit
        sandbox = ProcessSandbox(SandboxConfig(cpu_seconds=1, timeout_seconds=60))
        result = sandbox.execute(cpu_intensive)

        self.assertFalse(result.success)
        self.assertEqual(result.violation, SandboxViolation.CPU_EXCEEDED)


# =============================================================================
# Test: ProcessSandbox - Default Configuration
# =============================================================================

class TestProcessSandboxDefaults(unittest.TestCase):
    """Tests for default sandbox behavior."""

    def test_default_config(self):
        """Test sandbox with default configuration."""
        sandbox = ProcessSandbox()
        result = sandbox.execute(simple_add, 1, 1)

        self.assertTrue(result.success)
        self.assertEqual(result.output, 2)

    def test_none_config(self):
        """Test sandbox with None configuration."""
        sandbox = ProcessSandbox(None)
        result = sandbox.execute(simple_add, 1, 1)

        self.assertTrue(result.success)
        self.assertEqual(result.output, 2)


# =============================================================================
# Test: SandboxError
# =============================================================================

class TestSandboxError(unittest.TestCase):
    """Tests for SandboxError exception."""

    def test_error_creation(self):
        """Test SandboxError creation."""
        error = SandboxError(SandboxViolation.TIMEOUT, "Test message")
        self.assertEqual(error.violation, SandboxViolation.TIMEOUT)
        self.assertEqual(error.message, "Test message")
        self.assertIn("timeout", str(error))
        self.assertIn("Test message", str(error))


# =============================================================================
# Test: SandboxViolation
# =============================================================================

class TestSandboxViolation(unittest.TestCase):
    """Tests for SandboxViolation enum."""

    def test_violation_values(self):
        """Test that violation enum has expected values."""
        self.assertEqual(SandboxViolation.NONE.value, "none")
        self.assertEqual(SandboxViolation.TIMEOUT.value, "timeout")
        self.assertEqual(SandboxViolation.CPU_EXCEEDED.value, "cpu_exceeded")
        self.assertEqual(SandboxViolation.MEMORY_EXCEEDED.value, "memory_exceeded")
        self.assertEqual(SandboxViolation.EXCEPTION.value, "exception")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main()
