"""
Process Sandbox - Execute code with resource limits.

This module provides process-based isolation for handler execution.
Resource limits (CPU, memory, timeout) are enforced to prevent
runaway processes from affecting the orchestrator.

Platform notes:
- Unix: Full support for CPU and memory limits via resource.setrlimit
- Windows: Timeout only; CPU/memory limits logged as warnings

Design principles:
- Never crash the orchestrator
- Always return a structured result
- Deterministic and synchronous
- No network or filesystem isolation (out of scope)
"""

import multiprocessing
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple


class SandboxViolation(Enum):
    """
    Types of sandbox violations.

    These map to specific error codes returned when limits are exceeded.
    """
    NONE = "none"
    TIMEOUT = "timeout"
    CPU_EXCEEDED = "cpu_exceeded"
    MEMORY_EXCEEDED = "memory_exceeded"
    EXCEPTION = "exception"


class SandboxError(Exception):
    """
    Exception raised when sandbox execution fails.

    Contains the violation type and any additional context.
    """

    def __init__(self, violation: SandboxViolation, message: str):
        self.violation = violation
        self.message = message
        super().__init__(f"{violation.value}: {message}")


@dataclass
class SandboxConfig:
    """
    Configuration for sandbox execution.

    All limits are optional. If not set, no limit is enforced for that resource.

    Attributes:
        cpu_seconds: Maximum CPU time in seconds (Unix only)
        memory_mb: Maximum memory in megabytes (Unix only)
        timeout_seconds: Maximum wall-clock time in seconds (all platforms)
    """
    cpu_seconds: Optional[int] = None
    memory_mb: Optional[int] = None
    timeout_seconds: Optional[int] = 60

    def __post_init__(self):
        """Validate configuration values."""
        if self.cpu_seconds is not None and self.cpu_seconds <= 0:
            raise ValueError("cpu_seconds must be positive")
        if self.memory_mb is not None and self.memory_mb <= 0:
            raise ValueError("memory_mb must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass
class SandboxResult:
    """
    Result of sandbox execution.

    Attributes:
        success: True if execution completed without violation
        output: The return value of the executed function (if successful)
        error: Error message (if failed)
        violation: Type of violation that occurred (if any)
        elapsed_ms: Wall-clock time in milliseconds
        exit_code: Process exit code (0 = success)
    """
    success: bool
    output: Any = None
    error: Optional[str] = None
    violation: SandboxViolation = SandboxViolation.NONE
    elapsed_ms: int = 0
    exit_code: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "output": self.output if self.success else None,
            "error": self.error,
            "violation": self.violation.value,
            "elapsed_ms": self.elapsed_ms,
            "exit_code": self.exit_code,
        }


def _apply_resource_limits(cpu_seconds: Optional[int], memory_mb: Optional[int]) -> None:
    """
    Apply resource limits in the current process.

    Called inside the child process before executing user code.
    Only works on Unix systems; silently skipped on Windows.
    """
    if platform.system() == "Windows":
        return

    try:
        import resource

        if cpu_seconds is not None:
            # RLIMIT_CPU is CPU time in seconds
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

        if memory_mb is not None:
            # RLIMIT_AS is address space (virtual memory) in bytes
            memory_bytes = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    except (ImportError, OSError, ValueError) as e:
        # Log but don't fail - limits are best-effort
        print(f"[SANDBOX WARNING] Could not apply resource limits: {e}", file=sys.stderr)


def _worker_process(
    target_func: Callable[..., Any],
    args: Tuple,
    kwargs: Dict,
    result_queue: multiprocessing.Queue,
    cpu_seconds: Optional[int],
    memory_mb: Optional[int],
) -> None:
    """
    Worker function executed in child process.

    Applies resource limits, executes target function, and sends result back.
    """
    # Apply resource limits before executing user code
    _apply_resource_limits(cpu_seconds, memory_mb)

    try:
        # Execute the target function
        output = target_func(*args, **kwargs)
        result_queue.put(("success", output, None))

    except MemoryError:
        result_queue.put(("memory", None, "Memory limit exceeded"))

    except Exception as e:
        # Capture full traceback for debugging
        tb = traceback.format_exc()
        result_queue.put(("exception", None, f"{type(e).__name__}: {e}\n{tb}"))


class ProcessSandbox:
    """
    Process-based sandbox for executing code with resource limits.

    Usage:
        sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=30, memory_mb=512))
        result = sandbox.execute(my_function, arg1, arg2, kwarg1=value1)

        if result.success:
            print(result.output)
        else:
            print(f"Failed: {result.violation.value} - {result.error}")

    The sandbox:
    - Executes the function in a separate process
    - Enforces CPU, memory, and timeout limits
    - Returns a structured result (never raises exceptions)
    - Cleans up child processes on timeout

    Platform limitations:
    - Windows: Only timeout is enforced; CPU/memory limits are ignored
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize sandbox with configuration.

        Args:
            config: Resource limits configuration. Uses defaults if not provided.
        """
        self.config = config or SandboxConfig()
        self._is_windows = platform.system() == "Windows"

        if self._is_windows:
            if self.config.cpu_seconds is not None or self.config.memory_mb is not None:
                print(
                    "[SANDBOX WARNING] CPU and memory limits are not supported on Windows. "
                    "Only timeout will be enforced.",
                    file=sys.stderr
                )

    def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> SandboxResult:
        """
        Execute a function in the sandbox.

        Args:
            func: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            SandboxResult with execution outcome

        Note:
            This method never raises exceptions. All errors are captured
            in the SandboxResult object.
        """
        start_time = time.time()

        # Validate that func is callable
        if not callable(func):
            return SandboxResult(
                success=False,
                error="Target must be callable",
                violation=SandboxViolation.EXCEPTION,
                elapsed_ms=0,
                exit_code=1,
            )

        # Create queue for result communication
        result_queue = multiprocessing.Queue()

        # Create child process
        process = multiprocessing.Process(
            target=_worker_process,
            args=(
                func,
                args,
                kwargs,
                result_queue,
                self.config.cpu_seconds,
                self.config.memory_mb,
            ),
        )

        try:
            process.start()

            # Wait for process with timeout
            timeout = self.config.timeout_seconds
            process.join(timeout=timeout)

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Check if process is still running (timeout)
            if process.is_alive():
                # Terminate the process
                process.terminate()
                process.join(timeout=5)

                # Force kill if still alive
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)

                return SandboxResult(
                    success=False,
                    error=f"Execution timed out after {timeout} seconds",
                    violation=SandboxViolation.TIMEOUT,
                    elapsed_ms=elapsed_ms,
                    exit_code=-1,
                )

            # Process finished - check exit code
            exit_code = process.exitcode or 0

            # Check for CPU limit (SIGXCPU = -24 or exit code 137 on some systems)
            if exit_code == -24 or exit_code == 152:  # 152 = 128 + 24 (SIGXCPU)
                return SandboxResult(
                    success=False,
                    error="CPU time limit exceeded",
                    violation=SandboxViolation.CPU_EXCEEDED,
                    elapsed_ms=elapsed_ms,
                    exit_code=exit_code,
                )

            # Check for memory limit (SIGKILL from OOM or exit from MemoryError)
            if exit_code == -9 or exit_code == 137:  # 137 = 128 + 9 (SIGKILL)
                # Could be OOM killer, treat as memory exceeded
                return SandboxResult(
                    success=False,
                    error="Memory limit exceeded (process killed)",
                    violation=SandboxViolation.MEMORY_EXCEEDED,
                    elapsed_ms=elapsed_ms,
                    exit_code=exit_code,
                )

            # Try to get result from queue
            try:
                if not result_queue.empty():
                    status, output, error = result_queue.get_nowait()

                    if status == "success":
                        return SandboxResult(
                            success=True,
                            output=output,
                            violation=SandboxViolation.NONE,
                            elapsed_ms=elapsed_ms,
                            exit_code=0,
                        )
                    elif status == "memory":
                        return SandboxResult(
                            success=False,
                            error=error,
                            violation=SandboxViolation.MEMORY_EXCEEDED,
                            elapsed_ms=elapsed_ms,
                            exit_code=1,
                        )
                    else:  # exception
                        return SandboxResult(
                            success=False,
                            error=error,
                            violation=SandboxViolation.EXCEPTION,
                            elapsed_ms=elapsed_ms,
                            exit_code=1,
                        )
                else:
                    # Process exited without putting result in queue
                    if exit_code != 0:
                        return SandboxResult(
                            success=False,
                            error=f"Process exited with code {exit_code}",
                            violation=SandboxViolation.EXCEPTION,
                            elapsed_ms=elapsed_ms,
                            exit_code=exit_code,
                        )
                    else:
                        # Successful exit but no result - function returned None implicitly
                        return SandboxResult(
                            success=True,
                            output=None,
                            violation=SandboxViolation.NONE,
                            elapsed_ms=elapsed_ms,
                            exit_code=0,
                        )

            except Exception as e:
                return SandboxResult(
                    success=False,
                    error=f"Failed to retrieve result: {e}",
                    violation=SandboxViolation.EXCEPTION,
                    elapsed_ms=elapsed_ms,
                    exit_code=exit_code,
                )

        except Exception as e:
            # Catch-all for any unexpected errors in sandbox setup
            elapsed_ms = int((time.time() - start_time) * 1000)
            return SandboxResult(
                success=False,
                error=f"Sandbox error: {type(e).__name__}: {e}",
                violation=SandboxViolation.EXCEPTION,
                elapsed_ms=elapsed_ms,
                exit_code=1,
            )

        finally:
            # Ensure process is cleaned up
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()

            # Close the queue
            try:
                result_queue.close()
            except Exception:
                pass
