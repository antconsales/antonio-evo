"""Code execution tool - run Python/shell in sandboxed subprocess."""

import os
import sys
import subprocess
import tempfile
from .registry import ToolResult

DEFAULT_TIMEOUT = 30

DEFINITION = {
    "name": "execute_code",
    "description": (
        "Execute Python code or shell/PowerShell commands and return the output. "
        "Use this for calculations, data processing, system commands, installing packages, "
        "or any task that benefits from running code."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The code to execute",
            },
            "language": {
                "type": "string",
                "enum": ["python", "shell", "powershell"],
                "description": "Programming language (default: python)",
            },
        },
        "required": ["code"],
    },
}


def create_handler(timeout: int = DEFAULT_TIMEOUT):
    """Create code execution tool handler."""

    def execute_code(code: str, language: str = "python") -> ToolResult:
        try:
            if language == "python":
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False, encoding="utf-8"
                ) as f:
                    f.write(code)
                    temp_path = f.name
                try:
                    result = subprocess.run(
                        [sys.executable, temp_path],
                        capture_output=True,
                        timeout=timeout,
                        cwd=os.path.expanduser("~"),
                    )
                    stdout = result.stdout.decode("utf-8", errors="replace")
                    stderr = result.stderr.decode("utf-8", errors="replace")
                    output = ""
                    if stdout:
                        output += stdout
                    if stderr:
                        output += f"\n[STDERR]\n{stderr}"
                    if result.returncode != 0:
                        output += f"\n[Exit code: {result.returncode}]"
                    return ToolResult(
                        success=result.returncode == 0,
                        output=output.strip() or "(no output)",
                    )
                finally:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

            elif language == "powershell":
                result = subprocess.run(
                    ["powershell", "-Command", code],
                    capture_output=True,
                    timeout=timeout,
                    cwd=os.path.expanduser("~"),
                )
                stdout = result.stdout.decode("utf-8", errors="replace")
                stderr = result.stderr.decode("utf-8", errors="replace")
                output = ""
                if stdout:
                    output += stdout
                if stderr:
                    output += f"\n[STDERR]\n{stderr}"
                return ToolResult(
                    success=result.returncode == 0,
                    output=output.strip() or "(no output)",
                )

            elif language == "shell":
                result = subprocess.run(
                    code,
                    capture_output=True,
                    timeout=timeout,
                    shell=True,
                    cwd=os.path.expanduser("~"),
                )
                stdout = result.stdout.decode("utf-8", errors="replace")
                stderr = result.stderr.decode("utf-8", errors="replace")
                output = ""
                if stdout:
                    output += stdout
                if stderr:
                    output += f"\n[STDERR]\n{stderr}"
                return ToolResult(
                    success=result.returncode == 0,
                    output=output.strip() or "(no output)",
                )

            else:
                return ToolResult(
                    success=False,
                    output=f"Unsupported language: {language}. Use 'python', 'shell', or 'powershell'.",
                )

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=f"Execution timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output=f"Execution error: {e}")

    return execute_code
