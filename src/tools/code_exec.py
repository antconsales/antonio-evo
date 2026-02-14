"""
Code execution tool — run Python/shell in sandboxed subprocess (v8.5).

Security (v8.5):
- Pre-execution blocked command check
- Blocked read patterns (no .env, .ssh, credentials access)
- Restricted environment variables (API keys stripped)
- Dedicated workspace directory
- Windows process creation flags (CREATE_NO_WINDOW + BELOW_NORMAL_PRIORITY)
- Configurable via config/sandbox.json
"""

import json
import logging
import os
import re
import sys
import subprocess
import tempfile
from pathlib import Path
from .registry import ToolResult

logger = logging.getLogger(__name__)

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

# Default blocked commands (overridden by config/sandbox.json)
_DEFAULT_BLOCKED_COMMANDS = [
    "rm -rf /", "rm -rf ~", "rm -rf .", "format", "shutdown", "reboot",
    "net user", "net localgroup", "reg delete", "reg add",
    "diskpart", "bcdedit", "schtasks /delete",
]

# Default blocked read patterns
_DEFAULT_BLOCKED_READ_PATTERNS = [
    r"\.env", r"\.ssh", r"credentials", r"password", r"token", r"api_key",
    r"secret", r"\.aws", r"\.azure", r"\.gcp",
]

# Default blocked code patterns
_DEFAULT_BLOCKED_CODE_PATTERNS = [
    r"os\.remove.*\.env", r"shutil\.rmtree", r"subprocess\..*shell.*True",
    r"__import__\(.*subprocess", r"exec\(.*compile",
]

# Environment variables to strip from child processes
_SENSITIVE_ENV_PREFIXES = [
    "API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL",
    "OPENAI_", "ANTHROPIC_", "TAVILY_", "MISTRAL_", "AWS_",
    "AZURE_", "GCP_", "GITHUB_TOKEN", "DISCORD_", "TELEGRAM_",
]


def _load_sandbox_config() -> dict:
    """Load sandbox configuration from config/sandbox.json."""
    try:
        with open("config/sandbox.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _check_code_safety(code: str, language: str, config: dict) -> tuple:
    """
    Pre-execution safety check (v8.5).

    Returns:
        (safe: bool, reason: str)
    """
    code_lower = code.lower()

    # Check blocked commands
    blocked_cmds = config.get("blocked_commands", _DEFAULT_BLOCKED_COMMANDS)
    for cmd in blocked_cmds:
        if cmd.lower() in code_lower:
            return False, f"Blocked command detected: '{cmd}'"

    # Check blocked read patterns (accessing sensitive files)
    blocked_reads = config.get("blocked_read_patterns", _DEFAULT_BLOCKED_READ_PATTERNS)
    for pattern in blocked_reads:
        try:
            if re.search(pattern, code, re.IGNORECASE):
                # Allow if it's just a string literal or comment about patterns
                # Block if it's actual file access
                access_patterns = [
                    rf"open\(.*{pattern}", rf"read.*{pattern}", rf"Path\(.*{pattern}",
                    rf"os\.path.*{pattern}", rf"cat\s+.*{pattern}",
                    rf"type\s+.*{pattern}", rf"Get-Content.*{pattern}",
                ]
                for ap in access_patterns:
                    if re.search(ap, code, re.IGNORECASE):
                        return False, f"Blocked: accessing sensitive file matching '{pattern}'"
        except re.error:
            continue

    # Check blocked code patterns
    blocked_code = config.get("blocked_code_patterns", _DEFAULT_BLOCKED_CODE_PATTERNS)
    for pattern in blocked_code:
        try:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Blocked code pattern detected: '{pattern}'"
        except re.error:
            continue

    return True, ""


def _create_sandbox_env(config: dict) -> dict:
    """
    Create a restricted environment for subprocess execution (v8.5).

    Strips API keys, tokens, and other sensitive variables.
    """
    env = os.environ.copy()

    # Strip sensitive environment variables
    keys_to_remove = []
    for key in env:
        key_upper = key.upper()
        for prefix in _SENSITIVE_ENV_PREFIXES:
            if key_upper.startswith(prefix) or key_upper == prefix:
                keys_to_remove.append(key)
                break

    for key in keys_to_remove:
        del env[key]

    return env


def _get_workspace_dir(config: dict) -> str:
    """Get or create the sandbox workspace directory."""
    workspace = config.get("workspace_dir", "~/Documents/antonio-workspace")
    workspace_path = Path(os.path.expanduser(workspace))
    workspace_path.mkdir(parents=True, exist_ok=True)
    return str(workspace_path)


def _get_creation_flags() -> int:
    """Get Windows process creation flags for sandboxed execution."""
    flags = 0
    if sys.platform == "win32":
        CREATE_NO_WINDOW = 0x08000000
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        flags = CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
    return flags


def create_handler(timeout: int = DEFAULT_TIMEOUT):
    """Create code execution tool handler with sandbox hardening."""

    sandbox_config = _load_sandbox_config()
    effective_timeout = sandbox_config.get("timeout_secs", timeout)

    def execute_code(code: str, language: str = "python") -> ToolResult:
        # Pre-execution safety check (v8.5)
        safe, reason = _check_code_safety(code, language, sandbox_config)
        if not safe:
            logger.warning(f"Code execution blocked: {reason}")
            return ToolResult(
                success=False,
                output=f"Execution blocked by sandbox policy: {reason}",
                metadata={"blocked_by": "sandbox", "reason": reason},
            )

        sandbox_env = _create_sandbox_env(sandbox_config)
        workspace = _get_workspace_dir(sandbox_config)
        creation_flags = _get_creation_flags()

        try:
            if language == "python":
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False,
                    encoding="utf-8", dir=workspace,
                ) as f:
                    f.write(code)
                    temp_path = f.name
                try:
                    result = subprocess.run(
                        [sys.executable, temp_path],
                        capture_output=True,
                        timeout=effective_timeout,
                        cwd=workspace,
                        env=sandbox_env,
                        creationflags=creation_flags,
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
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", code],
                    capture_output=True,
                    timeout=effective_timeout,
                    cwd=workspace,
                    env=sandbox_env,
                    creationflags=creation_flags,
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
                # Split command into list to avoid shell=True (v8.5)
                import shlex
                try:
                    cmd_parts = shlex.split(code)
                except ValueError:
                    cmd_parts = code.split()

                result = subprocess.run(
                    cmd_parts,
                    capture_output=True,
                    timeout=effective_timeout,
                    cwd=workspace,
                    env=sandbox_env,
                    creationflags=creation_flags,
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
            return ToolResult(success=False, output=f"Execution timed out after {effective_timeout}s")
        except Exception as e:
            return ToolResult(success=False, output=f"Execution error: {e}")

    return execute_code
