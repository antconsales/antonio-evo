"""File operations tools - read, write, list with security sandboxing."""

import os
from pathlib import Path
from typing import Dict, List
from .registry import ToolResult

BLOCKED_PATTERNS = [
    ".env", "credentials", "secret", "password", "token",
    ".ssh", ".gnupg", ".aws", "api_key",
]

DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file and return its lines with line numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path to read"},
                "offset": {"type": "integer", "description": "Line to start from (0-based, default 0)"},
                "limit": {"type": "integer", "description": "Max lines to read (default 200)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file and parent directories if they don't exist, or overwrites if the file exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and subdirectories at a given path, showing type and size.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute directory path to list"},
            },
            "required": ["path"],
        },
    },
]


def _is_path_allowed(path: str, allowed_dirs: List[str]) -> bool:
    """Check if path is within allowed directories and not blocked."""
    try:
        resolved = os.path.realpath(os.path.abspath(path))
        for allowed in allowed_dirs:
            allowed_resolved = os.path.realpath(os.path.abspath(os.path.expanduser(allowed)))
            if resolved.startswith(allowed_resolved):
                lower = resolved.lower()
                for pattern in BLOCKED_PATTERNS:
                    if pattern in lower:
                        return False
                return True
        return False
    except Exception:
        return False


def create_handlers(allowed_dirs: List[str] = None) -> Dict:
    """Create file operation handlers with sandboxing."""
    dirs = allowed_dirs or [os.path.expanduser("~")]

    def read_file(path: str, offset: int = 0, limit: int = 200) -> ToolResult:
        if not _is_path_allowed(path, dirs):
            return ToolResult(success=False, output=f"Access denied: {path}")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            total = len(lines)
            selected = lines[offset:offset + limit]
            numbered = [f"{i + offset + 1}: {line.rstrip()}" for i, line in enumerate(selected)]
            header = f"[{path}] ({total} lines total, showing {offset + 1}-{offset + len(selected)})\n"
            return ToolResult(success=True, output=header + "\n".join(numbered))
        except FileNotFoundError:
            return ToolResult(success=False, output=f"File not found: {path}")
        except Exception as e:
            return ToolResult(success=False, output=f"Error reading {path}: {e}")

    def write_file(path: str, content: str) -> ToolResult:
        if not _is_path_allowed(path, dirs):
            return ToolResult(success=False, output=f"Access denied: {path}")
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            lines = content.count("\n") + 1
            return ToolResult(success=True, output=f"Written {len(content)} chars ({lines} lines) to {path}")
        except Exception as e:
            return ToolResult(success=False, output=f"Error writing {path}: {e}")

    def list_directory(path: str) -> ToolResult:
        if not _is_path_allowed(path, dirs):
            return ToolResult(success=False, output=f"Access denied: {path}")
        try:
            entries = sorted(os.listdir(path))
            lines = []
            for entry in entries[:100]:
                full = os.path.join(path, entry)
                if os.path.isdir(full):
                    lines.append(f"  [DIR]  {entry}/")
                else:
                    try:
                        size = os.path.getsize(full)
                        if size < 1024:
                            size_str = f"{size} B"
                        elif size < 1024 * 1024:
                            size_str = f"{size / 1024:.1f} KB"
                        else:
                            size_str = f"{size / (1024 * 1024):.1f} MB"
                    except OSError:
                        size_str = "?"
                    lines.append(f"  [FILE] {entry} ({size_str})")
            header = f"[{path}] ({len(entries)} entries"
            if len(entries) > 100:
                header += f", showing first 100"
            header += ")\n"
            return ToolResult(success=True, output=header + "\n".join(lines))
        except FileNotFoundError:
            return ToolResult(success=False, output=f"Directory not found: {path}")
        except Exception as e:
            return ToolResult(success=False, output=f"Error listing {path}: {e}")

    return {
        "read_file": read_file,
        "write_file": write_file,
        "list_directory": list_directory,
    }
