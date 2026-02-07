"""
Plugin Registry - Safe, read-only plugin manifest registry.

The registry loads and validates plugin manifests from disk,
checks them against a whitelist, and provides read-only access.
No plugin code is ever executed, imported, or loaded.

Design principles:
- Never execute plugin code
- Never import plugin modules
- Never load arbitrary Python
- Whitelist-based validation
- Read-only access
- Structured error handling
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .manifest import (
    PluginManifest,
    ManifestError,
    parse_manifest,
    validate_manifest_data,
    manifest_from_dict,
)


# =============================================================================
# Error Types
# =============================================================================

class RegistryErrorType(Enum):
    """Types of registry errors."""
    FILE_NOT_FOUND = "file_not_found"
    FILE_READ_ERROR = "file_read_error"
    INVALID_JSON = "invalid_json"
    INVALID_WHITELIST = "invalid_whitelist"
    MANIFEST_INVALID = "manifest_invalid"
    NOT_WHITELISTED = "not_whitelisted"
    DUPLICATE_PLUGIN = "duplicate_plugin"
    PLUGIN_NOT_FOUND = "plugin_not_found"


@dataclass
class RegistryError:
    """
    Structured registry error.

    Attributes:
        error_type: Type of error
        plugin_id: Plugin that caused the error (if applicable)
        message: Human-readable error message
        details: Additional error details
        manifest_errors: Nested manifest errors (if applicable)
    """
    error_type: RegistryErrorType
    plugin_id: Optional[str] = None
    message: str = ""
    details: Optional[Dict[str, Any]] = None
    manifest_errors: List[ManifestError] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.error_type.value,
            "plugin_id": self.plugin_id,
            "message": self.message,
            "details": self.details,
            "manifest_errors": [e.to_dict() for e in self.manifest_errors],
        }


class PluginStatus(Enum):
    """Plugin status in the registry."""
    VALID = "valid"
    INVALID_MANIFEST = "invalid_manifest"
    NOT_WHITELISTED = "not_whitelisted"
    DUPLICATE = "duplicate"


@dataclass
class PluginEntry:
    """
    A plugin entry in the registry.

    Attributes:
        manifest: The plugin manifest (if valid)
        status: Plugin status
        errors: Any errors encountered
        source_path: Path where manifest was loaded from
    """
    manifest: Optional[PluginManifest]
    status: PluginStatus
    errors: List[RegistryError] = field(default_factory=list)
    source_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "status": self.status.value,
            "errors": [e.to_dict() for e in self.errors],
            "source_path": self.source_path,
        }


# =============================================================================
# Whitelist
# =============================================================================

@dataclass
class PluginWhitelist:
    """
    Plugin whitelist configuration.

    Only plugins listed in the whitelist can be considered valid.
    This is a security measure to prevent arbitrary plugin loading.

    Attributes:
        allowed_plugins: Set of allowed plugin IDs
        enabled: Whether whitelist enforcement is enabled
    """
    allowed_plugins: Set[str] = field(default_factory=set)
    enabled: bool = True

    def is_allowed(self, plugin_id: str) -> bool:
        """Check if a plugin is whitelisted."""
        if not self.enabled:
            return True
        return plugin_id in self.allowed_plugins

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "allowed_plugins": list(self.allowed_plugins),
            "enabled": self.enabled,
        }


def load_whitelist(path: str) -> tuple[Optional[PluginWhitelist], Optional[RegistryError]]:
    """
    Load whitelist from JSON file.

    Args:
        path: Path to whitelist JSON file

    Returns:
        Tuple of (whitelist, error). If error is set, whitelist is None.
    """
    try:
        if not os.path.exists(path):
            return None, RegistryError(
                error_type=RegistryErrorType.FILE_NOT_FOUND,
                message=f"Whitelist file not found: {path}",
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return None, RegistryError(
                error_type=RegistryErrorType.INVALID_WHITELIST,
                message="Whitelist must be a JSON object",
            )

        allowed = data.get("allowed_plugins", [])
        if not isinstance(allowed, list):
            return None, RegistryError(
                error_type=RegistryErrorType.INVALID_WHITELIST,
                message="allowed_plugins must be a list",
            )

        # Validate all items are strings
        for i, item in enumerate(allowed):
            if not isinstance(item, str):
                return None, RegistryError(
                    error_type=RegistryErrorType.INVALID_WHITELIST,
                    message=f"allowed_plugins[{i}] must be a string",
                )

        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            return None, RegistryError(
                error_type=RegistryErrorType.INVALID_WHITELIST,
                message="enabled must be a boolean",
            )

        return PluginWhitelist(
            allowed_plugins=set(allowed),
            enabled=enabled,
        ), None

    except json.JSONDecodeError as e:
        return None, RegistryError(
            error_type=RegistryErrorType.INVALID_JSON,
            message=f"Invalid whitelist JSON: {e}",
        )
    except IOError as e:
        return None, RegistryError(
            error_type=RegistryErrorType.FILE_READ_ERROR,
            message=f"Failed to read whitelist: {e}",
        )


# =============================================================================
# Registry
# =============================================================================

class PluginRegistry:
    """
    Read-only plugin manifest registry.

    Loads plugin manifests from disk, validates them against a whitelist,
    and provides read-only access. No plugin code is ever executed.

    Usage:
        registry = PluginRegistry(whitelist_path="config/plugin_whitelist.json")
        registry.load_from_directory("plugins/")

        # Get a valid plugin
        manifest = registry.get("my-plugin")

        # List all valid plugins
        for manifest in registry.list_valid():
            print(manifest.name)

    Security guarantees:
        - Never executes plugin code
        - Never imports plugin modules
        - Never loads arbitrary Python
        - Only whitelisted plugins are considered valid
    """

    def __init__(self, whitelist_path: Optional[str] = None):
        """
        Initialize plugin registry.

        Args:
            whitelist_path: Path to whitelist JSON file (optional)
        """
        self._plugins: Dict[str, PluginEntry] = {}
        self._whitelist: PluginWhitelist = PluginWhitelist()
        self._whitelist_path = whitelist_path
        self._initialization_errors: List[RegistryError] = []

        # Load whitelist if path provided
        if whitelist_path:
            self._load_whitelist(whitelist_path)

    def _load_whitelist(self, path: str) -> None:
        """Load whitelist from file."""
        whitelist, error = load_whitelist(path)
        if error:
            self._initialization_errors.append(error)
            # Use empty whitelist (blocks all plugins)
            self._whitelist = PluginWhitelist(enabled=True)
        else:
            self._whitelist = whitelist

    def load_manifest(self, path: str) -> Optional[RegistryError]:
        """
        Load a single plugin manifest from file.

        Args:
            path: Path to manifest JSON file

        Returns:
            RegistryError if loading failed, None on success
        """
        try:
            if not os.path.exists(path):
                return RegistryError(
                    error_type=RegistryErrorType.FILE_NOT_FOUND,
                    message=f"Manifest file not found: {path}",
                )

            with open(path, "r", encoding="utf-8") as f:
                json_string = f.read()

            manifest, errors = parse_manifest(json_string)

            if errors:
                # Store as invalid entry if we can extract plugin_id
                try:
                    data = json.loads(json_string)
                    plugin_id = data.get("plugin_id", "")
                except Exception:
                    plugin_id = ""

                error = RegistryError(
                    error_type=RegistryErrorType.MANIFEST_INVALID,
                    plugin_id=plugin_id if plugin_id else None,
                    message="Manifest validation failed",
                    manifest_errors=errors,
                )

                if plugin_id:
                    self._plugins[plugin_id] = PluginEntry(
                        manifest=None,
                        status=PluginStatus.INVALID_MANIFEST,
                        errors=[error],
                        source_path=path,
                    )

                return error

            # Check for duplicate
            if manifest.plugin_id in self._plugins:
                existing = self._plugins[manifest.plugin_id]
                if existing.status == PluginStatus.VALID:
                    return RegistryError(
                        error_type=RegistryErrorType.DUPLICATE_PLUGIN,
                        plugin_id=manifest.plugin_id,
                        message=f"Duplicate plugin_id: {manifest.plugin_id}",
                        details={"existing_path": existing.source_path},
                    )

            # Check whitelist
            if not self._whitelist.is_allowed(manifest.plugin_id):
                self._plugins[manifest.plugin_id] = PluginEntry(
                    manifest=manifest,
                    status=PluginStatus.NOT_WHITELISTED,
                    errors=[RegistryError(
                        error_type=RegistryErrorType.NOT_WHITELISTED,
                        plugin_id=manifest.plugin_id,
                        message=f"Plugin not in whitelist: {manifest.plugin_id}",
                    )],
                    source_path=path,
                )
                return RegistryError(
                    error_type=RegistryErrorType.NOT_WHITELISTED,
                    plugin_id=manifest.plugin_id,
                    message=f"Plugin not in whitelist: {manifest.plugin_id}",
                )

            # Valid plugin
            self._plugins[manifest.plugin_id] = PluginEntry(
                manifest=manifest,
                status=PluginStatus.VALID,
                errors=[],
                source_path=path,
            )
            return None

        except IOError as e:
            return RegistryError(
                error_type=RegistryErrorType.FILE_READ_ERROR,
                message=f"Failed to read manifest: {e}",
            )

    def load_from_directory(self, directory: str) -> List[RegistryError]:
        """
        Load all plugin manifests from a directory.

        Looks for files matching pattern: */manifest.json or *.json

        Args:
            directory: Path to plugins directory

        Returns:
            List of errors encountered during loading
        """
        errors = []

        try:
            if not os.path.exists(directory):
                errors.append(RegistryError(
                    error_type=RegistryErrorType.FILE_NOT_FOUND,
                    message=f"Plugin directory not found: {directory}",
                ))
                return errors

            if not os.path.isdir(directory):
                errors.append(RegistryError(
                    error_type=RegistryErrorType.FILE_READ_ERROR,
                    message=f"Not a directory: {directory}",
                ))
                return errors

            # Look for manifest files
            for entry in os.listdir(directory):
                entry_path = os.path.join(directory, entry)

                # Check for plugin subdirectory with manifest.json
                if os.path.isdir(entry_path):
                    manifest_path = os.path.join(entry_path, "manifest.json")
                    if os.path.exists(manifest_path):
                        error = self.load_manifest(manifest_path)
                        if error:
                            errors.append(error)

                # Check for standalone manifest files
                elif entry.endswith(".json") and entry != "manifest.json":
                    error = self.load_manifest(entry_path)
                    if error:
                        errors.append(error)

        except IOError as e:
            errors.append(RegistryError(
                error_type=RegistryErrorType.FILE_READ_ERROR,
                message=f"Failed to read directory: {e}",
            ))

        return errors

    def get(self, plugin_id: str) -> Optional[PluginManifest]:
        """
        Get a valid plugin manifest by ID.

        Only returns manifests for valid, whitelisted plugins.

        Args:
            plugin_id: Plugin identifier

        Returns:
            PluginManifest if found and valid, None otherwise
        """
        entry = self._plugins.get(plugin_id)
        if entry and entry.status == PluginStatus.VALID:
            return entry.manifest
        return None

    def get_entry(self, plugin_id: str) -> Optional[PluginEntry]:
        """
        Get a plugin entry by ID (includes status and errors).

        Args:
            plugin_id: Plugin identifier

        Returns:
            PluginEntry if found, None otherwise
        """
        return self._plugins.get(plugin_id)

    def list_valid(self) -> List[PluginManifest]:
        """
        List all valid plugin manifests.

        Returns:
            List of valid PluginManifest instances
        """
        return [
            entry.manifest
            for entry in self._plugins.values()
            if entry.status == PluginStatus.VALID and entry.manifest
        ]

    def list_all(self) -> List[PluginEntry]:
        """
        List all plugin entries (including invalid).

        Returns:
            List of all PluginEntry instances
        """
        return list(self._plugins.values())

    def list_by_status(self, status: PluginStatus) -> List[PluginEntry]:
        """
        List plugin entries by status.

        Args:
            status: Plugin status to filter by

        Returns:
            List of matching PluginEntry instances
        """
        return [
            entry for entry in self._plugins.values()
            if entry.status == status
        ]

    def has_plugin(self, plugin_id: str) -> bool:
        """Check if a plugin is registered (any status)."""
        return plugin_id in self._plugins

    def is_valid(self, plugin_id: str) -> bool:
        """Check if a plugin is valid and whitelisted."""
        entry = self._plugins.get(plugin_id)
        return entry is not None and entry.status == PluginStatus.VALID

    def count(self) -> int:
        """Get total number of registered plugins."""
        return len(self._plugins)

    def count_valid(self) -> int:
        """Get number of valid plugins."""
        return sum(1 for e in self._plugins.values() if e.status == PluginStatus.VALID)

    @property
    def whitelist(self) -> PluginWhitelist:
        """Get the whitelist (read-only)."""
        return self._whitelist

    @property
    def initialization_errors(self) -> List[RegistryError]:
        """Get errors encountered during initialization."""
        return self._initialization_errors.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry state to dictionary."""
        return {
            "plugins": {
                plugin_id: entry.to_dict()
                for plugin_id, entry in self._plugins.items()
            },
            "whitelist": self._whitelist.to_dict(),
            "counts": {
                "total": self.count(),
                "valid": self.count_valid(),
            },
        }
