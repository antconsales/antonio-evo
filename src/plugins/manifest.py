"""
Plugin Manifest - Declarative plugin metadata structure.

Plugins are defined entirely through manifest files (JSON).
This module provides the data structures for parsing and
validating plugin manifests WITHOUT executing any plugin code.

Design principles:
- Declarative only (metadata, capabilities)
- Non-executable
- Non-dynamic
- Structured validation errors
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# Constants
# =============================================================================

# Valid plugin ID pattern: lowercase alphanumeric with hyphens
PLUGIN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$")

# Version pattern: semver-like (major.minor.patch)
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")

# Maximum lengths
MAX_PLUGIN_ID_LENGTH = 64
MAX_NAME_LENGTH = 128
MAX_DESCRIPTION_LENGTH = 1024
MAX_CAPABILITIES = 32
MAX_ENTRYPOINTS = 16


# =============================================================================
# Error Types
# =============================================================================

class ManifestErrorType(Enum):
    """Types of manifest validation errors."""
    INVALID_JSON = "invalid_json"
    MISSING_FIELD = "missing_field"
    INVALID_TYPE = "invalid_type"
    INVALID_FORMAT = "invalid_format"
    INVALID_LENGTH = "invalid_length"
    INVALID_VALUE = "invalid_value"
    TOO_MANY_ITEMS = "too_many_items"


@dataclass
class ManifestError:
    """
    Structured manifest validation error.

    Attributes:
        error_type: Type of error
        field: Field that caused the error (if applicable)
        message: Human-readable error message
        details: Additional error details
    """
    error_type: ManifestErrorType
    field: Optional[str] = None
    message: str = ""
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.error_type.value,
            "field": self.field,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PluginCapability:
    """
    A declared plugin capability.

    Capabilities are declarative strings that describe what
    a plugin can do. They do NOT grant any actual permissions
    or execute any code.

    Attributes:
        name: Capability identifier (e.g., "text_processing")
        description: Optional description of the capability
    """
    name: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginCapability":
        """Create from dictionary."""
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
        )

    @classmethod
    def from_string(cls, name: str) -> "PluginCapability":
        """Create from simple string."""
        return cls(name=name, description="")


@dataclass
class PluginEntrypoint:
    """
    A declared plugin entrypoint (DECLARATIVE ONLY).

    Entrypoints describe where plugin code would be located.
    They are NEVER imported, loaded, or executed.
    This is metadata only for documentation/planning purposes.

    Attributes:
        name: Entrypoint name (e.g., "main", "handler")
        module: Declared module path (NOT imported)
        function: Declared function name (NOT called)
        description: Optional description
    """
    name: str
    module: str = ""
    function: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "module": self.module,
            "function": self.function,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginEntrypoint":
        """Create from dictionary."""
        return cls(
            name=str(data.get("name", "")),
            module=str(data.get("module", "")),
            function=str(data.get("function", "")),
            description=str(data.get("description", "")),
        )


@dataclass
class PluginManifest:
    """
    Complete plugin manifest structure.

    This is a declarative-only data structure that describes
    a plugin's metadata, capabilities, and entrypoints.
    No plugin code is ever executed based on this manifest.

    Attributes:
        plugin_id: Unique plugin identifier
        name: Human-readable plugin name
        version: Plugin version (semver format)
        description: Plugin description
        capabilities: List of declared capabilities
        entrypoints: List of declared entrypoints (metadata only)
        author: Optional author information
        license: Optional license identifier
        metadata: Optional additional metadata
    """
    plugin_id: str
    name: str
    version: str
    description: str = ""
    capabilities: List[PluginCapability] = field(default_factory=list)
    entrypoints: List[PluginEntrypoint] = field(default_factory=list)
    author: str = ""
    license: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "entrypoints": [e.to_dict() for e in self.entrypoints],
            "author": self.author,
            "license": self.license,
            "metadata": self.metadata,
        }

    def has_capability(self, capability_name: str) -> bool:
        """Check if plugin declares a capability."""
        return any(c.name == capability_name for c in self.capabilities)

    def get_entrypoint(self, name: str) -> Optional[PluginEntrypoint]:
        """Get entrypoint by name (metadata only, never executed)."""
        for ep in self.entrypoints:
            if ep.name == name:
                return ep
        return None


# =============================================================================
# Validation
# =============================================================================

def validate_manifest_data(data: Dict[str, Any]) -> List[ManifestError]:
    """
    Validate manifest data structure.

    Args:
        data: Parsed manifest data

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Required fields
    required_fields = ["plugin_id", "name", "version"]
    for field_name in required_fields:
        if field_name not in data:
            errors.append(ManifestError(
                error_type=ManifestErrorType.MISSING_FIELD,
                field=field_name,
                message=f"Missing required field: {field_name}",
            ))

    # If missing required fields, return early
    if errors:
        return errors

    # Validate plugin_id
    plugin_id = data.get("plugin_id", "")
    if not isinstance(plugin_id, str):
        errors.append(ManifestError(
            error_type=ManifestErrorType.INVALID_TYPE,
            field="plugin_id",
            message="plugin_id must be a string",
        ))
    elif len(plugin_id) < 2:
        errors.append(ManifestError(
            error_type=ManifestErrorType.INVALID_LENGTH,
            field="plugin_id",
            message="plugin_id must be at least 2 characters",
        ))
    elif len(plugin_id) > MAX_PLUGIN_ID_LENGTH:
        errors.append(ManifestError(
            error_type=ManifestErrorType.INVALID_LENGTH,
            field="plugin_id",
            message=f"plugin_id exceeds maximum length of {MAX_PLUGIN_ID_LENGTH}",
        ))
    elif not PLUGIN_ID_PATTERN.match(plugin_id):
        errors.append(ManifestError(
            error_type=ManifestErrorType.INVALID_FORMAT,
            field="plugin_id",
            message="plugin_id must be lowercase alphanumeric with hyphens",
        ))

    # Validate name
    name = data.get("name", "")
    if not isinstance(name, str):
        errors.append(ManifestError(
            error_type=ManifestErrorType.INVALID_TYPE,
            field="name",
            message="name must be a string",
        ))
    elif len(name) == 0:
        errors.append(ManifestError(
            error_type=ManifestErrorType.INVALID_LENGTH,
            field="name",
            message="name cannot be empty",
        ))
    elif len(name) > MAX_NAME_LENGTH:
        errors.append(ManifestError(
            error_type=ManifestErrorType.INVALID_LENGTH,
            field="name",
            message=f"name exceeds maximum length of {MAX_NAME_LENGTH}",
        ))

    # Validate version
    version = data.get("version", "")
    if not isinstance(version, str):
        errors.append(ManifestError(
            error_type=ManifestErrorType.INVALID_TYPE,
            field="version",
            message="version must be a string",
        ))
    elif not VERSION_PATTERN.match(version):
        errors.append(ManifestError(
            error_type=ManifestErrorType.INVALID_FORMAT,
            field="version",
            message="version must be in format X.Y.Z (e.g., 1.0.0)",
        ))

    # Validate description (optional)
    description = data.get("description", "")
    if description and not isinstance(description, str):
        errors.append(ManifestError(
            error_type=ManifestErrorType.INVALID_TYPE,
            field="description",
            message="description must be a string",
        ))
    elif isinstance(description, str) and len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(ManifestError(
            error_type=ManifestErrorType.INVALID_LENGTH,
            field="description",
            message=f"description exceeds maximum length of {MAX_DESCRIPTION_LENGTH}",
        ))

    # Validate capabilities (optional)
    capabilities = data.get("capabilities", [])
    if not isinstance(capabilities, list):
        errors.append(ManifestError(
            error_type=ManifestErrorType.INVALID_TYPE,
            field="capabilities",
            message="capabilities must be a list",
        ))
    elif len(capabilities) > MAX_CAPABILITIES:
        errors.append(ManifestError(
            error_type=ManifestErrorType.TOO_MANY_ITEMS,
            field="capabilities",
            message=f"capabilities exceeds maximum of {MAX_CAPABILITIES}",
        ))
    else:
        for i, cap in enumerate(capabilities):
            if isinstance(cap, str):
                if not cap:
                    errors.append(ManifestError(
                        error_type=ManifestErrorType.INVALID_VALUE,
                        field=f"capabilities[{i}]",
                        message="capability name cannot be empty",
                    ))
            elif isinstance(cap, dict):
                if "name" not in cap:
                    errors.append(ManifestError(
                        error_type=ManifestErrorType.MISSING_FIELD,
                        field=f"capabilities[{i}].name",
                        message="capability must have a name",
                    ))
            else:
                errors.append(ManifestError(
                    error_type=ManifestErrorType.INVALID_TYPE,
                    field=f"capabilities[{i}]",
                    message="capability must be a string or object",
                ))

    # Validate entrypoints (optional)
    entrypoints = data.get("entrypoints", [])
    if not isinstance(entrypoints, list):
        errors.append(ManifestError(
            error_type=ManifestErrorType.INVALID_TYPE,
            field="entrypoints",
            message="entrypoints must be a list",
        ))
    elif len(entrypoints) > MAX_ENTRYPOINTS:
        errors.append(ManifestError(
            error_type=ManifestErrorType.TOO_MANY_ITEMS,
            field="entrypoints",
            message=f"entrypoints exceeds maximum of {MAX_ENTRYPOINTS}",
        ))
    else:
        for i, ep in enumerate(entrypoints):
            if not isinstance(ep, dict):
                errors.append(ManifestError(
                    error_type=ManifestErrorType.INVALID_TYPE,
                    field=f"entrypoints[{i}]",
                    message="entrypoint must be an object",
                ))
            elif "name" not in ep:
                errors.append(ManifestError(
                    error_type=ManifestErrorType.MISSING_FIELD,
                    field=f"entrypoints[{i}].name",
                    message="entrypoint must have a name",
                ))

    return errors


def parse_manifest(json_string: str) -> tuple[Optional[PluginManifest], List[ManifestError]]:
    """
    Parse and validate a manifest from JSON string.

    Args:
        json_string: JSON string containing manifest data

    Returns:
        Tuple of (manifest, errors). If errors is non-empty, manifest is None.
    """
    # Parse JSON
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        return None, [ManifestError(
            error_type=ManifestErrorType.INVALID_JSON,
            message=f"Invalid JSON: {e}",
        )]

    if not isinstance(data, dict):
        return None, [ManifestError(
            error_type=ManifestErrorType.INVALID_TYPE,
            message="Manifest must be a JSON object",
        )]

    # Validate structure
    errors = validate_manifest_data(data)
    if errors:
        return None, errors

    # Build manifest
    return manifest_from_dict(data), []


def manifest_from_dict(data: Dict[str, Any]) -> PluginManifest:
    """
    Create PluginManifest from validated dictionary.

    Args:
        data: Validated manifest data

    Returns:
        PluginManifest instance
    """
    # Parse capabilities
    capabilities = []
    for cap in data.get("capabilities", []):
        if isinstance(cap, str):
            capabilities.append(PluginCapability.from_string(cap))
        elif isinstance(cap, dict):
            capabilities.append(PluginCapability.from_dict(cap))

    # Parse entrypoints
    entrypoints = []
    for ep in data.get("entrypoints", []):
        if isinstance(ep, dict):
            entrypoints.append(PluginEntrypoint.from_dict(ep))

    return PluginManifest(
        plugin_id=data["plugin_id"],
        name=data["name"],
        version=data["version"],
        description=data.get("description", ""),
        capabilities=capabilities,
        entrypoints=entrypoints,
        author=data.get("author", ""),
        license=data.get("license", ""),
        metadata=data.get("metadata", {}),
    )
