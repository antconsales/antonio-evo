"""
Plugin module for declarative plugin structure.

This module provides a safe, non-executable plugin system.
Plugins are defined declaratively via manifest files and
validated against a whitelist. No plugin code is ever executed.

Design principles:
- Declarative only (no code execution)
- Whitelist-based validation
- Read-only access to plugin metadata
- Structured error handling
"""

from .manifest import (
    PluginManifest,
    PluginCapability,
    PluginEntrypoint,
    ManifestError,
    ManifestErrorType,
)

from .registry import (
    PluginRegistry,
    RegistryError,
    RegistryErrorType,
    PluginStatus,
)

__all__ = [
    # Manifest
    "PluginManifest",
    "PluginCapability",
    "PluginEntrypoint",
    "ManifestError",
    "ManifestErrorType",
    # Registry
    "PluginRegistry",
    "RegistryError",
    "RegistryErrorType",
    "PluginStatus",
]
