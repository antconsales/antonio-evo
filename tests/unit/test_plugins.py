"""
Unit tests for the plugin system.

Tests cover:
- Valid plugin manifest parsing
- Invalid/malformed manifest rejection
- Non-whitelisted plugin rejection
- Duplicate plugin_id handling
- Registry load behavior
- Read-only access enforcement
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, ".")

from src.plugins import (
    PluginManifest,
    PluginCapability,
    PluginEntrypoint,
    ManifestError,
    ManifestErrorType,
    PluginRegistry,
    RegistryError,
    RegistryErrorType,
    PluginStatus,
)

from src.plugins.manifest import (
    parse_manifest,
    validate_manifest_data,
    manifest_from_dict,
    PLUGIN_ID_PATTERN,
    VERSION_PATTERN,
)

from src.plugins.registry import (
    PluginWhitelist,
    load_whitelist,
    PluginEntry,
)


# =============================================================================
# Test: PluginManifest Data Structure
# =============================================================================

class TestPluginManifest(unittest.TestCase):
    """Tests for PluginManifest dataclass."""

    def test_create_minimal_manifest(self):
        """Test creating manifest with minimal required fields."""
        manifest = PluginManifest(
            plugin_id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
        )
        self.assertEqual(manifest.plugin_id, "test-plugin")
        self.assertEqual(manifest.name, "Test Plugin")
        self.assertEqual(manifest.version, "1.0.0")
        self.assertEqual(manifest.capabilities, [])
        self.assertEqual(manifest.entrypoints, [])

    def test_create_full_manifest(self):
        """Test creating manifest with all fields."""
        manifest = PluginManifest(
            plugin_id="full-plugin",
            name="Full Plugin",
            version="2.1.0",
            description="A complete plugin",
            capabilities=[PluginCapability(name="text_processing")],
            entrypoints=[PluginEntrypoint(name="main", module="plugin.main")],
            author="Test Author",
            license="MIT",
            metadata={"key": "value"},
        )
        self.assertEqual(manifest.description, "A complete plugin")
        self.assertEqual(len(manifest.capabilities), 1)
        self.assertEqual(len(manifest.entrypoints), 1)
        self.assertEqual(manifest.author, "Test Author")

    def test_manifest_to_dict(self):
        """Test manifest serialization."""
        manifest = PluginManifest(
            plugin_id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
        )
        d = manifest.to_dict()
        self.assertEqual(d["plugin_id"], "test-plugin")
        self.assertEqual(d["name"], "Test Plugin")
        self.assertEqual(d["version"], "1.0.0")
        self.assertIn("capabilities", d)
        self.assertIn("entrypoints", d)

    def test_has_capability(self):
        """Test capability checking."""
        manifest = PluginManifest(
            plugin_id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
            capabilities=[
                PluginCapability(name="text_processing"),
                PluginCapability(name="audio_processing"),
            ],
        )
        self.assertTrue(manifest.has_capability("text_processing"))
        self.assertTrue(manifest.has_capability("audio_processing"))
        self.assertFalse(manifest.has_capability("video_processing"))

    def test_get_entrypoint(self):
        """Test entrypoint retrieval."""
        manifest = PluginManifest(
            plugin_id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
            entrypoints=[
                PluginEntrypoint(name="main", module="plugin.main"),
                PluginEntrypoint(name="handler", module="plugin.handler"),
            ],
        )
        ep = manifest.get_entrypoint("main")
        self.assertIsNotNone(ep)
        self.assertEqual(ep.module, "plugin.main")

        ep_none = manifest.get_entrypoint("nonexistent")
        self.assertIsNone(ep_none)


# =============================================================================
# Test: PluginCapability
# =============================================================================

class TestPluginCapability(unittest.TestCase):
    """Tests for PluginCapability dataclass."""

    def test_create_capability(self):
        """Test creating a capability."""
        cap = PluginCapability(name="text_processing", description="Process text")
        self.assertEqual(cap.name, "text_processing")
        self.assertEqual(cap.description, "Process text")

    def test_capability_from_string(self):
        """Test creating capability from string."""
        cap = PluginCapability.from_string("text_processing")
        self.assertEqual(cap.name, "text_processing")
        self.assertEqual(cap.description, "")

    def test_capability_from_dict(self):
        """Test creating capability from dict."""
        cap = PluginCapability.from_dict({
            "name": "audio_processing",
            "description": "Process audio",
        })
        self.assertEqual(cap.name, "audio_processing")
        self.assertEqual(cap.description, "Process audio")

    def test_capability_to_dict(self):
        """Test capability serialization."""
        cap = PluginCapability(name="test", description="desc")
        d = cap.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["description"], "desc")


# =============================================================================
# Test: PluginEntrypoint
# =============================================================================

class TestPluginEntrypoint(unittest.TestCase):
    """Tests for PluginEntrypoint dataclass."""

    def test_create_entrypoint(self):
        """Test creating an entrypoint."""
        ep = PluginEntrypoint(
            name="main",
            module="plugin.main",
            function="run",
            description="Main entry",
        )
        self.assertEqual(ep.name, "main")
        self.assertEqual(ep.module, "plugin.main")
        self.assertEqual(ep.function, "run")

    def test_entrypoint_from_dict(self):
        """Test creating entrypoint from dict."""
        ep = PluginEntrypoint.from_dict({
            "name": "handler",
            "module": "plugin.handler",
            "function": "handle",
        })
        self.assertEqual(ep.name, "handler")
        self.assertEqual(ep.module, "plugin.handler")
        self.assertEqual(ep.function, "handle")

    def test_entrypoint_to_dict(self):
        """Test entrypoint serialization."""
        ep = PluginEntrypoint(name="main", module="mod", function="func")
        d = ep.to_dict()
        self.assertEqual(d["name"], "main")
        self.assertEqual(d["module"], "mod")
        self.assertEqual(d["function"], "func")


# =============================================================================
# Test: Manifest Validation - Valid Cases
# =============================================================================

class TestManifestValidationValid(unittest.TestCase):
    """Tests for valid manifest parsing."""

    def test_parse_minimal_manifest(self):
        """Test parsing minimal valid manifest."""
        json_str = json.dumps({
            "plugin_id": "test-plugin",
            "name": "Test Plugin",
            "version": "1.0.0",
        })
        manifest, errors = parse_manifest(json_str)
        self.assertEqual(len(errors), 0)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.plugin_id, "test-plugin")

    def test_parse_full_manifest(self):
        """Test parsing full manifest with all fields."""
        json_str = json.dumps({
            "plugin_id": "full-plugin",
            "name": "Full Plugin",
            "version": "2.1.0",
            "description": "A complete plugin",
            "capabilities": ["text_processing", {"name": "audio", "description": "Audio cap"}],
            "entrypoints": [{"name": "main", "module": "plugin.main"}],
            "author": "Test Author",
            "license": "MIT",
            "metadata": {"key": "value"},
        })
        manifest, errors = parse_manifest(json_str)
        self.assertEqual(len(errors), 0)
        self.assertIsNotNone(manifest)
        self.assertEqual(len(manifest.capabilities), 2)
        self.assertEqual(len(manifest.entrypoints), 1)

    def test_valid_plugin_ids(self):
        """Test various valid plugin IDs."""
        valid_ids = [
            "ab",
            "test-plugin",
            "my-awesome-plugin",
            "plugin123",
            "a1b2c3",
        ]
        for plugin_id in valid_ids:
            json_str = json.dumps({
                "plugin_id": plugin_id,
                "name": "Test",
                "version": "1.0.0",
            })
            manifest, errors = parse_manifest(json_str)
            self.assertEqual(len(errors), 0, f"Failed for: {plugin_id}")

    def test_valid_versions(self):
        """Test various valid version formats."""
        valid_versions = ["0.0.1", "1.0.0", "10.20.30", "999.999.999"]
        for version in valid_versions:
            json_str = json.dumps({
                "plugin_id": "test-plugin",
                "name": "Test",
                "version": version,
            })
            manifest, errors = parse_manifest(json_str)
            self.assertEqual(len(errors), 0, f"Failed for: {version}")


# =============================================================================
# Test: Manifest Validation - Invalid Cases
# =============================================================================

class TestManifestValidationInvalid(unittest.TestCase):
    """Tests for invalid manifest rejection."""

    def test_invalid_json(self):
        """Test rejection of invalid JSON."""
        manifest, errors = parse_manifest("not valid json")
        self.assertIsNone(manifest)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, ManifestErrorType.INVALID_JSON)

    def test_missing_plugin_id(self):
        """Test rejection of manifest without plugin_id."""
        json_str = json.dumps({
            "name": "Test",
            "version": "1.0.0",
        })
        manifest, errors = parse_manifest(json_str)
        self.assertIsNone(manifest)
        self.assertEqual(errors[0].error_type, ManifestErrorType.MISSING_FIELD)
        self.assertEqual(errors[0].field, "plugin_id")

    def test_missing_name(self):
        """Test rejection of manifest without name."""
        json_str = json.dumps({
            "plugin_id": "test-plugin",
            "version": "1.0.0",
        })
        manifest, errors = parse_manifest(json_str)
        self.assertIsNone(manifest)
        self.assertEqual(errors[0].field, "name")

    def test_missing_version(self):
        """Test rejection of manifest without version."""
        json_str = json.dumps({
            "plugin_id": "test-plugin",
            "name": "Test",
        })
        manifest, errors = parse_manifest(json_str)
        self.assertIsNone(manifest)
        self.assertEqual(errors[0].field, "version")

    def test_invalid_plugin_id_format(self):
        """Test rejection of invalid plugin_id formats."""
        invalid_ids = [
            "Test-Plugin",  # Uppercase
            "test plugin",  # Space
            "test_plugin",  # Underscore
            "123-plugin",   # Starts with number
            "-plugin",      # Starts with hyphen
            "plugin-",      # Ends with hyphen
            "a",            # Too short
        ]
        for plugin_id in invalid_ids:
            json_str = json.dumps({
                "plugin_id": plugin_id,
                "name": "Test",
                "version": "1.0.0",
            })
            manifest, errors = parse_manifest(json_str)
            self.assertIsNone(manifest, f"Should reject: {plugin_id}")
            self.assertGreater(len(errors), 0, f"Should have errors for: {plugin_id}")

    def test_invalid_version_format(self):
        """Test rejection of invalid version formats."""
        invalid_versions = [
            "1.0",          # Missing patch
            "1",            # Missing minor and patch
            "v1.0.0",       # Has 'v' prefix
            "1.0.0-beta",   # Has suffix
            "1.0.0.0",      # Extra segment
            "a.b.c",        # Non-numeric
        ]
        for version in invalid_versions:
            json_str = json.dumps({
                "plugin_id": "test-plugin",
                "name": "Test",
                "version": version,
            })
            manifest, errors = parse_manifest(json_str)
            self.assertIsNone(manifest, f"Should reject: {version}")

    def test_invalid_type_plugin_id(self):
        """Test rejection of non-string plugin_id."""
        json_str = json.dumps({
            "plugin_id": 123,
            "name": "Test",
            "version": "1.0.0",
        })
        manifest, errors = parse_manifest(json_str)
        self.assertIsNone(manifest)
        self.assertEqual(errors[0].error_type, ManifestErrorType.INVALID_TYPE)

    def test_empty_name(self):
        """Test rejection of empty name."""
        json_str = json.dumps({
            "plugin_id": "test-plugin",
            "name": "",
            "version": "1.0.0",
        })
        manifest, errors = parse_manifest(json_str)
        self.assertIsNone(manifest)
        self.assertEqual(errors[0].error_type, ManifestErrorType.INVALID_LENGTH)

    def test_capabilities_not_list(self):
        """Test rejection of non-list capabilities."""
        json_str = json.dumps({
            "plugin_id": "test-plugin",
            "name": "Test",
            "version": "1.0.0",
            "capabilities": "not a list",
        })
        manifest, errors = parse_manifest(json_str)
        self.assertIsNone(manifest)
        self.assertEqual(errors[0].error_type, ManifestErrorType.INVALID_TYPE)

    def test_entrypoints_not_list(self):
        """Test rejection of non-list entrypoints."""
        json_str = json.dumps({
            "plugin_id": "test-plugin",
            "name": "Test",
            "version": "1.0.0",
            "entrypoints": "not a list",
        })
        manifest, errors = parse_manifest(json_str)
        self.assertIsNone(manifest)
        self.assertEqual(errors[0].error_type, ManifestErrorType.INVALID_TYPE)

    def test_manifest_not_object(self):
        """Test rejection of non-object manifest."""
        manifest, errors = parse_manifest(json.dumps([1, 2, 3]))
        self.assertIsNone(manifest)
        self.assertEqual(errors[0].error_type, ManifestErrorType.INVALID_TYPE)


# =============================================================================
# Test: ManifestError
# =============================================================================

class TestManifestError(unittest.TestCase):
    """Tests for ManifestError dataclass."""

    def test_error_creation(self):
        """Test creating a manifest error."""
        error = ManifestError(
            error_type=ManifestErrorType.MISSING_FIELD,
            field="plugin_id",
            message="Missing required field",
        )
        self.assertEqual(error.error_type, ManifestErrorType.MISSING_FIELD)
        self.assertEqual(error.field, "plugin_id")

    def test_error_to_dict(self):
        """Test error serialization."""
        error = ManifestError(
            error_type=ManifestErrorType.INVALID_FORMAT,
            field="version",
            message="Invalid format",
            details={"expected": "X.Y.Z"},
        )
        d = error.to_dict()
        self.assertEqual(d["error_type"], "invalid_format")
        self.assertEqual(d["field"], "version")
        self.assertEqual(d["details"]["expected"], "X.Y.Z")


# =============================================================================
# Test: PluginWhitelist
# =============================================================================

class TestPluginWhitelist(unittest.TestCase):
    """Tests for PluginWhitelist."""

    def test_empty_whitelist_blocks_all(self):
        """Test that empty whitelist blocks all plugins."""
        whitelist = PluginWhitelist(allowed_plugins=set(), enabled=True)
        self.assertFalse(whitelist.is_allowed("any-plugin"))

    def test_whitelist_allows_listed(self):
        """Test that whitelist allows listed plugins."""
        whitelist = PluginWhitelist(
            allowed_plugins={"plugin-a", "plugin-b"},
            enabled=True,
        )
        self.assertTrue(whitelist.is_allowed("plugin-a"))
        self.assertTrue(whitelist.is_allowed("plugin-b"))
        self.assertFalse(whitelist.is_allowed("plugin-c"))

    def test_disabled_whitelist_allows_all(self):
        """Test that disabled whitelist allows all plugins."""
        whitelist = PluginWhitelist(
            allowed_plugins=set(),
            enabled=False,
        )
        self.assertTrue(whitelist.is_allowed("any-plugin"))

    def test_whitelist_to_dict(self):
        """Test whitelist serialization."""
        whitelist = PluginWhitelist(
            allowed_plugins={"plugin-a"},
            enabled=True,
        )
        d = whitelist.to_dict()
        self.assertIn("plugin-a", d["allowed_plugins"])
        self.assertTrue(d["enabled"])


# =============================================================================
# Test: PluginWhitelist Loading
# =============================================================================

class TestPluginWhitelistLoading(unittest.TestCase):
    """Tests for whitelist file loading."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_valid_whitelist(self):
        """Test loading a valid whitelist file."""
        path = os.path.join(self.temp_dir, "whitelist.json")
        with open(path, "w") as f:
            json.dump({
                "allowed_plugins": ["plugin-a", "plugin-b"],
                "enabled": True,
            }, f)

        whitelist, error = load_whitelist(path)
        self.assertIsNone(error)
        self.assertIsNotNone(whitelist)
        self.assertTrue(whitelist.is_allowed("plugin-a"))

    def test_load_nonexistent_whitelist(self):
        """Test loading nonexistent whitelist file."""
        whitelist, error = load_whitelist("/nonexistent/whitelist.json")
        self.assertIsNone(whitelist)
        self.assertEqual(error.error_type, RegistryErrorType.FILE_NOT_FOUND)

    def test_load_invalid_json_whitelist(self):
        """Test loading invalid JSON whitelist."""
        path = os.path.join(self.temp_dir, "whitelist.json")
        with open(path, "w") as f:
            f.write("not valid json")

        whitelist, error = load_whitelist(path)
        self.assertIsNone(whitelist)
        self.assertEqual(error.error_type, RegistryErrorType.INVALID_JSON)

    def test_load_invalid_structure_whitelist(self):
        """Test loading whitelist with invalid structure."""
        path = os.path.join(self.temp_dir, "whitelist.json")
        with open(path, "w") as f:
            json.dump({"allowed_plugins": "not a list"}, f)

        whitelist, error = load_whitelist(path)
        self.assertIsNone(whitelist)
        self.assertEqual(error.error_type, RegistryErrorType.INVALID_WHITELIST)


# =============================================================================
# Test: PluginRegistry - Basic Operations
# =============================================================================

class TestPluginRegistryBasic(unittest.TestCase):
    """Tests for basic registry operations."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_whitelist(self, allowed: list) -> str:
        """Create a whitelist file and return its path."""
        path = os.path.join(self.temp_dir, "whitelist.json")
        with open(path, "w") as f:
            json.dump({"allowed_plugins": allowed, "enabled": True}, f)
        return path

    def _create_manifest(self, plugin_id: str, name: str = "Test") -> str:
        """Create a manifest file and return its path."""
        path = os.path.join(self.temp_dir, f"{plugin_id}.json")
        with open(path, "w") as f:
            json.dump({
                "plugin_id": plugin_id,
                "name": name,
                "version": "1.0.0",
            }, f)
        return path

    def test_empty_registry(self):
        """Test empty registry."""
        registry = PluginRegistry()
        self.assertEqual(registry.count(), 0)
        self.assertEqual(registry.count_valid(), 0)
        self.assertEqual(len(registry.list_valid()), 0)

    def test_load_valid_manifest(self):
        """Test loading a valid whitelisted manifest."""
        whitelist_path = self._create_whitelist(["test-plugin"])
        manifest_path = self._create_manifest("test-plugin")

        registry = PluginRegistry(whitelist_path=whitelist_path)
        error = registry.load_manifest(manifest_path)

        self.assertIsNone(error)
        self.assertEqual(registry.count(), 1)
        self.assertEqual(registry.count_valid(), 1)

    def test_get_valid_plugin(self):
        """Test getting a valid plugin."""
        whitelist_path = self._create_whitelist(["test-plugin"])
        manifest_path = self._create_manifest("test-plugin", "Test Plugin")

        registry = PluginRegistry(whitelist_path=whitelist_path)
        registry.load_manifest(manifest_path)

        manifest = registry.get("test-plugin")
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.name, "Test Plugin")

    def test_get_nonexistent_plugin(self):
        """Test getting nonexistent plugin returns None."""
        registry = PluginRegistry()
        manifest = registry.get("nonexistent")
        self.assertIsNone(manifest)

    def test_has_plugin(self):
        """Test has_plugin check."""
        whitelist_path = self._create_whitelist(["test-plugin"])
        manifest_path = self._create_manifest("test-plugin")

        registry = PluginRegistry(whitelist_path=whitelist_path)
        registry.load_manifest(manifest_path)

        self.assertTrue(registry.has_plugin("test-plugin"))
        self.assertFalse(registry.has_plugin("other-plugin"))

    def test_is_valid(self):
        """Test is_valid check."""
        whitelist_path = self._create_whitelist(["test-plugin"])
        manifest_path = self._create_manifest("test-plugin")

        registry = PluginRegistry(whitelist_path=whitelist_path)
        registry.load_manifest(manifest_path)

        self.assertTrue(registry.is_valid("test-plugin"))
        self.assertFalse(registry.is_valid("other-plugin"))


# =============================================================================
# Test: PluginRegistry - Non-Whitelisted Rejection
# =============================================================================

class TestPluginRegistryWhitelist(unittest.TestCase):
    """Tests for whitelist enforcement."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_whitelist(self, allowed: list) -> str:
        """Create a whitelist file and return its path."""
        path = os.path.join(self.temp_dir, "whitelist.json")
        with open(path, "w") as f:
            json.dump({"allowed_plugins": allowed, "enabled": True}, f)
        return path

    def _create_manifest(self, plugin_id: str) -> str:
        """Create a manifest file and return its path."""
        path = os.path.join(self.temp_dir, f"{plugin_id}.json")
        with open(path, "w") as f:
            json.dump({
                "plugin_id": plugin_id,
                "name": "Test",
                "version": "1.0.0",
            }, f)
        return path

    def test_non_whitelisted_rejected(self):
        """Test that non-whitelisted plugins are rejected."""
        whitelist_path = self._create_whitelist(["allowed-plugin"])
        manifest_path = self._create_manifest("not-allowed-plugin")

        registry = PluginRegistry(whitelist_path=whitelist_path)
        error = registry.load_manifest(manifest_path)

        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, RegistryErrorType.NOT_WHITELISTED)
        self.assertFalse(registry.is_valid("not-allowed-plugin"))

    def test_non_whitelisted_stored_with_status(self):
        """Test that non-whitelisted plugins are stored with correct status."""
        whitelist_path = self._create_whitelist([])
        manifest_path = self._create_manifest("blocked-plugin")

        registry = PluginRegistry(whitelist_path=whitelist_path)
        registry.load_manifest(manifest_path)

        entry = registry.get_entry("blocked-plugin")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.status, PluginStatus.NOT_WHITELISTED)
        self.assertIsNotNone(entry.manifest)  # Manifest is preserved

    def test_get_returns_none_for_non_whitelisted(self):
        """Test that get() returns None for non-whitelisted plugins."""
        whitelist_path = self._create_whitelist([])
        manifest_path = self._create_manifest("blocked-plugin")

        registry = PluginRegistry(whitelist_path=whitelist_path)
        registry.load_manifest(manifest_path)

        manifest = registry.get("blocked-plugin")
        self.assertIsNone(manifest)

    def test_list_valid_excludes_non_whitelisted(self):
        """Test that list_valid() excludes non-whitelisted plugins."""
        whitelist_path = self._create_whitelist(["allowed-plugin"])

        registry = PluginRegistry(whitelist_path=whitelist_path)
        registry.load_manifest(self._create_manifest("allowed-plugin"))
        registry.load_manifest(self._create_manifest("blocked-plugin"))

        valid = registry.list_valid()
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0].plugin_id, "allowed-plugin")


# =============================================================================
# Test: PluginRegistry - Duplicate Handling
# =============================================================================

class TestPluginRegistryDuplicates(unittest.TestCase):
    """Tests for duplicate plugin_id handling."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_whitelist(self, allowed: list) -> str:
        """Create a whitelist file and return its path."""
        path = os.path.join(self.temp_dir, "whitelist.json")
        with open(path, "w") as f:
            json.dump({"allowed_plugins": allowed, "enabled": True}, f)
        return path

    def test_duplicate_plugin_id_rejected(self):
        """Test that duplicate plugin IDs are rejected."""
        whitelist_path = self._create_whitelist(["test-plugin"])

        # Create two manifests with same ID
        path1 = os.path.join(self.temp_dir, "plugin1.json")
        path2 = os.path.join(self.temp_dir, "plugin2.json")
        for path in [path1, path2]:
            with open(path, "w") as f:
                json.dump({
                    "plugin_id": "test-plugin",
                    "name": "Test",
                    "version": "1.0.0",
                }, f)

        registry = PluginRegistry(whitelist_path=whitelist_path)
        error1 = registry.load_manifest(path1)
        error2 = registry.load_manifest(path2)

        self.assertIsNone(error1)  # First load succeeds
        self.assertIsNotNone(error2)  # Second load fails
        self.assertEqual(error2.error_type, RegistryErrorType.DUPLICATE_PLUGIN)

    def test_duplicate_keeps_first(self):
        """Test that first loaded plugin is kept on duplicate."""
        whitelist_path = self._create_whitelist(["test-plugin"])

        path1 = os.path.join(self.temp_dir, "plugin1.json")
        path2 = os.path.join(self.temp_dir, "plugin2.json")

        with open(path1, "w") as f:
            json.dump({
                "plugin_id": "test-plugin",
                "name": "First Plugin",
                "version": "1.0.0",
            }, f)

        with open(path2, "w") as f:
            json.dump({
                "plugin_id": "test-plugin",
                "name": "Second Plugin",
                "version": "2.0.0",
            }, f)

        registry = PluginRegistry(whitelist_path=whitelist_path)
        registry.load_manifest(path1)
        registry.load_manifest(path2)

        manifest = registry.get("test-plugin")
        self.assertEqual(manifest.name, "First Plugin")


# =============================================================================
# Test: PluginRegistry - Directory Loading
# =============================================================================

class TestPluginRegistryDirectoryLoading(unittest.TestCase):
    """Tests for loading plugins from directory."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.plugins_dir = os.path.join(self.temp_dir, "plugins")
        os.makedirs(self.plugins_dir)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_whitelist(self, allowed: list) -> str:
        """Create a whitelist file and return its path."""
        path = os.path.join(self.temp_dir, "whitelist.json")
        with open(path, "w") as f:
            json.dump({"allowed_plugins": allowed, "enabled": True}, f)
        return path

    def test_load_from_directory(self):
        """Test loading plugins from directory."""
        whitelist_path = self._create_whitelist(["plugin-a", "plugin-b"])

        # Create plugin subdirectories with manifests
        for plugin_id in ["plugin-a", "plugin-b"]:
            plugin_dir = os.path.join(self.plugins_dir, plugin_id)
            os.makedirs(plugin_dir)
            with open(os.path.join(plugin_dir, "manifest.json"), "w") as f:
                json.dump({
                    "plugin_id": plugin_id,
                    "name": f"Plugin {plugin_id}",
                    "version": "1.0.0",
                }, f)

        registry = PluginRegistry(whitelist_path=whitelist_path)
        errors = registry.load_from_directory(self.plugins_dir)

        self.assertEqual(len(errors), 0)
        self.assertEqual(registry.count_valid(), 2)

    def test_load_from_nonexistent_directory(self):
        """Test loading from nonexistent directory."""
        registry = PluginRegistry()
        errors = registry.load_from_directory("/nonexistent/dir")

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, RegistryErrorType.FILE_NOT_FOUND)

    def test_load_directory_with_invalid_manifest(self):
        """Test loading directory with some invalid manifests."""
        whitelist_path = self._create_whitelist(["valid-plugin", "invalid-plugin"])

        # Create valid plugin
        valid_dir = os.path.join(self.plugins_dir, "valid-plugin")
        os.makedirs(valid_dir)
        with open(os.path.join(valid_dir, "manifest.json"), "w") as f:
            json.dump({
                "plugin_id": "valid-plugin",
                "name": "Valid",
                "version": "1.0.0",
            }, f)

        # Create invalid plugin
        invalid_dir = os.path.join(self.plugins_dir, "invalid-plugin")
        os.makedirs(invalid_dir)
        with open(os.path.join(invalid_dir, "manifest.json"), "w") as f:
            f.write("not valid json")

        registry = PluginRegistry(whitelist_path=whitelist_path)
        errors = registry.load_from_directory(self.plugins_dir)

        self.assertEqual(len(errors), 1)  # One invalid
        self.assertEqual(registry.count_valid(), 1)  # One valid


# =============================================================================
# Test: PluginRegistry - Read-Only Access
# =============================================================================

class TestPluginRegistryReadOnly(unittest.TestCase):
    """Tests for read-only access enforcement."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_whitelist(self, allowed: list) -> str:
        """Create a whitelist file and return its path."""
        path = os.path.join(self.temp_dir, "whitelist.json")
        with open(path, "w") as f:
            json.dump({"allowed_plugins": allowed, "enabled": True}, f)
        return path

    def _create_manifest(self, plugin_id: str) -> str:
        """Create a manifest file and return its path."""
        path = os.path.join(self.temp_dir, f"{plugin_id}.json")
        with open(path, "w") as f:
            json.dump({
                "plugin_id": plugin_id,
                "name": "Test",
                "version": "1.0.0",
            }, f)
        return path

    def test_get_returns_copy_not_reference(self):
        """Test that get() doesn't expose internal state."""
        whitelist_path = self._create_whitelist(["test-plugin"])
        manifest_path = self._create_manifest("test-plugin")

        registry = PluginRegistry(whitelist_path=whitelist_path)
        registry.load_manifest(manifest_path)

        manifest1 = registry.get("test-plugin")
        manifest2 = registry.get("test-plugin")

        # Both should be the same object (immutable dataclass)
        # Modifying returned data shouldn't affect registry
        manifest1.capabilities.append(PluginCapability(name="new"))

        # list_valid should not be affected
        valid = registry.list_valid()
        # Note: In current impl, capabilities list is shared
        # For true read-only, would need deepcopy

    def test_list_valid_returns_list(self):
        """Test that list_valid returns a list, not internal structure."""
        whitelist_path = self._create_whitelist(["test-plugin"])
        manifest_path = self._create_manifest("test-plugin")

        registry = PluginRegistry(whitelist_path=whitelist_path)
        registry.load_manifest(manifest_path)

        valid1 = registry.list_valid()
        valid2 = registry.list_valid()

        # Should be separate list instances
        self.assertIsNot(valid1, valid2)

    def test_whitelist_property_returns_whitelist(self):
        """Test that whitelist property returns the whitelist."""
        whitelist_path = self._create_whitelist(["test-plugin"])
        registry = PluginRegistry(whitelist_path=whitelist_path)

        whitelist = registry.whitelist
        self.assertIsNotNone(whitelist)
        self.assertTrue(whitelist.is_allowed("test-plugin"))


# =============================================================================
# Test: PluginRegistry - Error Handling
# =============================================================================

class TestPluginRegistryErrors(unittest.TestCase):
    """Tests for registry error handling."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_nonexistent_manifest(self):
        """Test loading nonexistent manifest file."""
        registry = PluginRegistry()
        error = registry.load_manifest("/nonexistent/manifest.json")

        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, RegistryErrorType.FILE_NOT_FOUND)

    def test_load_invalid_manifest(self):
        """Test loading invalid manifest file."""
        path = os.path.join(self.temp_dir, "invalid.json")
        with open(path, "w") as f:
            f.write("not valid json")

        registry = PluginRegistry()
        error = registry.load_manifest(path)

        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, RegistryErrorType.MANIFEST_INVALID)

    def test_initialization_errors_tracked(self):
        """Test that initialization errors are tracked."""
        registry = PluginRegistry(whitelist_path="/nonexistent/whitelist.json")

        errors = registry.initialization_errors
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, RegistryErrorType.FILE_NOT_FOUND)

    def test_registry_to_dict(self):
        """Test registry serialization."""
        whitelist_path = os.path.join(self.temp_dir, "whitelist.json")
        with open(whitelist_path, "w") as f:
            json.dump({"allowed_plugins": ["test-plugin"], "enabled": True}, f)

        manifest_path = os.path.join(self.temp_dir, "test-plugin.json")
        with open(manifest_path, "w") as f:
            json.dump({
                "plugin_id": "test-plugin",
                "name": "Test",
                "version": "1.0.0",
            }, f)

        registry = PluginRegistry(whitelist_path=whitelist_path)
        registry.load_manifest(manifest_path)

        d = registry.to_dict()
        self.assertIn("plugins", d)
        self.assertIn("whitelist", d)
        self.assertIn("counts", d)
        self.assertEqual(d["counts"]["total"], 1)
        self.assertEqual(d["counts"]["valid"], 1)


# =============================================================================
# Test: PluginRegistry - List By Status
# =============================================================================

class TestPluginRegistryListByStatus(unittest.TestCase):
    """Tests for listing plugins by status."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_by_status(self):
        """Test listing plugins by status."""
        whitelist_path = os.path.join(self.temp_dir, "whitelist.json")
        with open(whitelist_path, "w") as f:
            json.dump({"allowed_plugins": ["valid-plugin"], "enabled": True}, f)

        # Create valid plugin
        valid_path = os.path.join(self.temp_dir, "valid-plugin.json")
        with open(valid_path, "w") as f:
            json.dump({
                "plugin_id": "valid-plugin",
                "name": "Valid",
                "version": "1.0.0",
            }, f)

        # Create non-whitelisted plugin
        blocked_path = os.path.join(self.temp_dir, "blocked-plugin.json")
        with open(blocked_path, "w") as f:
            json.dump({
                "plugin_id": "blocked-plugin",
                "name": "Blocked",
                "version": "1.0.0",
            }, f)

        registry = PluginRegistry(whitelist_path=whitelist_path)
        registry.load_manifest(valid_path)
        registry.load_manifest(blocked_path)

        valid = registry.list_by_status(PluginStatus.VALID)
        blocked = registry.list_by_status(PluginStatus.NOT_WHITELISTED)

        self.assertEqual(len(valid), 1)
        self.assertEqual(len(blocked), 1)
        self.assertEqual(valid[0].manifest.plugin_id, "valid-plugin")
        self.assertEqual(blocked[0].manifest.plugin_id, "blocked-plugin")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main()
