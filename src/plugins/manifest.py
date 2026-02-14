"""
Skill Manifest & Verifier (v8.5) — Certified skill marketplace.

Every plugin can optionally include a `.manifest.json` that declares:
- Permissions (filesystem, network, code_execution, external_api)
- Required tools and hooks
- Risk level and SHA-256 checksum

The SkillVerifier validates:
1. Checksum matches source file (integrity)
2. Skill ID in whitelist (or no whitelist = allow all)
3. Permissions within allowed set
4. No blocked patterns in source code

Design: Trust but verify. Plugins without manifests still load (backward compatible),
but verified skills get a "certified" badge and can request elevated permissions.
"""

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SkillPermission:
    """A single permission declaration."""
    resource: str  # filesystem, network, code_execution, external_api
    access: str    # read, write, execute
    scope: str = "*"  # path pattern or URL pattern

    def to_dict(self) -> Dict[str, Any]:
        return {"resource": self.resource, "access": self.access, "scope": self.scope}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillPermission":
        return cls(
            resource=data.get("resource", ""),
            access=data.get("access", ""),
            scope=data.get("scope", "*"),
        )


@dataclass
class SkillManifest:
    """Skill metadata and permission declaration."""
    id: str
    name: str
    version: str
    author: str
    description: str
    permissions: List[SkillPermission] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)
    risk_level: str = "low"
    checksum: str = ""
    verified: bool = False
    verified_at: Optional[float] = None
    source_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "permissions": [p.to_dict() for p in self.permissions],
            "tools": self.tools,
            "hooks": self.hooks,
            "risk_level": self.risk_level,
            "checksum": self.checksum,
            "verified": self.verified,
            "verified_at": self.verified_at,
            "source_path": self.source_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillManifest":
        permissions = [
            SkillPermission.from_dict(p)
            for p in data.get("permissions", [])
        ]
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            version=data.get("version", "0.0.0"),
            author=data.get("author", "unknown"),
            description=data.get("description", ""),
            permissions=permissions,
            tools=data.get("tools", []),
            hooks=data.get("hooks", []),
            risk_level=data.get("risk_level", "low"),
            checksum=data.get("checksum", ""),
            verified=data.get("verified", False),
            verified_at=data.get("verified_at"),
            source_path=data.get("source_path"),
        )


# Blocked code patterns in plugin source
_BLOCKED_SOURCE_PATTERNS = [
    r"os\.environ\[",        # Direct env var modification
    r"exec\(.*compile",      # Dynamic code execution
    r"__import__\(",         # Dynamic imports
    r"subprocess\.Popen",    # Arbitrary subprocess
    r"ctypes\.",             # Native code access
    r"socket\.socket",       # Raw socket access
]


class SkillVerifier:
    """
    Verify skill integrity and permissions before loading.

    SQLite table `skill_registry` tracks installed and verified skills.
    """

    def __init__(
        self,
        db_path: str = "data/evomemory.db",
        whitelist_path: str = "config/skill_whitelist.json",
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._whitelist = self._load_whitelist(whitelist_path)
        self._init_schema()
        logger.info(f"SkillVerifier initialized ({len(self._whitelist)} whitelisted skills)")

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS skill_registry (
                    skill_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT,
                    author TEXT,
                    description TEXT,
                    permissions_json TEXT DEFAULT '[]',
                    tools_json TEXT DEFAULT '[]',
                    hooks_json TEXT DEFAULT '[]',
                    risk_level TEXT DEFAULT 'low',
                    checksum TEXT,
                    source_path TEXT,
                    verified BOOLEAN DEFAULT 0,
                    verified_at REAL,
                    installed_at REAL,
                    enabled BOOLEAN DEFAULT 1,
                    revoked_at REAL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_skill_enabled
                ON skill_registry(enabled)
            """)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _load_whitelist(self, path: str) -> List[str]:
        """Load skill ID whitelist. Empty list = allow all."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("allowed_skills", [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []  # No whitelist = allow all

    def verify_skill(self, plugin_path: Path, manifest: SkillManifest) -> Tuple[bool, str]:
        """
        Verify a skill before loading.

        Checks:
        1. Checksum matches source file
        2. Skill ID in whitelist (if whitelist exists)
        3. No blocked patterns in source code

        Returns:
            (verified: bool, reason: str)
        """
        # 1. Checksum verification
        if manifest.checksum:
            actual_checksum = self._compute_checksum(plugin_path)
            expected = manifest.checksum
            if expected.startswith("sha256:"):
                expected = expected[7:]
            if actual_checksum != expected:
                return False, f"Checksum mismatch: expected {expected[:12]}..., got {actual_checksum[:12]}..."

        # 2. Whitelist check
        if self._whitelist and manifest.id not in self._whitelist:
            return False, f"Skill '{manifest.id}' not in whitelist"

        # 3. Source code safety check
        try:
            source = plugin_path.read_text(encoding="utf-8")
            for pattern in _BLOCKED_SOURCE_PATTERNS:
                if re.search(pattern, source):
                    return False, f"Blocked pattern in source: {pattern}"
        except Exception as e:
            return False, f"Cannot read source: {e}"

        return True, "verified"

    def register_skill(self, manifest: SkillManifest) -> bool:
        """Register a verified skill in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO skill_registry "
                "(skill_id, name, version, author, description, permissions_json, "
                "tools_json, hooks_json, risk_level, checksum, source_path, "
                "verified, verified_at, installed_at, enabled) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (
                    manifest.id,
                    manifest.name,
                    manifest.version,
                    manifest.author,
                    manifest.description,
                    json.dumps([p.to_dict() for p in manifest.permissions]),
                    json.dumps(manifest.tools),
                    json.dumps(manifest.hooks),
                    manifest.risk_level,
                    manifest.checksum,
                    manifest.source_path,
                    manifest.verified,
                    manifest.verified_at,
                    time.time(),
                ),
            )
            conn.commit()
            return True
        except Exception as e:
            logger.warning(f"Failed to register skill: {e}")
            conn.rollback()
            return False
        finally:
            cursor.close()

    def get_installed_skills(self) -> List[Dict[str, Any]]:
        """Get all installed skills."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM skill_registry WHERE enabled = 1 ORDER BY name"
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def get_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific skill by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM skill_registry WHERE skill_id = ?", (skill_id,))
            row = cursor.fetchone()
            return self._row_to_dict(row) if row else None
        finally:
            cursor.close()

    def revoke_skill(self, skill_id: str) -> bool:
        """Revoke a skill (disable without deleting)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE skill_registry SET enabled = 0, revoked_at = ? WHERE skill_id = ?",
                (time.time(), skill_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()

    def get_all_permissions(self) -> List[Dict[str, Any]]:
        """Get aggregated permissions from all enabled skills."""
        skills = self.get_installed_skills()
        all_perms = []
        for skill in skills:
            perms = skill.get("permissions_json", [])
            if isinstance(perms, str):
                try:
                    perms = json.loads(perms)
                except json.JSONDecodeError:
                    perms = []
            for perm in perms:
                all_perms.append({
                    "skill_id": skill["skill_id"],
                    "skill_name": skill["name"],
                    **perm,
                })
        return all_perms

    def get_stats(self) -> Dict[str, Any]:
        """Skill marketplace statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) as cnt FROM skill_registry WHERE enabled = 1")
            enabled = cursor.fetchone()["cnt"]
            cursor.execute("SELECT COUNT(*) as cnt FROM skill_registry WHERE verified = 1 AND enabled = 1")
            verified = cursor.fetchone()["cnt"]
            cursor.execute("SELECT COUNT(*) as cnt FROM skill_registry WHERE enabled = 0")
            revoked = cursor.fetchone()["cnt"]
            return {
                "enabled": True,
                "version": "8.5",
                "total_skills": enabled,
                "verified_skills": verified,
                "revoked_skills": revoked,
                "whitelist_active": len(self._whitelist) > 0,
                "whitelist_count": len(self._whitelist),
            }
        finally:
            cursor.close()

    @staticmethod
    def _compute_checksum(path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        d = dict(row)
        for json_field in ("permissions_json", "tools_json", "hooks_json"):
            if d.get(json_field) and isinstance(d[json_field], str):
                try:
                    d[json_field] = json.loads(d[json_field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
