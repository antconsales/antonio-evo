"""
Prompt Library (v8.0) — Versioned prompt templates with performance tracking.

Provides:
- Named, versioned prompt templates (system, persona, task, tool)
- Version history with rollback
- Performance tracking per template
- Few-shot example retrieval from high-confidence memory neurons
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A versioned prompt template."""
    id: int
    name: str
    category: str  # system, persona, task, tool
    template: str
    version: int
    variables: Dict[str, str]
    is_active: bool
    performance_score: float
    usage_count: int
    parent_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "template": self.template,
            "version": self.version,
            "variables": self.variables,
            "is_active": self.is_active,
            "performance_score": self.performance_score,
            "usage_count": self.usage_count,
        }


class PromptLibrary:
    """
    SQLite-backed prompt template library with versioning and performance tracking.

    Tables:
    - prompt_templates: name, category, template, version, performance
    - prompt_performance: per-use tracking linked to neurons
    """

    def __init__(self, db_path: str = "data/evomemory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        self._seed_default_templates()

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
                CREATE TABLE IF NOT EXISTS prompt_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'system',
                    template TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    variables TEXT DEFAULT '{}',
                    is_active BOOLEAN DEFAULT 1,
                    performance_score REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    parent_id INTEGER,
                    created_at REAL NOT NULL,
                    UNIQUE(name, version)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pt_name
                ON prompt_templates(name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pt_active
                ON prompt_templates(is_active)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompt_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    template_id INTEGER NOT NULL,
                    neuron_id TEXT,
                    feedback_score REAL DEFAULT 0,
                    response_length INTEGER DEFAULT 0,
                    elapsed_ms INTEGER DEFAULT 0,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (template_id) REFERENCES prompt_templates(id)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pp_template
                ON prompt_performance(template_id)
            """)

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _seed_default_templates(self):
        """Seed default templates if none exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) as cnt FROM prompt_templates")
            if cursor.fetchone()["cnt"] > 0:
                return  # Already seeded

            defaults = [
                {
                    "name": "social_system",
                    "category": "persona",
                    "template": (
                        "Sei Antonio, un assistente AI amichevole. "
                        "Rispondi in modo naturale nella lingua dell'utente. "
                        "NON analizzare - rispondi direttamente alla domanda come farebbe un amico."
                    ),
                },
                {
                    "name": "logic_system",
                    "category": "persona",
                    "template": (
                        "Sei Antonio in modalità LOGIC. "
                        "Rispondi in modo analitico e preciso. "
                        "Struttura le risposte con chiarezza. Usa dati e fatti."
                    ),
                },
                {
                    "name": "classify_task",
                    "category": "task",
                    "template": (
                        "TASK: CLASSIFY\n\n"
                        "INPUT: {user_message}\n\n"
                        "Respond with JSON: intent, domain, complexity, requires_external, confidence, reasoning."
                    ),
                },
                {
                    "name": "tool_instructions",
                    "category": "tool",
                    "template": (
                        "You have access to tools you can call to help the user. "
                        "When a question requires current information, files, code execution, or image analysis, "
                        "use the appropriate tool. After getting tool results, synthesize them into a clear response."
                    ),
                },
            ]

            now = time.time()
            for d in defaults:
                cursor.execute(
                    "INSERT INTO prompt_templates (name, category, template, version, variables, is_active, performance_score, usage_count, created_at) "
                    "VALUES (?, ?, ?, 1, '{}', 1, 0.5, 0, ?)",
                    (d["name"], d["category"], d["template"], now),
                )
            conn.commit()
        except Exception:
            conn.rollback()
        finally:
            cursor.close()

    # ---- Template CRUD ----

    def get_template(self, name: str, category: Optional[str] = None) -> Optional[PromptTemplate]:
        """Get the active version of a template by name."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            if category:
                cursor.execute(
                    "SELECT * FROM prompt_templates WHERE name = ? AND category = ? AND is_active = 1 "
                    "ORDER BY version DESC LIMIT 1",
                    (name, category),
                )
            else:
                cursor.execute(
                    "SELECT * FROM prompt_templates WHERE name = ? AND is_active = 1 "
                    "ORDER BY version DESC LIMIT 1",
                    (name,),
                )
            row = cursor.fetchone()
            if row:
                return self._row_to_template(row)
            return None
        finally:
            cursor.close()

    def get_all_templates(self, category: Optional[str] = None) -> List[PromptTemplate]:
        """Get all active templates."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            if category:
                cursor.execute(
                    "SELECT * FROM prompt_templates WHERE category = ? AND is_active = 1 ORDER BY name",
                    (category,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM prompt_templates WHERE is_active = 1 ORDER BY category, name"
                )
            return [self._row_to_template(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def update_template(self, name: str, new_template: str) -> Optional[PromptTemplate]:
        """
        Update a template by creating a new version (old version deactivated).
        """
        current = self.get_template(name)
        if not current:
            return None

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            # Deactivate current version
            cursor.execute(
                "UPDATE prompt_templates SET is_active = 0 WHERE name = ? AND is_active = 1",
                (name,),
            )
            # Create new version
            now = time.time()
            cursor.execute(
                "INSERT INTO prompt_templates (name, category, template, version, variables, is_active, performance_score, usage_count, parent_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, 1, 0.5, 0, ?, ?)",
                (name, current.category, new_template, current.version + 1, json.dumps(current.variables), current.id, now),
            )
            conn.commit()
            return self.get_template(name)
        except Exception:
            conn.rollback()
            return None
        finally:
            cursor.close()

    def rollback_template(self, name: str) -> Optional[PromptTemplate]:
        """Rollback a template to its previous version."""
        current = self.get_template(name)
        if not current or not current.parent_id:
            return None

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            # Deactivate current
            cursor.execute(
                "UPDATE prompt_templates SET is_active = 0 WHERE id = ?",
                (current.id,),
            )
            # Reactivate parent
            cursor.execute(
                "UPDATE prompt_templates SET is_active = 1 WHERE id = ?",
                (current.parent_id,),
            )
            conn.commit()
            return self.get_template(name)
        except Exception:
            conn.rollback()
            return None
        finally:
            cursor.close()

    # ---- Performance Tracking ----

    def record_performance(
        self, template_id: int, neuron_id: Optional[str], feedback_score: float, elapsed_ms: int = 0
    ):
        """Record performance of a template usage."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO prompt_performance (template_id, neuron_id, feedback_score, elapsed_ms, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (template_id, neuron_id, feedback_score, elapsed_ms, time.time()),
            )
            # Update aggregate score and usage count
            cursor.execute(
                "UPDATE prompt_templates SET usage_count = usage_count + 1, "
                "performance_score = (performance_score * usage_count + ?) / (usage_count + 1) "
                "WHERE id = ?",
                (feedback_score, template_id),
            )
            conn.commit()
        finally:
            cursor.close()

    def get_best_template(self, category: str) -> Optional[PromptTemplate]:
        """Get the highest-performing template in a category."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM prompt_templates WHERE category = ? AND is_active = 1 "
                "AND usage_count >= 3 ORDER BY performance_score DESC LIMIT 1",
                (category,),
            )
            row = cursor.fetchone()
            return self._row_to_template(row) if row else None
        finally:
            cursor.close()

    # ---- Few-Shot Examples ----

    def get_few_shot_examples(
        self, query: str, limit: int = 3, memory_storage=None
    ) -> List[Dict[str, str]]:
        """
        Get few-shot examples from high-confidence memory neurons.

        Returns User/Assistant pairs from neurons with confidence >= 0.8.
        """
        if not memory_storage:
            return []

        try:
            # Search high-confidence neurons
            results = memory_storage.search_bm25(query, limit=limit, min_confidence=0.8)
            examples = []
            for neuron, score in results:
                examples.append({
                    "user": neuron.input[:200],
                    "assistant": neuron.output[:300],
                })
            return examples
        except Exception:
            return []

    # ---- Version History ----

    def get_version_history(self, name: str) -> List[PromptTemplate]:
        """Get all versions of a template."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM prompt_templates WHERE name = ? ORDER BY version DESC",
                (name,),
            )
            return [self._row_to_template(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    # ---- Stats ----

    def get_stats(self) -> Dict[str, Any]:
        """Get prompt library statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) as cnt FROM prompt_templates WHERE is_active = 1")
            active = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM prompt_templates")
            total = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM prompt_performance")
            perf_records = cursor.fetchone()["cnt"]

            cursor.execute(
                "SELECT category, COUNT(*) as cnt FROM prompt_templates WHERE is_active = 1 GROUP BY category"
            )
            by_category = {row["category"]: row["cnt"] for row in cursor.fetchall()}

            return {
                "enabled": True,
                "version": "8.0",
                "active_templates": active,
                "total_versions": total,
                "performance_records": perf_records,
                "by_category": by_category,
            }
        finally:
            cursor.close()

    # ---- Helpers ----

    def _row_to_template(self, row) -> PromptTemplate:
        variables = {}
        try:
            variables = json.loads(row["variables"]) if row["variables"] else {}
        except (json.JSONDecodeError, TypeError):
            pass

        return PromptTemplate(
            id=row["id"],
            name=row["name"],
            category=row["category"],
            template=row["template"],
            version=row["version"],
            variables=variables,
            is_active=bool(row["is_active"]),
            performance_score=row["performance_score"],
            usage_count=row["usage_count"],
            parent_id=row.get("parent_id"),
        )
