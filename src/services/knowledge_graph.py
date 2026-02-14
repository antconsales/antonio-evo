"""
Knowledge Graph Service (v8.0) — Entity and relationship extraction from conversations.

Extracts entities (person, project, tool, concept) and relationships from every
conversation, stores in SQLite, injects graph context into prompts so the LLM
"reasons" about the user's knowledge topology.
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A knowledge graph entity."""
    id: int
    name: str
    entity_type: str  # person, project, tool, concept, place, organization
    description: str = ""
    frequency: int = 1
    confidence: float = 0.5
    first_seen: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class Relationship:
    """A knowledge graph relationship."""
    id: int
    subject_id: int
    predicate: str  # works_on, uses, knows, is_part_of, related_to
    object_id: int
    confidence: float = 0.5
    frequency: int = 1
    source_neuron_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "confidence": self.confidence,
            "frequency": self.frequency,
            "source_neuron_id": self.source_neuron_id,
        }


class KnowledgeGraphService:
    """
    SQLite-backed knowledge graph with LLM-powered entity extraction.

    Tables:
    - kg_entities: name, type, description, frequency, confidence
    - kg_relationships: subject → predicate → object with confidence
    """

    def __init__(
        self,
        db_path: str = "data/evomemory.db",
        ollama_url: str = "http://localhost:11434",
        model: str = "mistral",
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ollama_url = ollama_url
        self.model = model
        self._local = threading.local()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA foreign_keys = ON")
        return self._local.conn

    def _init_schema(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kg_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    entity_type TEXT NOT NULL DEFAULT 'concept',
                    description TEXT DEFAULT '',
                    frequency INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.5,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_kg_entities_name
                ON kg_entities(name COLLATE NOCASE)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_kg_entities_type
                ON kg_entities(entity_type)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kg_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id INTEGER NOT NULL,
                    predicate TEXT NOT NULL,
                    object_id INTEGER NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    frequency INTEGER DEFAULT 1,
                    source_neuron_id TEXT,
                    FOREIGN KEY (subject_id) REFERENCES kg_entities(id),
                    FOREIGN KEY (object_id) REFERENCES kg_entities(id),
                    UNIQUE(subject_id, predicate, object_id)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_kg_rel_subject
                ON kg_relationships(subject_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_kg_rel_object
                ON kg_relationships(object_id)
            """)

            # FTS for entity name search
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS kg_entities_fts USING fts5(
                    name, description,
                    content='kg_entities',
                    content_rowid='id'
                )
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS kg_entities_ai AFTER INSERT ON kg_entities BEGIN
                    INSERT INTO kg_entities_fts(rowid, name, description)
                    VALUES (NEW.id, NEW.name, NEW.description);
                END
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS kg_entities_ad AFTER DELETE ON kg_entities BEGIN
                    INSERT INTO kg_entities_fts(kg_entities_fts, rowid, name, description)
                    VALUES ('delete', OLD.id, OLD.name, OLD.description);
                END
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS kg_entities_au AFTER UPDATE ON kg_entities BEGIN
                    INSERT INTO kg_entities_fts(kg_entities_fts, rowid, name, description)
                    VALUES ('delete', OLD.id, OLD.name, OLD.description);
                    INSERT INTO kg_entities_fts(rowid, name, description)
                    VALUES (NEW.id, NEW.name, NEW.description);
                END
            """)

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    # ---- Entity extraction via Ollama ----

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relationships from text using Ollama.

        Returns dict with "entities" and "relationships" lists.
        """
        if not text or len(text.strip()) < 10:
            return {"entities": [], "relationships": []}

        prompt = (
            "Extract entities and relationships from this conversation text.\n"
            "Return JSON only. Format:\n"
            '{"entities": [{"name": "...", "type": "person|project|tool|concept|place|organization", "description": "..."}],\n'
            ' "relationships": [{"subject": "...", "predicate": "works_on|uses|knows|is_part_of|related_to", "object": "..."}]}\n\n'
            "Rules:\n"
            "- Only extract clearly mentioned entities, not generic words\n"
            "- Entity names should be proper nouns or specific terms\n"
            "- If no entities found, return empty arrays\n\n"
            f"Text:\n{text[:1500]}"
        )

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            text_output = result.get("response", "")
            parsed = json.loads(text_output)
            return {
                "entities": parsed.get("entities", []),
                "relationships": parsed.get("relationships", []),
            }
        except Exception as e:
            logger.debug(f"KG extraction failed: {e}")
            return {"entities": [], "relationships": []}

    # ---- Storage ----

    def _upsert_entity(self, name: str, entity_type: str, description: str = "") -> int:
        """Upsert an entity, returning its ID."""
        now = time.time()
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT id, frequency FROM kg_entities WHERE name = ? COLLATE NOCASE",
                (name,),
            )
            row = cursor.fetchone()
            if row:
                cursor.execute(
                    "UPDATE kg_entities SET frequency = frequency + 1, last_seen = ?, "
                    "confidence = MIN(1.0, confidence + 0.05) WHERE id = ?",
                    (now, row["id"]),
                )
                conn.commit()
                return row["id"]
            else:
                cursor.execute(
                    "INSERT INTO kg_entities (name, entity_type, description, frequency, confidence, first_seen, last_seen) "
                    "VALUES (?, ?, ?, 1, 0.5, ?, ?)",
                    (name, entity_type, description, now, now),
                )
                conn.commit()
                return cursor.lastrowid
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _upsert_relationship(
        self, subject_id: int, predicate: str, object_id: int, neuron_id: Optional[str] = None
    ):
        """Upsert a relationship."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT id FROM kg_relationships WHERE subject_id = ? AND predicate = ? AND object_id = ?",
                (subject_id, predicate, object_id),
            )
            row = cursor.fetchone()
            if row:
                cursor.execute(
                    "UPDATE kg_relationships SET frequency = frequency + 1, "
                    "confidence = MIN(1.0, confidence + 0.05) WHERE id = ?",
                    (row["id"],),
                )
            else:
                cursor.execute(
                    "INSERT INTO kg_relationships (subject_id, predicate, object_id, confidence, frequency, source_neuron_id) "
                    "VALUES (?, ?, ?, 0.5, 1, ?)",
                    (subject_id, predicate, object_id, neuron_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def store_entities(
        self, entities: List[Dict], relationships: List[Dict], neuron_id: Optional[str] = None
    ):
        """Store extracted entities and relationships."""
        entity_ids = {}

        for ent in entities:
            name = ent.get("name", "").strip()
            if not name or len(name) < 2:
                continue
            etype = ent.get("type", "concept")
            desc = ent.get("description", "")
            try:
                eid = self._upsert_entity(name, etype, desc)
                entity_ids[name.lower()] = eid
            except Exception as e:
                logger.debug(f"KG entity store failed for '{name}': {e}")

        for rel in relationships:
            subj = rel.get("subject", "").strip().lower()
            obj = rel.get("object", "").strip().lower()
            pred = rel.get("predicate", "related_to")
            if subj in entity_ids and obj in entity_ids:
                try:
                    self._upsert_relationship(
                        entity_ids[subj], pred, entity_ids[obj], neuron_id
                    )
                except Exception as e:
                    logger.debug(f"KG relationship store failed: {e}")

    # ---- Retrieval ----

    def get_related_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities by FTS and return with 1-hop relationships."""
        conn = self._get_connection()
        cursor = conn.cursor()
        results = []
        try:
            # Clean query for FTS
            words = [w for w in query.split() if len(w) > 1]
            if not words:
                return []
            fts_query = " OR ".join(words[:8])

            cursor.execute(
                "SELECT e.* FROM kg_entities_fts fts "
                "JOIN kg_entities e ON fts.rowid = e.id "
                "WHERE kg_entities_fts MATCH ? "
                "ORDER BY e.frequency DESC LIMIT ?",
                (fts_query, limit),
            )
            for row in cursor.fetchall():
                entity = dict(row)
                # Get relationships where this entity is subject
                cursor.execute(
                    "SELECT r.predicate, e2.name as object_name, e2.entity_type as object_type "
                    "FROM kg_relationships r JOIN kg_entities e2 ON r.object_id = e2.id "
                    "WHERE r.subject_id = ? ORDER BY r.frequency DESC LIMIT 5",
                    (entity["id"],),
                )
                entity["relationships"] = [dict(r) for r in cursor.fetchall()]
                results.append(entity)
        except Exception:
            pass
        finally:
            cursor.close()
        return results

    def get_entity_context(self, query: str, max_chars: int = 800) -> Optional[str]:
        """Get formatted entity context for prompt injection."""
        entities = self.get_related_entities(query, limit=5)
        if not entities:
            return None

        parts = []
        total = 0
        for e in entities:
            line = f"- {e['name']} ({e['entity_type']})"
            if e.get("description"):
                line += f": {e['description']}"
            rels = e.get("relationships", [])
            if rels:
                rel_strs = [f"{r['predicate']} {r['object_name']}" for r in rels[:3]]
                line += f" [{', '.join(rel_strs)}]"
            if total + len(line) > max_chars:
                break
            parts.append(line)
            total += len(line)

        return "\n".join(parts) if parts else None

    def process_interaction(
        self, user_input: str, response_text: str, neuron_id: Optional[str] = None
    ):
        """Extract entities from a full interaction and store them."""
        combined = f"User: {user_input}\nAssistant: {response_text}"
        extracted = self.extract_entities(combined)
        if extracted.get("entities") or extracted.get("relationships"):
            self.store_entities(
                extracted.get("entities", []),
                extracted.get("relationships", []),
                neuron_id=neuron_id,
            )

    # ---- Query ----

    def get_entities(
        self, entity_type: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all entities, optionally filtered by type."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            if entity_type:
                cursor.execute(
                    "SELECT * FROM kg_entities WHERE entity_type = ? ORDER BY frequency DESC LIMIT ?",
                    (entity_type, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM kg_entities ORDER BY frequency DESC LIMIT ?",
                    (limit,),
                )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def search_entities(self, query: str, entity_type: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Search entities by name with optional type filter."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            words = [w for w in query.split() if len(w) > 1]
            if not words:
                return []
            fts_query = " OR ".join(words[:8])

            if entity_type:
                cursor.execute(
                    "SELECT e.* FROM kg_entities_fts fts "
                    "JOIN kg_entities e ON fts.rowid = e.id "
                    "WHERE kg_entities_fts MATCH ? AND e.entity_type = ? "
                    "ORDER BY e.frequency DESC LIMIT ?",
                    (fts_query, entity_type, limit),
                )
            else:
                cursor.execute(
                    "SELECT e.* FROM kg_entities_fts fts "
                    "JOIN kg_entities e ON fts.rowid = e.id "
                    "WHERE kg_entities_fts MATCH ? "
                    "ORDER BY e.frequency DESC LIMIT ?",
                    (fts_query, limit),
                )
            return [dict(row) for row in cursor.fetchall()]
        except Exception:
            return []
        finally:
            cursor.close()

    # ---- Stats ----

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) as cnt FROM kg_entities")
            entity_count = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM kg_relationships")
            rel_count = cursor.fetchone()["cnt"]

            cursor.execute(
                "SELECT name, entity_type, frequency FROM kg_entities ORDER BY frequency DESC LIMIT 10"
            )
            top_entities = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                "SELECT entity_type, COUNT(*) as cnt FROM kg_entities GROUP BY entity_type"
            )
            type_distribution = {row["entity_type"]: row["cnt"] for row in cursor.fetchall()}

            return {
                "enabled": True,
                "version": "8.0",
                "entity_count": entity_count,
                "relationship_count": rel_count,
                "top_entities": top_entities,
                "type_distribution": type_distribution,
            }
        finally:
            cursor.close()
