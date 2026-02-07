"""
SQLite storage layer for EvoMemory.

Handles:
- Database initialization and schema
- CRUD operations for neurons
- FTS5 full-text search index
- Preference storage
- Session tracking
"""

import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

from .neuron import (
    Neuron,
    NeuronCreate,
    Mood,
    Persona,
    UserPreference,
)


class MemoryStorage:
    """
    SQLite-based storage for EvoMemory.

    Thread-safe with connection pooling.
    Uses FTS5 for full-text search (BM25).
    """

    # Schema version for migrations
    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "data/evomemory.db"):
        """
        Initialize storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()

        # Initialize schema
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection

    @contextmanager
    def _cursor(self):
        """Context manager for database cursor with auto-commit."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_schema(self):
        """Initialize database schema."""
        with self._cursor() as cursor:
            # Main neurons table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS neurons (
                    id TEXT PRIMARY KEY,
                    input TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    output TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    mood TEXT DEFAULT 'neutral',
                    handler TEXT NOT NULL,
                    persona TEXT DEFAULT 'unknown',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL,
                    session_id TEXT,
                    request_id TEXT,
                    classification_domain TEXT
                )
            """)

            # Indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_neurons_input_hash
                ON neurons(input_hash)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_neurons_confidence
                ON neurons(confidence DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_neurons_created
                ON neurons(created_at DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_neurons_session
                ON neurons(session_id)
            """)

            # FTS5 virtual table for full-text search
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS neurons_fts USING fts5(
                    input,
                    output,
                    content='neurons',
                    content_rowid='rowid'
                )
            """)

            # Triggers to keep FTS in sync
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS neurons_ai AFTER INSERT ON neurons BEGIN
                    INSERT INTO neurons_fts(rowid, input, output)
                    VALUES (NEW.rowid, NEW.input, NEW.output);
                END
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS neurons_ad AFTER DELETE ON neurons BEGIN
                    INSERT INTO neurons_fts(neurons_fts, rowid, input, output)
                    VALUES ('delete', OLD.rowid, OLD.input, OLD.output);
                END
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS neurons_au AFTER UPDATE ON neurons BEGIN
                    INSERT INTO neurons_fts(neurons_fts, rowid, input, output)
                    VALUES ('delete', OLD.rowid, OLD.input, OLD.output);
                    INSERT INTO neurons_fts(rowid, input, output)
                    VALUES (NEW.rowid, NEW.input, NEW.output);
                END
            """)

            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    learned_from TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    neuron_count INTEGER DEFAULT 0,
                    avg_confidence REAL,
                    dominant_mood TEXT
                )
            """)

            # Schema version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            cursor.execute("""
                INSERT OR REPLACE INTO schema_info (key, value)
                VALUES ('version', ?)
            """, (str(self.SCHEMA_VERSION),))

    # ===================
    # NEURON OPERATIONS
    # ===================

    def store(self, neuron_create: NeuronCreate) -> Neuron:
        """
        Store a new neuron.

        Args:
            neuron_create: Data for new neuron

        Returns:
            Created Neuron with generated fields
        """
        neuron = neuron_create.to_neuron()

        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO neurons (
                    id, input, input_hash, output, confidence, mood,
                    handler, persona, created_at, updated_at,
                    access_count, last_accessed, session_id,
                    request_id, classification_domain
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                neuron.id,
                neuron.input,
                neuron.input_hash,
                neuron.output,
                neuron.confidence,
                neuron.mood.value,
                neuron.handler,
                neuron.persona.value,
                neuron.created_at,
                neuron.updated_at,
                neuron.access_count,
                neuron.last_accessed,
                neuron.session_id,
                neuron.request_id,
                neuron.classification_domain,
            ))

        return neuron

    def get(self, neuron_id: str) -> Optional[Neuron]:
        """
        Get a neuron by ID.

        Args:
            neuron_id: Neuron ID

        Returns:
            Neuron if found, None otherwise
        """
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT * FROM neurons WHERE id = ?
            """, (neuron_id,))
            row = cursor.fetchone()

            if row:
                return self._row_to_neuron(row)
            return None

    def get_by_input_hash(self, input_hash: str) -> Optional[Neuron]:
        """
        Get a neuron by input hash (exact match lookup).

        Args:
            input_hash: SHA256 hash of input text

        Returns:
            Neuron if found, None otherwise
        """
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT * FROM neurons
                WHERE input_hash = ?
                ORDER BY confidence DESC
                LIMIT 1
            """, (input_hash,))
            row = cursor.fetchone()

            if row:
                return self._row_to_neuron(row)
            return None

    def search_bm25(
        self,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Tuple[Neuron, float]]:
        """
        Search neurons using BM25 full-text search.

        Args:
            query: Search query
            limit: Maximum results
            min_confidence: Minimum confidence threshold

        Returns:
            List of (Neuron, bm25_score) tuples, highest score first
        """
        with self._cursor() as cursor:
            # BM25 search with ranking
            cursor.execute("""
                SELECT
                    n.*,
                    bm25(neurons_fts) as score
                FROM neurons_fts
                JOIN neurons n ON neurons_fts.rowid = n.rowid
                WHERE neurons_fts MATCH ?
                AND n.confidence >= ?
                ORDER BY score
                LIMIT ?
            """, (query, min_confidence, limit))

            results = []
            for row in cursor.fetchall():
                neuron = self._row_to_neuron(row)
                # BM25 returns negative scores, lower is better
                # Convert to positive score where higher is better
                score = -row["score"] if row["score"] else 0.0
                results.append((neuron, score))

            return results

    def update_confidence(
        self,
        neuron_id: str,
        new_confidence: float,
    ) -> bool:
        """
        Update a neuron's confidence score.

        Args:
            neuron_id: Neuron ID
            new_confidence: New confidence value (0.0-1.0)

        Returns:
            True if updated, False if not found
        """
        new_confidence = max(0.0, min(1.0, new_confidence))

        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE neurons
                SET confidence = ?, updated_at = ?
                WHERE id = ?
            """, (new_confidence, time.time(), neuron_id))

            return cursor.rowcount > 0

    def increment_access(self, neuron_id: str, confidence_boost: float = 0.02) -> bool:
        """
        Increment access count and optionally boost confidence.

        Called when a neuron is retrieved and used.

        Args:
            neuron_id: Neuron ID
            confidence_boost: Amount to boost confidence (capped at 1.0)

        Returns:
            True if updated, False if not found
        """
        now = time.time()

        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE neurons
                SET
                    access_count = access_count + 1,
                    last_accessed = ?,
                    confidence = MIN(1.0, confidence + ?),
                    updated_at = ?
                WHERE id = ?
            """, (now, confidence_boost, now, neuron_id))

            return cursor.rowcount > 0

    def get_recent(self, limit: int = 10) -> List[Neuron]:
        """
        Get most recent neurons.

        Args:
            limit: Maximum results

        Returns:
            List of neurons, newest first
        """
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT * FROM neurons
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            return [self._row_to_neuron(row) for row in cursor.fetchall()]

    def get_by_session(self, session_id: str) -> List[Neuron]:
        """
        Get all neurons in a session.

        Args:
            session_id: Session ID

        Returns:
            List of neurons in session order
        """
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT * FROM neurons
                WHERE session_id = ?
                ORDER BY created_at ASC
            """, (session_id,))

            return [self._row_to_neuron(row) for row in cursor.fetchall()]

    def delete(self, neuron_id: str) -> bool:
        """
        Delete a neuron.

        Args:
            neuron_id: Neuron ID

        Returns:
            True if deleted, False if not found
        """
        with self._cursor() as cursor:
            cursor.execute("""
                DELETE FROM neurons WHERE id = ?
            """, (neuron_id,))

            return cursor.rowcount > 0

    def count(self) -> int:
        """Get total neuron count."""
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM neurons")
            return cursor.fetchone()[0]

    def cleanup_old_low_confidence(
        self,
        max_neurons: int = 100000,
        min_confidence: float = 0.3,
        days_old: int = 30,
    ) -> int:
        """
        Clean up old, low-confidence neurons.

        Keeps the database size manageable.

        Args:
            max_neurons: Maximum neurons to keep
            min_confidence: Delete neurons below this confidence
            days_old: Only delete neurons older than this

        Returns:
            Number of deleted neurons
        """
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)

        with self._cursor() as cursor:
            # First, check if we need cleanup
            cursor.execute("SELECT COUNT(*) FROM neurons")
            total = cursor.fetchone()[0]

            if total <= max_neurons:
                return 0

            # Delete old, low-confidence neurons
            cursor.execute("""
                DELETE FROM neurons
                WHERE confidence < ?
                AND created_at < ?
                AND id NOT IN (
                    SELECT id FROM neurons
                    ORDER BY confidence DESC, access_count DESC
                    LIMIT ?
                )
            """, (min_confidence, cutoff_time, max_neurons))

            return cursor.rowcount

    # ===================
    # PREFERENCE OPERATIONS
    # ===================

    def set_preference(
        self,
        key: str,
        value: str,
        confidence: float = 0.5,
        learned_from: Optional[str] = None,
    ) -> UserPreference:
        """
        Set or update a user preference.

        Args:
            key: Preference key (e.g., "persona_code")
            value: Preference value (e.g., "LOGIC")
            confidence: Confidence in this preference
            learned_from: Neuron ID that established this

        Returns:
            Created/updated preference
        """
        now = time.time()

        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO preferences (key, value, confidence, learned_from, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    confidence = excluded.confidence,
                    learned_from = excluded.learned_from,
                    updated_at = excluded.updated_at
            """, (key, value, confidence, learned_from, now, now))

        return UserPreference(
            key=key,
            value=value,
            confidence=confidence,
            learned_from=learned_from,
            created_at=now,
        )

    def get_preference(self, key: str) -> Optional[UserPreference]:
        """
        Get a user preference.

        Args:
            key: Preference key

        Returns:
            Preference if found, None otherwise
        """
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT * FROM preferences WHERE key = ?
            """, (key,))
            row = cursor.fetchone()

            if row:
                return UserPreference(
                    key=row["key"],
                    value=row["value"],
                    confidence=row["confidence"],
                    learned_from=row["learned_from"],
                    created_at=row["created_at"],
                )
            return None

    def get_all_preferences(self) -> Dict[str, UserPreference]:
        """
        Get all user preferences.

        Returns:
            Dictionary of key -> UserPreference
        """
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM preferences")

            return {
                row["key"]: UserPreference(
                    key=row["key"],
                    value=row["value"],
                    confidence=row["confidence"],
                    learned_from=row["learned_from"],
                    created_at=row["created_at"],
                )
                for row in cursor.fetchall()
            }

    # ===================
    # SESSION OPERATIONS
    # ===================

    def create_session(self, session_id: str) -> None:
        """Create a new session."""
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT OR IGNORE INTO sessions (id, started_at, neuron_count)
                VALUES (?, ?, 0)
            """, (session_id, time.time()))

    def end_session(self, session_id: str) -> None:
        """End a session and compute stats."""
        with self._cursor() as cursor:
            # Compute session stats
            cursor.execute("""
                SELECT
                    COUNT(*) as neuron_count,
                    AVG(confidence) as avg_confidence,
                    mood
                FROM neurons
                WHERE session_id = ?
                GROUP BY mood
                ORDER BY COUNT(*) DESC
                LIMIT 1
            """, (session_id,))
            row = cursor.fetchone()

            if row:
                cursor.execute("""
                    UPDATE sessions
                    SET
                        ended_at = ?,
                        neuron_count = ?,
                        avg_confidence = ?,
                        dominant_mood = ?
                    WHERE id = ?
                """, (
                    time.time(),
                    row["neuron_count"],
                    row["avg_confidence"],
                    row["mood"],
                    session_id,
                ))

    # ===================
    # STATS
    # ===================

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_neurons,
                    AVG(confidence) as avg_confidence,
                    SUM(access_count) as total_accesses,
                    MIN(created_at) as oldest_neuron,
                    MAX(created_at) as newest_neuron
                FROM neurons
            """)
            row = cursor.fetchone()

            cursor.execute("SELECT COUNT(*) FROM preferences")
            pref_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM sessions")
            session_count = cursor.fetchone()[0]

            return {
                "total_neurons": row["total_neurons"] or 0,
                "avg_confidence": row["avg_confidence"] or 0.0,
                "total_accesses": row["total_accesses"] or 0,
                "oldest_neuron": row["oldest_neuron"],
                "newest_neuron": row["newest_neuron"],
                "preference_count": pref_count,
                "session_count": session_count,
            }

    # ===================
    # HELPERS
    # ===================

    def _row_to_neuron(self, row: sqlite3.Row) -> Neuron:
        """Convert database row to Neuron object."""
        return Neuron(
            id=row["id"],
            input=row["input"],
            input_hash=row["input_hash"],
            output=row["output"],
            confidence=row["confidence"],
            mood=Mood(row["mood"]) if row["mood"] else Mood.NEUTRAL,
            handler=row["handler"],
            persona=Persona(row["persona"]) if row["persona"] else Persona.UNKNOWN,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"] or 0,
            last_accessed=row["last_accessed"],
            session_id=row["session_id"],
            request_id=row["request_id"],
            classification_domain=row["classification_domain"],
        )

    def close(self):
        """Close database connection."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
