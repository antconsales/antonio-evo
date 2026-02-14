"""
Self-Improvement Engine (v8.0) — Track response quality, detect failure patterns,
auto-generate prompt improvements.

Tracks:
- Explicit feedback (thumbs up/down from UI)
- Implicit signals (retry detection, clarification requests, topic changes)
- Error patterns (recurring failures by handler/domain)
- Prompt improvement suggestions (LLM-generated from failure data)
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Any
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


@dataclass
class OutcomeRecord:
    """A recorded outcome for a neuron."""
    id: int
    neuron_id: Optional[str]
    feedback_type: str  # explicit_positive, explicit_negative, implicit_retry, implicit_clarification, implicit_followup, implicit_gratitude
    signal_value: float  # -1.0 to 1.0
    user_message: str
    original_query: Optional[str]
    original_response: Optional[str]
    session_id: Optional[str]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "neuron_id": self.neuron_id,
            "feedback_type": self.feedback_type,
            "signal_value": self.signal_value,
            "user_message": self.user_message[:200],
            "timestamp": self.timestamp,
        }


@dataclass
class PromptImprovement:
    """A suggested prompt improvement."""
    id: int
    target_area: str  # system_prompt, tool_instructions, persona
    suggestion: str
    reasoning: str
    confidence: float
    status: str  # pending, applied, rejected
    source_outcomes: str  # JSON list of outcome IDs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "target_area": self.target_area,
            "suggestion": self.suggestion,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "status": self.status,
        }


class SelfImprovementEngine:
    """
    Tracks response quality and generates improvement suggestions.

    SQLite tables:
    - outcome_tracking: feedback records (explicit + implicit signals)
    - prompt_improvements: LLM-generated prompt suggestions
    - error_patterns: recurring error tracking
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
        self._last_interaction = {}  # session_id -> {query, response}
        self._init_schema()

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
                CREATE TABLE IF NOT EXISTS outcome_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    neuron_id TEXT,
                    feedback_type TEXT NOT NULL,
                    signal_value REAL DEFAULT 0,
                    user_message TEXT,
                    original_query TEXT,
                    original_response TEXT,
                    session_id TEXT,
                    timestamp REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcome_session
                ON outcome_tracking(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcome_type
                ON outcome_tracking(feedback_type)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompt_improvements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_area TEXT NOT NULL,
                    suggestion TEXT NOT NULL,
                    reasoning TEXT,
                    confidence REAL DEFAULT 0.5,
                    status TEXT DEFAULT 'pending',
                    source_outcomes TEXT,
                    created_at REAL NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL,
                    error_message TEXT,
                    query_pattern TEXT,
                    occurrence_count INTEGER DEFAULT 1,
                    suggested_fix TEXT,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    UNIQUE(error_type, error_message)
                )
            """)

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    # ---- Explicit Feedback ----

    def record_explicit_feedback(
        self,
        neuron_id: Optional[str],
        feedback: str,
        session_id: Optional[str] = None,
    ):
        """
        Record explicit feedback (positive/negative) from user.

        Args:
            neuron_id: ID of the neuron being rated
            feedback: "positive" or "negative"
            session_id: Current session
        """
        signal = 1.0 if feedback == "positive" else -1.0
        feedback_type = f"explicit_{feedback}"

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO outcome_tracking (neuron_id, feedback_type, signal_value, user_message, session_id, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (neuron_id, feedback_type, signal, "", session_id, time.time()),
            )
            conn.commit()
        finally:
            cursor.close()

    # ---- Implicit Signal Detection ----

    def detect_implicit_signal(
        self,
        current_msg: str,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Detect implicit quality signals by comparing current message to previous interaction.

        Signals:
        - "retry": User rephrased same question (negative)
        - "clarification": User asked "what do you mean?" (negative)
        - "followup": User builds on response (positive)
        - "gratitude": User says thanks (positive)
        - "topic_change": Abrupt topic change (neutral/negative)

        Returns detected signal type or None.
        """
        if not session_id or not current_msg:
            return None

        prev = self._last_interaction.get(session_id)
        if not prev:
            return None

        prev_query = prev.get("query", "")
        prev_response = prev.get("response", "")
        msg_lower = current_msg.lower().strip()

        signal = None
        signal_value = 0.0

        # Check for gratitude
        gratitude_words = ["grazie", "thanks", "thank you", "perfetto", "great", "ottimo", "bravo"]
        if any(w in msg_lower for w in gratitude_words):
            signal = "implicit_gratitude"
            signal_value = 0.8

        # Check for clarification request
        elif any(p in msg_lower for p in [
            "cosa intendi", "what do you mean", "non capisco", "don't understand",
            "puoi spiegare", "can you explain", "che vuol dire",
        ]):
            signal = "implicit_clarification"
            signal_value = -0.5

        # Check for retry (similar query rephrased)
        elif prev_query and self._text_similarity(current_msg, prev_query) > 0.6:
            signal = "implicit_retry"
            signal_value = -0.7

        # Check for followup (builds on response)
        elif prev_response and any(w in msg_lower for w in [
            "e anche", "and also", "inoltre", "another", "un'altra",
            "in addition", "can you also", "puoi anche",
        ]):
            signal = "implicit_followup"
            signal_value = 0.5

        if signal:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO outcome_tracking (neuron_id, feedback_type, signal_value, "
                    "user_message, original_query, original_response, session_id, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        None, signal, signal_value,
                        current_msg[:500], prev_query[:500], prev_response[:500],
                        session_id, time.time(),
                    ),
                )
                conn.commit()
            finally:
                cursor.close()

        return signal

    def update_last_interaction(
        self, session_id: str, query: str, response: str
    ):
        """Update the last interaction for a session (called after response generation)."""
        self._last_interaction[session_id] = {
            "query": query,
            "response": response,
        }

    # ---- Error Tracking ----

    def record_error(self, error_type: str, error_message: str, query: str = ""):
        """Record an error occurrence for pattern analysis."""
        now = time.time()
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT id FROM error_patterns WHERE error_type = ? AND error_message = ?",
                (error_type, error_message[:200]),
            )
            row = cursor.fetchone()
            if row:
                cursor.execute(
                    "UPDATE error_patterns SET occurrence_count = occurrence_count + 1, last_seen = ? WHERE id = ?",
                    (now, row["id"]),
                )
            else:
                cursor.execute(
                    "INSERT INTO error_patterns (error_type, error_message, query_pattern, occurrence_count, first_seen, last_seen) "
                    "VALUES (?, ?, ?, 1, ?, ?)",
                    (error_type, error_message[:200], query[:200], now, now),
                )
            conn.commit()
        finally:
            cursor.close()

    # ---- Analysis ----

    def analyze_failures(self, days: int = 7) -> List[Dict[str, Any]]:
        """Analyze failure patterns from recent outcomes."""
        cutoff = time.time() - (days * 86400)
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT feedback_type, COUNT(*) as cnt, AVG(signal_value) as avg_signal "
                "FROM outcome_tracking WHERE timestamp > ? AND signal_value < 0 "
                "GROUP BY feedback_type ORDER BY cnt DESC",
                (cutoff,),
            )
            failures = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                "SELECT error_type, error_message, occurrence_count, suggested_fix "
                "FROM error_patterns WHERE last_seen > ? ORDER BY occurrence_count DESC LIMIT 10",
                (cutoff,),
            )
            errors = [dict(row) for row in cursor.fetchall()]

            return {"failure_signals": failures, "error_patterns": errors}
        finally:
            cursor.close()

    def generate_improvements(self, max_suggestions: int = 3) -> List[PromptImprovement]:
        """Generate prompt improvement suggestions from failure data using LLM."""
        failures = self.analyze_failures(days=7)
        if not failures.get("failure_signals") and not failures.get("error_patterns"):
            return []

        prompt = (
            "Based on these failure patterns, suggest prompt improvements.\n"
            "Return JSON array of improvements.\n"
            "Format: [{\"target_area\": \"system_prompt|tool_instructions|persona\", "
            "\"suggestion\": \"...\", \"reasoning\": \"...\", \"confidence\": 0.0-1.0}]\n\n"
            f"Failure signals: {json.dumps(failures.get('failure_signals', []))}\n"
            f"Error patterns: {json.dumps(failures.get('error_patterns', []))}\n"
        )

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 512},
                },
                timeout=30,
            )
            response.raise_for_status()
            text = response.json().get("response", "")
            parsed = json.loads(text)
            suggestions = parsed if isinstance(parsed, list) else parsed.get("improvements", [])
        except Exception as e:
            logger.debug(f"Improvement generation failed: {e}")
            return []

        # Store suggestions
        improvements = []
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            for s in suggestions[:max_suggestions]:
                cursor.execute(
                    "INSERT INTO prompt_improvements (target_area, suggestion, reasoning, confidence, status, source_outcomes, created_at) "
                    "VALUES (?, ?, ?, ?, 'pending', '[]', ?)",
                    (
                        s.get("target_area", "system_prompt"),
                        s.get("suggestion", ""),
                        s.get("reasoning", ""),
                        s.get("confidence", 0.5),
                        time.time(),
                    ),
                )
                improvements.append(PromptImprovement(
                    id=cursor.lastrowid,
                    target_area=s.get("target_area", "system_prompt"),
                    suggestion=s.get("suggestion", ""),
                    reasoning=s.get("reasoning", ""),
                    confidence=s.get("confidence", 0.5),
                    status="pending",
                    source_outcomes="[]",
                ))
            conn.commit()
        finally:
            cursor.close()

        return improvements

    # ---- Improvement Management ----

    def get_pending_improvements(self) -> List[PromptImprovement]:
        """Get pending improvement suggestions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM prompt_improvements WHERE status = 'pending' ORDER BY confidence DESC"
            )
            return [
                PromptImprovement(
                    id=row["id"],
                    target_area=row["target_area"],
                    suggestion=row["suggestion"],
                    reasoning=row["reasoning"],
                    confidence=row["confidence"],
                    status=row["status"],
                    source_outcomes=row["source_outcomes"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            cursor.close()

    def get_active_improvements(self) -> List[PromptImprovement]:
        """Get applied improvements for prompt injection."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM prompt_improvements WHERE status = 'applied' ORDER BY confidence DESC"
            )
            return [
                PromptImprovement(
                    id=row["id"],
                    target_area=row["target_area"],
                    suggestion=row["suggestion"],
                    reasoning=row["reasoning"],
                    confidence=row["confidence"],
                    status=row["status"],
                    source_outcomes=row["source_outcomes"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            cursor.close()

    def apply_improvement(self, improvement_id: int) -> bool:
        """Mark an improvement as applied."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE prompt_improvements SET status = 'applied' WHERE id = ?",
                (improvement_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()

    def reject_improvement(self, improvement_id: int) -> bool:
        """Mark an improvement as rejected."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE prompt_improvements SET status = 'rejected' WHERE id = ?",
                (improvement_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()

    # ---- Stats ----

    def get_stats(self) -> Dict[str, Any]:
        """Get self-improvement statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) as cnt FROM outcome_tracking")
            total_outcomes = cursor.fetchone()["cnt"]

            cursor.execute(
                "SELECT COUNT(*) as cnt FROM outcome_tracking WHERE signal_value > 0"
            )
            positive = cursor.fetchone()["cnt"]

            cursor.execute(
                "SELECT COUNT(*) as cnt FROM outcome_tracking WHERE signal_value < 0"
            )
            negative = cursor.fetchone()["cnt"]

            cursor.execute(
                "SELECT COUNT(*) as cnt FROM prompt_improvements WHERE status = 'applied'"
            )
            applied = cursor.fetchone()["cnt"]

            cursor.execute(
                "SELECT COUNT(*) as cnt FROM prompt_improvements WHERE status = 'pending'"
            )
            pending = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM error_patterns")
            error_count = cursor.fetchone()["cnt"]

            return {
                "enabled": True,
                "version": "8.0",
                "total_outcomes": total_outcomes,
                "positive_signals": positive,
                "negative_signals": negative,
                "success_rate": positive / max(total_outcomes, 1),
                "applied_improvements": applied,
                "pending_improvements": pending,
                "error_patterns": error_count,
            }
        finally:
            cursor.close()

    # ---- Helpers ----

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """Quick similarity check between two texts."""
        return SequenceMatcher(None, text1.lower()[:200], text2.lower()[:200]).ratio()
