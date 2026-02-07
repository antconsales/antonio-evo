"""
Emotional Memory System for Antonio Evo (v2.1).

Tracks user sentiment over time and adapts responses accordingly:
- Detects emotional states from user messages
- Maintains emotional history per session and globally
- Provides tone adaptation recommendations
- Remembers if user was stressed/happy/frustrated
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import time
import re
import sqlite3
from collections import defaultdict


class UserEmotionalState(Enum):
    """User emotional states - distinct from Antonio's Mood."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    STRESSED = "stressed"
    FRUSTRATED = "frustrated"
    ANXIOUS = "anxious"
    SAD = "sad"
    ANGRY = "angry"
    CALM = "calm"
    CONFUSED = "confused"
    TIRED = "tired"


class ToneRecommendation(Enum):
    """Recommended tone adjustments based on emotional context."""
    NORMAL = "normal"           # Standard response
    SUPPORTIVE = "supportive"   # User seems stressed/sad
    PATIENT = "patient"         # User seems frustrated/confused
    ENTHUSIASTIC = "enthusiastic"  # User is happy/excited
    CALMING = "calming"         # User seems anxious/angry
    CONCISE = "concise"         # User seems tired/impatient
    ENCOURAGING = "encouraging" # User seems down


@dataclass
class EmotionalSignal:
    """A single emotional signal detected from a message."""
    state: UserEmotionalState
    confidence: float  # 0.0 to 1.0
    indicators: List[str]  # Keywords/patterns that triggered this
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "confidence": self.confidence,
            "indicators": self.indicators,
            "timestamp": self.timestamp,
        }


@dataclass
class EmotionalContext:
    """
    Aggregated emotional context for response adaptation.

    Provides:
    - Current detected emotion
    - Session emotional trend
    - Historical patterns
    - Tone recommendations
    """
    current_state: UserEmotionalState = UserEmotionalState.NEUTRAL
    current_confidence: float = 0.0
    session_emotions: List[EmotionalSignal] = field(default_factory=list)
    dominant_session_emotion: Optional[UserEmotionalState] = None
    emotional_trend: str = "stable"  # "improving", "declining", "stable", "volatile"
    tone_recommendation: ToneRecommendation = ToneRecommendation.NORMAL
    adaptation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_state": self.current_state.value,
            "current_confidence": self.current_confidence,
            "session_emotions": [e.to_dict() for e in self.session_emotions[-5:]],  # Last 5
            "dominant_session_emotion": self.dominant_session_emotion.value if self.dominant_session_emotion else None,
            "emotional_trend": self.emotional_trend,
            "tone_recommendation": self.tone_recommendation.value,
            "adaptation_notes": self.adaptation_notes,
        }


class EmotionalAnalyzer:
    """
    Analyzes text for emotional indicators.

    Uses pattern matching and keyword analysis - no external dependencies.
    Future: Can be enhanced with sentiment models.
    """

    # Emotional indicators (keywords and patterns)
    EMOTIONAL_PATTERNS: Dict[UserEmotionalState, Dict[str, Any]] = {
        UserEmotionalState.HAPPY: {
            "keywords": [
                "grazie", "fantastico", "perfetto", "ottimo", "bene", "bellissimo",
                "thanks", "great", "perfect", "awesome", "excellent", "amazing",
                "love it", "wonderful", "fantastic", ":)", ":-)", "haha", "lol"
            ],
            "patterns": [
                r"!\s*$",  # Ends with exclamation
                r"(?:molto|really|so)\s+(?:bene|good|happy|contento)",
            ],
            "weight": 1.0,
        },
        UserEmotionalState.EXCITED: {
            "keywords": [
                "wow", "incredibile", "non ci credo", "finalmente",
                "omg", "amazing", "finally", "can't wait", "so excited"
            ],
            "patterns": [
                r"!{2,}",  # Multiple exclamation marks
                r"(?:non vedo l'ora|can't wait)",
            ],
            "weight": 1.0,
        },
        UserEmotionalState.STRESSED: {
            "keywords": [
                "urgente", "fretta", "deadline", "asap", "subito",
                "urgent", "hurry", "quickly", "emergency", "panic",
                "stress", "stressato", "overwhelmed", "troppo"
            ],
            "patterns": [
                r"(?:devo|must|need to).*(?:subito|now|immediately)",
                r"(?:non ho|no)\s+(?:tempo|time)",
            ],
            "weight": 1.2,
        },
        UserEmotionalState.FRUSTRATED: {
            "keywords": [
                "ancora", "di nuovo", "non funziona", "perche",
                "again", "still", "doesn't work", "broken", "why",
                "annoyed", "annoying", "frustrated", "uffa", "argh"
            ],
            "patterns": [
                r"(?:non|doesn't|won't|can't).*(?:funziona|work|capisco|understand)",
                r"\?{2,}",  # Multiple question marks
                r"(?:ma|but).*(?:perche|why)",
            ],
            "weight": 1.3,
        },
        UserEmotionalState.ANXIOUS: {
            "keywords": [
                "preoccupato", "paura", "timore", "nervoso",
                "worried", "scared", "afraid", "nervous", "anxious",
                "what if", "hope", "spero"
            ],
            "patterns": [
                r"(?:e se|what if|cosa succede se)",
                r"(?:non sono sicuro|not sure|uncertain)",
            ],
            "weight": 1.1,
        },
        UserEmotionalState.SAD: {
            "keywords": [
                "triste", "peccato", "sfortunatamente", "purtroppo",
                "sad", "unfortunately", "too bad", "disappointed",
                ":(", ":-(", "sigh"
            ],
            "patterns": [
                r"(?:mi dispiace|sorry|peccato)",
                r"\.{3,}$",  # Trailing ellipsis
            ],
            "weight": 1.0,
        },
        UserEmotionalState.ANGRY: {
            "keywords": [
                "arrabbiato", "furioso", "inaccettabile", "ridicolo",
                "angry", "furious", "unacceptable", "ridiculous",
                "terrible", "worst", "hate"
            ],
            "patterns": [
                r"[A-Z]{3,}",  # CAPS LOCK
                r"!{3,}",  # Many exclamation marks
            ],
            "weight": 1.5,
        },
        UserEmotionalState.CONFUSED: {
            "keywords": [
                "confuso", "non capisco", "come", "cosa",
                "confused", "don't understand", "what", "how", "huh",
                "lost", "unclear"
            ],
            "patterns": [
                r"\?\s*\?",  # Double question marks
                r"(?:non|don't).*(?:capisco|understand|get)",
            ],
            "weight": 1.0,
        },
        UserEmotionalState.TIRED: {
            "keywords": [
                "stanco", "esausto", "lungo", "finalmente finito",
                "tired", "exhausted", "long day", "finally done",
                "sleepy", "drained"
            ],
            "patterns": [
                r"(?:che|what a).*(?:giornata|day)",
                r"\.{2,}$",  # Ellipsis suggesting fatigue
            ],
            "weight": 0.9,
        },
        UserEmotionalState.CALM: {
            "keywords": [
                "tranquillo", "ok", "va bene", "nessun problema",
                "okay", "fine", "no worries", "no rush", "whenever",
                "take your time"
            ],
            "patterns": [
                r"(?:quando puoi|when you can|no rush)",
            ],
            "weight": 0.8,
        },
    }

    # Tone mapping based on emotional state
    TONE_MAPPING: Dict[UserEmotionalState, ToneRecommendation] = {
        UserEmotionalState.NEUTRAL: ToneRecommendation.NORMAL,
        UserEmotionalState.HAPPY: ToneRecommendation.ENTHUSIASTIC,
        UserEmotionalState.EXCITED: ToneRecommendation.ENTHUSIASTIC,
        UserEmotionalState.STRESSED: ToneRecommendation.SUPPORTIVE,
        UserEmotionalState.FRUSTRATED: ToneRecommendation.PATIENT,
        UserEmotionalState.ANXIOUS: ToneRecommendation.CALMING,
        UserEmotionalState.SAD: ToneRecommendation.SUPPORTIVE,
        UserEmotionalState.ANGRY: ToneRecommendation.CALMING,
        UserEmotionalState.CONFUSED: ToneRecommendation.PATIENT,
        UserEmotionalState.TIRED: ToneRecommendation.CONCISE,
        UserEmotionalState.CALM: ToneRecommendation.NORMAL,
    }

    def analyze(self, text: str) -> EmotionalSignal:
        """
        Analyze text for emotional indicators.

        Args:
            text: User message text

        Returns:
            EmotionalSignal with detected state and confidence
        """
        text_lower = text.lower()
        scores: Dict[UserEmotionalState, Tuple[float, List[str]]] = defaultdict(lambda: (0.0, []))

        for state, config in self.EMOTIONAL_PATTERNS.items():
            weight = config["weight"]
            indicators = []
            score = 0.0

            # Check keywords
            for keyword in config["keywords"]:
                if keyword.lower() in text_lower:
                    score += 0.3 * weight
                    indicators.append(f"keyword:{keyword}")

            # Check patterns
            for pattern in config["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 0.4 * weight
                    indicators.append(f"pattern:{pattern[:20]}")

            if score > 0:
                scores[state] = (min(score, 1.0), indicators)

        # Find highest scoring emotion
        if scores:
            best_state = max(scores, key=lambda k: scores[k][0])
            confidence, indicators = scores[best_state]

            # Require minimum confidence
            if confidence >= 0.25:
                return EmotionalSignal(
                    state=best_state,
                    confidence=confidence,
                    indicators=indicators[:5],  # Limit indicators
                )

        # Default to neutral
        return EmotionalSignal(
            state=UserEmotionalState.NEUTRAL,
            confidence=0.5,
            indicators=[],
        )

    def get_tone_recommendation(
        self,
        current: EmotionalSignal,
        history: List[EmotionalSignal],
    ) -> Tuple[ToneRecommendation, List[str]]:
        """
        Get tone recommendation based on current emotion and history.

        Args:
            current: Current emotional signal
            history: Recent emotional history

        Returns:
            Tuple of (ToneRecommendation, adaptation_notes)
        """
        notes = []
        base_tone = self.TONE_MAPPING.get(current.state, ToneRecommendation.NORMAL)

        # Check for emotional patterns in history
        if history:
            recent_states = [h.state for h in history[-5:]]

            # Persistent frustration
            frustration_count = sum(1 for s in recent_states if s == UserEmotionalState.FRUSTRATED)
            if frustration_count >= 2:
                notes.append("User showing persistent frustration - be extra patient")
                return ToneRecommendation.PATIENT, notes

            # Escalating stress
            stress_count = sum(1 for s in recent_states if s in [
                UserEmotionalState.STRESSED, UserEmotionalState.ANXIOUS
            ])
            if stress_count >= 2:
                notes.append("User under sustained stress - be supportive and efficient")
                return ToneRecommendation.SUPPORTIVE, notes

            # Improving mood
            if len(recent_states) >= 3:
                positive = [UserEmotionalState.HAPPY, UserEmotionalState.EXCITED, UserEmotionalState.CALM]
                if recent_states[-1] in positive and recent_states[0] not in positive:
                    notes.append("User mood improving - maintain positive momentum")

        # Context-specific adjustments
        if current.state == UserEmotionalState.CONFUSED:
            notes.append("User confused - provide clear step-by-step guidance")
        elif current.state == UserEmotionalState.TIRED:
            notes.append("User seems tired - keep responses concise")
        elif current.state == UserEmotionalState.FRUSTRATED:
            notes.append("User frustrated - acknowledge the difficulty, be solution-focused")

        return base_tone, notes


class EmotionalMemory:
    """
    Persistent emotional memory storage and retrieval.

    Extends the existing MemoryStorage with emotional tracking.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "data/evomemory.db"):
        """
        Initialize emotional memory with shared database.

        Args:
            db_path: Path to SQLite database (shared with main memory)
        """
        self.db_path = db_path
        self.analyzer = EmotionalAnalyzer()
        self._init_emotional_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_emotional_schema(self):
        """Initialize emotional memory tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Emotional signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emotional_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    state TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    indicators TEXT,
                    message_preview TEXT,
                    created_at REAL NOT NULL
                )
            """)

            # Index for session queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_emotional_session
                ON emotional_signals(session_id, created_at DESC)
            """)

            # Index for time-based queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_emotional_time
                ON emotional_signals(created_at DESC)
            """)

            # Emotional patterns table (aggregated insights)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emotional_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    state TEXT,
                    time_of_day TEXT,
                    day_of_week INTEGER,
                    frequency INTEGER DEFAULT 1,
                    last_seen REAL NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

            conn.commit()

    def record_emotion(
        self,
        message: str,
        session_id: Optional[str] = None,
        override_state: Optional[UserEmotionalState] = None,
    ) -> EmotionalSignal:
        """
        Analyze and record emotional state from a message.

        Args:
            message: User message text
            session_id: Current session ID
            override_state: Optional override (e.g., from explicit user input)

        Returns:
            Detected EmotionalSignal
        """
        # Analyze message
        if override_state:
            signal = EmotionalSignal(
                state=override_state,
                confidence=1.0,
                indicators=["user_explicit"],
            )
        else:
            signal = self.analyzer.analyze(message)

        # Store in database
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO emotional_signals
                (session_id, state, confidence, indicators, message_preview, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                signal.state.value,
                signal.confidence,
                ",".join(signal.indicators) if signal.indicators else None,
                message[:100] if message else None,  # Preview only
                signal.timestamp,
            ))
            conn.commit()

        return signal

    def get_session_emotions(
        self,
        session_id: str,
        limit: int = 20,
    ) -> List[EmotionalSignal]:
        """
        Get emotional history for a session.

        Args:
            session_id: Session ID
            limit: Maximum signals to return

        Returns:
            List of EmotionalSignal objects, newest first
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT state, confidence, indicators, created_at
                FROM emotional_signals
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (session_id, limit))

            signals = []
            for row in cursor.fetchall():
                signals.append(EmotionalSignal(
                    state=UserEmotionalState(row["state"]),
                    confidence=row["confidence"],
                    indicators=row["indicators"].split(",") if row["indicators"] else [],
                    timestamp=row["created_at"],
                ))

            return signals

    def get_recent_emotions(
        self,
        hours: int = 24,
        limit: int = 50,
    ) -> List[EmotionalSignal]:
        """
        Get recent emotional signals across all sessions.

        Args:
            hours: How many hours back to look
            limit: Maximum signals to return

        Returns:
            List of EmotionalSignal objects
        """
        cutoff = time.time() - (hours * 3600)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT state, confidence, indicators, created_at
                FROM emotional_signals
                WHERE created_at >= ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (cutoff, limit))

            return [
                EmotionalSignal(
                    state=UserEmotionalState(row["state"]),
                    confidence=row["confidence"],
                    indicators=row["indicators"].split(",") if row["indicators"] else [],
                    timestamp=row["created_at"],
                )
                for row in cursor.fetchall()
            ]

    def get_emotional_context(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> EmotionalContext:
        """
        Get full emotional context for response adaptation.

        This is the main entry point for the pipeline.

        Args:
            message: Current user message
            session_id: Current session ID

        Returns:
            EmotionalContext with recommendations
        """
        # Analyze current message
        current_signal = self.record_emotion(message, session_id)

        # Get session history
        session_emotions = []
        if session_id:
            session_emotions = self.get_session_emotions(session_id, limit=10)

        # Calculate session dominant emotion
        dominant_session = self._calculate_dominant_emotion(session_emotions)

        # Calculate emotional trend
        trend = self._calculate_trend(session_emotions)

        # Get tone recommendation
        tone, notes = self.analyzer.get_tone_recommendation(
            current_signal,
            session_emotions,
        )

        return EmotionalContext(
            current_state=current_signal.state,
            current_confidence=current_signal.confidence,
            session_emotions=session_emotions,
            dominant_session_emotion=dominant_session,
            emotional_trend=trend,
            tone_recommendation=tone,
            adaptation_notes=notes,
        )

    def _calculate_dominant_emotion(
        self,
        emotions: List[EmotionalSignal],
    ) -> Optional[UserEmotionalState]:
        """Calculate most frequent emotion in a list."""
        if not emotions:
            return None

        counts: Dict[UserEmotionalState, float] = defaultdict(float)
        for e in emotions:
            # Weight by confidence and recency
            counts[e.state] += e.confidence

        if counts:
            return max(counts, key=counts.get)
        return None

    def _calculate_trend(
        self,
        emotions: List[EmotionalSignal],
    ) -> str:
        """Calculate emotional trend from history."""
        if len(emotions) < 3:
            return "stable"

        # Define positive/negative states
        positive = {UserEmotionalState.HAPPY, UserEmotionalState.EXCITED, UserEmotionalState.CALM}
        negative = {UserEmotionalState.STRESSED, UserEmotionalState.FRUSTRATED,
                   UserEmotionalState.ANXIOUS, UserEmotionalState.ANGRY, UserEmotionalState.SAD}

        # Score recent vs older emotions
        recent = emotions[:3]
        older = emotions[3:6] if len(emotions) > 3 else emotions[3:]

        def score_group(group: List[EmotionalSignal]) -> float:
            total = 0.0
            for e in group:
                if e.state in positive:
                    total += e.confidence
                elif e.state in negative:
                    total -= e.confidence
            return total / len(group) if group else 0

        recent_score = score_group(recent)
        older_score = score_group(older) if older else 0

        diff = recent_score - older_score

        # Check for volatility
        unique_states = len(set(e.state for e in emotions[:5]))
        if unique_states >= 4:
            return "volatile"

        if diff > 0.3:
            return "improving"
        elif diff < -0.3:
            return "declining"
        return "stable"

    def get_stats(self) -> Dict[str, Any]:
        """Get emotional memory statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total signals
            cursor.execute("SELECT COUNT(*) FROM emotional_signals")
            total = cursor.fetchone()[0]

            # State distribution (last 7 days)
            week_ago = time.time() - (7 * 24 * 3600)
            cursor.execute("""
                SELECT state, COUNT(*) as count
                FROM emotional_signals
                WHERE created_at >= ?
                GROUP BY state
                ORDER BY count DESC
            """, (week_ago,))

            distribution = {row["state"]: row["count"] for row in cursor.fetchall()}

            # Average confidence
            cursor.execute("""
                SELECT AVG(confidence) FROM emotional_signals
                WHERE created_at >= ?
            """, (week_ago,))
            avg_confidence = cursor.fetchone()[0] or 0.0

            return {
                "total_signals": total,
                "weekly_distribution": distribution,
                "avg_confidence": round(avg_confidence, 2),
            }
