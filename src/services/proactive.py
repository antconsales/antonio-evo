"""
Proactive Mode Service for Antonio Evo (v3.1).

Per Antonio Evo Unified Spec (v3.1):
- You do not interrupt
- You do not recommend
- You surface observations ONLY

Proactive insights require:
- Repeated patterns
- Sufficient historical data
- Confidence threshold met (>=0.7)

Every proactive message MUST include:
1. Pattern detected
2. Evidence summary
3. Confidence score
4. Reason for surfacing

MANDATORY FOOTER: "This is an observation, not an instruction."

Philosophy: Observe, learn, assist - but never intrude.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import sqlite3
import json
import logging

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns we can detect."""
    TIME_OF_DAY = "time_of_day"       # Late night work, early mornings
    TOPIC_FREQUENCY = "topic_frequency"  # Frequent topics
    EMOTIONAL_TREND = "emotional_trend"  # Recurring emotional states
    QUERY_FREQUENCY = "query_frequency"  # How often user interacts
    SESSION_LENGTH = "session_length"    # Long/short sessions
    DAY_OF_WEEK = "day_of_week"          # Weekend vs weekday patterns


class InsightPriority(Enum):
    """Priority levels for insights."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Pattern:
    """A detected pattern in user behavior."""
    pattern_type: PatternType
    description: str
    confidence: float  # 0.0 to 1.0
    occurrences: int
    first_seen: float
    last_seen: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.pattern_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "occurrences": self.occurrences,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "metadata": self.metadata,
        }


@dataclass
class ProactiveInsight:
    """
    A proactive insight or suggestion for the user.

    Per Antonio Evo Unified Spec (v3.1):
    Every proactive message MUST include the mandatory disclosure.
    """
    id: str
    message: str
    pattern_type: PatternType
    priority: InsightPriority
    confidence: float
    evidence_summary: str = ""  # Required per spec
    reason_for_surfacing: str = ""  # Required per spec
    created_at: float = field(default_factory=time.time)
    shown: bool = False
    dismissed: bool = False

    # Per spec: mandatory disclosure
    MANDATORY_DISCLOSURE = "This is an observation, not an instruction."
    MANDATORY_DISCLOSURE_IT = "Questa e un'osservazione, non un'istruzione."

    def get_full_message(self, language: str = "en") -> str:
        """
        Get the full message with mandatory disclosure.

        Per Antonio Evo spec: every proactive message must end with disclosure.
        """
        disclosure = self.MANDATORY_DISCLOSURE_IT if language == "it" else self.MANDATORY_DISCLOSURE
        return f"{self.message}\n\n[Confidence: {self.confidence:.0%}]\n{disclosure}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "message": self.message,
            "full_message": self.get_full_message(),
            "pattern_type": self.pattern_type.value,
            "priority": self.priority.value,
            "confidence": self.confidence,
            "evidence_summary": self.evidence_summary,
            "reason_for_surfacing": self.reason_for_surfacing,
            "created_at": self.created_at,
            "shown": self.shown,
            "dismissed": self.dismissed,
            "disclosure": self.MANDATORY_DISCLOSURE,
        }


class PatternAnalyzer:
    """
    Analyzes user interactions to detect patterns.
    """

    # Time ranges for "time of day" patterns
    TIME_RANGES = {
        "early_morning": (5, 8),    # 5am - 8am
        "morning": (8, 12),          # 8am - 12pm
        "afternoon": (12, 17),       # 12pm - 5pm
        "evening": (17, 21),         # 5pm - 9pm
        "night": (21, 24),           # 9pm - midnight
        "late_night": (0, 5),        # midnight - 5am
    }

    # Insight templates (Italian and English)
    INSIGHT_TEMPLATES = {
        "late_night_work": [
            "Noto che lavori spesso di notte. Ricordati di riposare!",
            "Ho notato che sei spesso attivo a tarda notte. Va tutto bene?",
            "Working late again? Remember to take breaks!",
        ],
        "frequent_topic": [
            "Noti che chiedi spesso di {topic}. Vuoi che approfondisca?",
            "Seems like {topic} is a frequent topic. Want me to create a summary?",
        ],
        "stress_pattern": [
            "Ho notato che ultimamente sembri sotto stress. Posso aiutarti?",
            "I've noticed you seem stressed lately. Want to talk about it?",
        ],
        "productivity_pattern": [
            "Noto che sei molto produttivo il {day}. Ottimo lavoro!",
            "Your productivity peaks on {day}s. Keep it up!",
        ],
        "weekend_work": [
            "Lavori anche nel weekend? Ricordati di prenderti del tempo per te.",
            "Working on weekends too? Remember to take time for yourself.",
        ],
    }

    def __init__(self, db_path: str = "data/evomemory.db"):
        """Initialize the pattern analyzer."""
        self.db_path = db_path
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Initialize proactive pattern tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Interaction log for pattern analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interaction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    hour INTEGER NOT NULL,
                    day_of_week INTEGER NOT NULL,
                    session_id TEXT,
                    topic_keywords TEXT,
                    emotional_state TEXT,
                    message_length INTEGER,
                    response_success INTEGER
                )
            """)

            # Detected patterns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_key TEXT NOT NULL,
                    description TEXT,
                    confidence REAL NOT NULL,
                    occurrences INTEGER DEFAULT 1,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    metadata TEXT,
                    UNIQUE(pattern_type, pattern_key)
                )
            """)

            # Proactive insights
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY,
                    message TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at REAL NOT NULL,
                    shown INTEGER DEFAULT 0,
                    dismissed INTEGER DEFAULT 0
                )
            """)

            # Indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interaction_time
                ON interaction_log(timestamp DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interaction_hour
                ON interaction_log(hour)
            """)

            conn.commit()

    def log_interaction(
        self,
        message: str,
        session_id: Optional[str] = None,
        emotional_state: Optional[str] = None,
        response_success: bool = True,
    ):
        """
        Log an interaction for pattern analysis.

        Args:
            message: User message text
            session_id: Current session ID
            emotional_state: Detected emotional state
            response_success: Whether response was successful
        """
        now = datetime.now()

        # Extract basic keywords (simple approach - no NLP dependency)
        keywords = self._extract_keywords(message)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO interaction_log
                (timestamp, hour, day_of_week, session_id, topic_keywords,
                 emotional_state, message_length, response_success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                now.hour,
                now.weekday(),  # 0 = Monday, 6 = Sunday
                session_id,
                ",".join(keywords) if keywords else None,
                emotional_state,
                len(message),
                1 if response_success else 0,
            ))
            conn.commit()

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract simple keywords from text.

        Basic implementation without NLP - just extracts significant words.
        """
        # Common words to exclude (stopwords)
        stopwords = {
            # Italian
            "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
            "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
            "che", "chi", "come", "cosa", "quando", "dove", "perche",
            "questo", "quello", "mio", "tuo", "suo", "nostro",
            "sono", "sei", "e", "siamo", "siete", "hanno",
            "mi", "ti", "ci", "vi", "si", "me", "te", "lui", "lei",
            "non", "ma", "se", "anche", "poi", "molto", "tanto", "poco",
            # English
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "that", "which", "who", "what", "where", "when", "how", "why",
            "this", "these", "those", "it", "its", "they", "them", "their",
            "i", "me", "my", "we", "our", "you", "your", "he", "she",
            "him", "her", "and", "but", "or", "not", "if", "so", "just",
        }

        # Clean and tokenize
        words = text.lower().split()
        words = [w.strip(".,!?;:()[]{}\"'") for w in words]

        # Filter
        keywords = [
            w for w in words
            if len(w) > 3 and w not in stopwords and w.isalpha()
        ]

        return keywords[:max_keywords]

    def analyze_patterns(self, days: int = 7) -> List[Pattern]:
        """
        Analyze recent interactions to detect patterns.

        Args:
            days: How many days back to analyze

        Returns:
            List of detected patterns
        """
        patterns = []
        cutoff = time.time() - (days * 24 * 3600)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Time of day patterns
            cursor.execute("""
                SELECT hour, COUNT(*) as count
                FROM interaction_log
                WHERE timestamp >= ?
                GROUP BY hour
                ORDER BY count DESC
            """, (cutoff,))

            hour_counts = {row["hour"]: row["count"] for row in cursor.fetchall()}
            time_patterns = self._analyze_time_patterns(hour_counts)
            patterns.extend(time_patterns)

            # 2. Day of week patterns
            cursor.execute("""
                SELECT day_of_week, COUNT(*) as count
                FROM interaction_log
                WHERE timestamp >= ?
                GROUP BY day_of_week
                ORDER BY count DESC
            """, (cutoff,))

            day_counts = {row["day_of_week"]: row["count"] for row in cursor.fetchall()}
            day_patterns = self._analyze_day_patterns(day_counts)
            patterns.extend(day_patterns)

            # 3. Topic frequency patterns
            cursor.execute("""
                SELECT topic_keywords
                FROM interaction_log
                WHERE timestamp >= ? AND topic_keywords IS NOT NULL
            """, (cutoff,))

            all_keywords = []
            for row in cursor.fetchall():
                if row["topic_keywords"]:
                    all_keywords.extend(row["topic_keywords"].split(","))

            topic_patterns = self._analyze_topic_patterns(all_keywords)
            patterns.extend(topic_patterns)

            # 4. Emotional trend patterns
            cursor.execute("""
                SELECT emotional_state, COUNT(*) as count
                FROM interaction_log
                WHERE timestamp >= ? AND emotional_state IS NOT NULL
                GROUP BY emotional_state
                ORDER BY count DESC
            """, (cutoff,))

            emotion_counts = {row["emotional_state"]: row["count"] for row in cursor.fetchall()}
            emotion_patterns = self._analyze_emotion_patterns(emotion_counts)
            patterns.extend(emotion_patterns)

        # Store patterns
        self._store_patterns(patterns)

        return patterns

    def _analyze_time_patterns(
        self,
        hour_counts: Dict[int, int],
    ) -> List[Pattern]:
        """Analyze time-of-day usage patterns."""
        patterns = []
        total = sum(hour_counts.values())

        if total < 10:  # Need minimum data
            return patterns

        # Check for late night usage
        late_night_hours = [0, 1, 2, 3, 4, 22, 23]
        late_night_count = sum(hour_counts.get(h, 0) for h in late_night_hours)

        if late_night_count / total > 0.3:  # >30% late night usage
            patterns.append(Pattern(
                pattern_type=PatternType.TIME_OF_DAY,
                description="Frequent late night usage detected",
                confidence=min(late_night_count / total + 0.2, 1.0),
                occurrences=late_night_count,
                first_seen=time.time() - 7 * 24 * 3600,  # Approximate
                last_seen=time.time(),
                metadata={"time_range": "late_night", "percentage": round(late_night_count / total * 100)},
            ))

        # Check for early morning usage
        early_morning = [5, 6, 7]
        early_count = sum(hour_counts.get(h, 0) for h in early_morning)

        if early_count / total > 0.2:  # >20% early morning
            patterns.append(Pattern(
                pattern_type=PatternType.TIME_OF_DAY,
                description="Early morning productivity pattern",
                confidence=min(early_count / total + 0.2, 1.0),
                occurrences=early_count,
                first_seen=time.time() - 7 * 24 * 3600,
                last_seen=time.time(),
                metadata={"time_range": "early_morning", "percentage": round(early_count / total * 100)},
            ))

        return patterns

    def _analyze_day_patterns(
        self,
        day_counts: Dict[int, int],
    ) -> List[Pattern]:
        """Analyze day-of-week usage patterns."""
        patterns = []
        total = sum(day_counts.values())

        if total < 10:
            return patterns

        # Check for weekend usage
        weekend_count = day_counts.get(5, 0) + day_counts.get(6, 0)  # Sat + Sun

        if weekend_count / total > 0.35:  # >35% weekend usage
            patterns.append(Pattern(
                pattern_type=PatternType.DAY_OF_WEEK,
                description="Significant weekend usage detected",
                confidence=min(weekend_count / total + 0.1, 1.0),
                occurrences=weekend_count,
                first_seen=time.time() - 7 * 24 * 3600,
                last_seen=time.time(),
                metadata={"weekend_percentage": round(weekend_count / total * 100)},
            ))

        # Find peak day
        if day_counts:
            peak_day = max(day_counts, key=day_counts.get)
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            if day_counts[peak_day] / total > 0.25:  # Single day >25%
                patterns.append(Pattern(
                    pattern_type=PatternType.DAY_OF_WEEK,
                    description=f"Peak productivity on {day_names[peak_day]}",
                    confidence=day_counts[peak_day] / total,
                    occurrences=day_counts[peak_day],
                    first_seen=time.time() - 7 * 24 * 3600,
                    last_seen=time.time(),
                    metadata={"peak_day": day_names[peak_day], "day_index": peak_day},
                ))

        return patterns

    def _analyze_topic_patterns(
        self,
        keywords: List[str],
    ) -> List[Pattern]:
        """Analyze frequently discussed topics."""
        patterns = []

        if len(keywords) < 10:
            return patterns

        # Count keyword frequency
        keyword_counts = defaultdict(int)
        for kw in keywords:
            keyword_counts[kw] += 1

        total = len(keywords)

        # Find top topics
        for keyword, count in sorted(keyword_counts.items(), key=lambda x: -x[1])[:3]:
            if count >= 3 and count / total > 0.05:  # At least 3 occurrences, >5%
                patterns.append(Pattern(
                    pattern_type=PatternType.TOPIC_FREQUENCY,
                    description=f"Frequent topic: {keyword}",
                    confidence=min(count / total + 0.3, 0.9),
                    occurrences=count,
                    first_seen=time.time() - 7 * 24 * 3600,
                    last_seen=time.time(),
                    metadata={"topic": keyword, "frequency": count},
                ))

        return patterns

    def _analyze_emotion_patterns(
        self,
        emotion_counts: Dict[str, int],
    ) -> List[Pattern]:
        """Analyze emotional state patterns."""
        patterns = []
        total = sum(emotion_counts.values())

        if total < 5:
            return patterns

        # Check for stress/negative patterns
        negative_states = ["stressed", "frustrated", "anxious", "angry", "sad"]
        negative_count = sum(emotion_counts.get(s, 0) for s in negative_states)

        if negative_count / total > 0.4:  # >40% negative emotions
            patterns.append(Pattern(
                pattern_type=PatternType.EMOTIONAL_TREND,
                description="High stress/negative emotion pattern detected",
                confidence=min(negative_count / total + 0.1, 0.9),
                occurrences=negative_count,
                first_seen=time.time() - 7 * 24 * 3600,
                last_seen=time.time(),
                metadata={
                    "negative_percentage": round(negative_count / total * 100),
                    "dominant_states": [s for s in negative_states if emotion_counts.get(s, 0) > 0],
                },
            ))

        return patterns

    def _store_patterns(self, patterns: List[Pattern]):
        """Store detected patterns in database."""
        if not patterns:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()

            for pattern in patterns:
                cursor.execute("""
                    INSERT INTO patterns
                    (pattern_type, pattern_key, description, confidence, occurrences,
                     first_seen, last_seen, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pattern_type, pattern_key) DO UPDATE SET
                        confidence = excluded.confidence,
                        occurrences = excluded.occurrences,
                        last_seen = excluded.last_seen,
                        metadata = excluded.metadata
                """, (
                    pattern.pattern_type.value,
                    pattern.metadata.get("topic", pattern.metadata.get("time_range", "default")),
                    pattern.description,
                    pattern.confidence,
                    pattern.occurrences,
                    pattern.first_seen,
                    pattern.last_seen,
                    json.dumps(pattern.metadata),
                ))

            conn.commit()

    # Per Antonio Evo Unified Spec (v3.1): minimum confidence threshold
    MIN_CONFIDENCE_THRESHOLD = 0.7

    def generate_insights(
        self,
        patterns: Optional[List[Pattern]] = None,
        max_insights: int = 3,
    ) -> List[ProactiveInsight]:
        """
        Generate proactive insights from detected patterns.

        Per Antonio Evo Unified Spec (v3.1):
        - Proactive insights require confidence threshold >= 0.7
        - Each insight must include evidence and reason for surfacing

        Args:
            patterns: Pre-computed patterns (or analyze fresh)
            max_insights: Maximum insights to generate

        Returns:
            List of ProactiveInsight objects
        """
        import uuid

        if patterns is None:
            patterns = self.analyze_patterns()

        # Per spec: filter by confidence threshold
        qualified_patterns = [
            p for p in patterns
            if p.confidence >= self.MIN_CONFIDENCE_THRESHOLD
        ]

        insights = []

        for pattern in sorted(qualified_patterns, key=lambda p: -p.confidence)[:max_insights]:
            insight = self._pattern_to_insight(pattern)
            if insight:
                insight.id = str(uuid.uuid4())[:8]
                # Per spec: set evidence and reason
                insight.evidence_summary = f"Detected {pattern.occurrences} occurrences over analyzed period"
                insight.reason_for_surfacing = f"Pattern confidence ({pattern.confidence:.0%}) exceeds threshold"
                insights.append(insight)
                self._store_insight(insight)

        return insights

    def _pattern_to_insight(self, pattern: Pattern) -> Optional[ProactiveInsight]:
        """Convert a pattern to a proactive insight."""
        import random

        if pattern.pattern_type == PatternType.TIME_OF_DAY:
            if pattern.metadata.get("time_range") == "late_night":
                templates = self.INSIGHT_TEMPLATES["late_night_work"]
                return ProactiveInsight(
                    id="",
                    message=random.choice(templates),
                    pattern_type=pattern.pattern_type,
                    priority=InsightPriority.MEDIUM,
                    confidence=pattern.confidence,
                )

        elif pattern.pattern_type == PatternType.TOPIC_FREQUENCY:
            topic = pattern.metadata.get("topic", "")
            if topic:
                templates = self.INSIGHT_TEMPLATES["frequent_topic"]
                return ProactiveInsight(
                    id="",
                    message=random.choice(templates).format(topic=topic),
                    pattern_type=pattern.pattern_type,
                    priority=InsightPriority.LOW,
                    confidence=pattern.confidence,
                )

        elif pattern.pattern_type == PatternType.EMOTIONAL_TREND:
            if pattern.metadata.get("negative_percentage", 0) > 40:
                templates = self.INSIGHT_TEMPLATES["stress_pattern"]
                return ProactiveInsight(
                    id="",
                    message=random.choice(templates),
                    pattern_type=pattern.pattern_type,
                    priority=InsightPriority.HIGH,
                    confidence=pattern.confidence,
                )

        elif pattern.pattern_type == PatternType.DAY_OF_WEEK:
            if pattern.metadata.get("weekend_percentage", 0) > 35:
                templates = self.INSIGHT_TEMPLATES["weekend_work"]
                return ProactiveInsight(
                    id="",
                    message=random.choice(templates),
                    pattern_type=pattern.pattern_type,
                    priority=InsightPriority.MEDIUM,
                    confidence=pattern.confidence,
                )
            elif pattern.metadata.get("peak_day"):
                templates = self.INSIGHT_TEMPLATES["productivity_pattern"]
                return ProactiveInsight(
                    id="",
                    message=random.choice(templates).format(day=pattern.metadata["peak_day"]),
                    pattern_type=pattern.pattern_type,
                    priority=InsightPriority.LOW,
                    confidence=pattern.confidence,
                )

        return None

    def _store_insight(self, insight: ProactiveInsight):
        """Store insight in database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO insights
                (id, message, pattern_type, priority, confidence, created_at, shown, dismissed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                insight.id,
                insight.message,
                insight.pattern_type.value,
                insight.priority.value,
                insight.confidence,
                insight.created_at,
                1 if insight.shown else 0,
                1 if insight.dismissed else 0,
            ))
            conn.commit()

    def get_pending_insights(self, limit: int = 5) -> List[ProactiveInsight]:
        """Get insights that haven't been shown yet."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM insights
                WHERE shown = 0 AND dismissed = 0
                ORDER BY
                    CASE priority
                        WHEN 'high' THEN 1
                        WHEN 'medium' THEN 2
                        WHEN 'low' THEN 3
                    END,
                    confidence DESC
                LIMIT ?
            """, (limit,))

            return [
                ProactiveInsight(
                    id=row["id"],
                    message=row["message"],
                    pattern_type=PatternType(row["pattern_type"]),
                    priority=InsightPriority(row["priority"]),
                    confidence=row["confidence"],
                    created_at=row["created_at"],
                    shown=row["shown"] == 1,
                    dismissed=row["dismissed"] == 1,
                )
                for row in cursor.fetchall()
            ]

    def mark_insight_shown(self, insight_id: str):
        """Mark an insight as shown."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE insights SET shown = 1 WHERE id = ?
            """, (insight_id,))
            conn.commit()

    def dismiss_insight(self, insight_id: str):
        """Dismiss an insight."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE insights SET dismissed = 1 WHERE id = ?
            """, (insight_id,))
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get proactive mode statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total interactions logged
            cursor.execute("SELECT COUNT(*) FROM interaction_log")
            total_interactions = cursor.fetchone()[0]

            # Patterns detected
            cursor.execute("SELECT COUNT(*) FROM patterns")
            total_patterns = cursor.fetchone()[0]

            # Insights generated
            cursor.execute("SELECT COUNT(*) FROM insights")
            total_insights = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM insights WHERE shown = 1")
            shown_insights = cursor.fetchone()[0]

            return {
                "enabled": True,
                "version": "2.2",
                "total_interactions": total_interactions,
                "patterns_detected": total_patterns,
                "insights_generated": total_insights,
                "insights_shown": shown_insights,
            }


class ProactiveService:
    """
    Background service for proactive pattern analysis.

    Runs periodic analysis and generates insights.
    """

    def __init__(
        self,
        analyzer: PatternAnalyzer,
        analysis_interval: int = 3600,  # 1 hour
        enabled: bool = True,
    ):
        """
        Initialize proactive service.

        Args:
            analyzer: PatternAnalyzer instance
            analysis_interval: Seconds between analyses
            enabled: Whether service is enabled
        """
        self.analyzer = analyzer
        self.analysis_interval = analysis_interval
        self.enabled = enabled
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._insight_callbacks: List[Callable[[ProactiveInsight], None]] = []
        self._workflow_orchestrator = None

    def set_workflow_orchestrator(self, orchestrator) -> None:
        """Inject WorkflowOrchestrator for scheduled task execution (v8.0)."""
        self._workflow_orchestrator = orchestrator

    def register_insight_callback(self, callback: Callable[[ProactiveInsight], None]):
        """Register callback for new insights."""
        self._insight_callbacks.append(callback)

    def start(self):
        """Start the background analysis thread."""
        if not self.enabled:
            logger.info("Proactive service disabled")
            return

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._thread.start()
        logger.info("Proactive service started")

    def stop(self):
        """Stop the background analysis thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Proactive service stopped")

    def _analysis_loop(self):
        """Background analysis loop."""
        while not self._stop_event.is_set():
            try:
                # Analyze patterns
                patterns = self.analyzer.analyze_patterns(days=7)

                # Generate insights
                insights = self.analyzer.generate_insights(patterns, max_insights=2)

                # Notify callbacks
                for insight in insights:
                    for callback in self._insight_callbacks:
                        try:
                            callback(insight)
                        except Exception as e:
                            logger.error(f"Insight callback error: {e}")

            except Exception as e:
                logger.error(f"Proactive analysis error: {e}")

            # v8.0: Run workflow scheduler tick
            if self._workflow_orchestrator:
                try:
                    executed = self._workflow_orchestrator.run_scheduler_tick()
                    if executed:
                        logger.info(f"Scheduler executed {len(executed)} task(s)")
                except Exception as e:
                    logger.debug(f"Scheduler tick error: {e}")

            # Wait for next analysis
            self._stop_event.wait(self.analysis_interval)

    def log_interaction(self, *args, **kwargs):
        """Proxy to analyzer's log_interaction."""
        self.analyzer.log_interaction(*args, **kwargs)
