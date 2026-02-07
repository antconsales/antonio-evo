"""
Digital Twin System for Antonio Evo (v3.1).

Per Antonio Evo Unified Spec (v3.1):
- Digital Twin mode is DISABLED by default
- Must be EXPLICITLY enabled
- Is SCOPE-LIMITED

When active:
- You may approximate writing style
- You may mimic structure and tone

You may NEVER:
- Make decisions
- Send messages
- Act on behalf of the user

MANDATORY DISCLOSURE:
"Digital Twin mode active. Output is a stylistic approximation."

Learns the user's communication style and can respond "as" the user:
- Analyzes user vocabulary, sentence patterns, tone
- Stores stylistic patterns for future reference
- Requires explicit approval for all outputs

Philosophy: Learn to help, not to replace.
"""

import time
import sqlite3
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple
from enum import Enum
from collections import defaultdict, Counter
import json
import logging

logger = logging.getLogger(__name__)


class StyleDimension(Enum):
    """Dimensions of communication style."""
    VOCABULARY = "vocabulary"        # Word choice patterns
    SENTENCE_LENGTH = "sentence_length"  # Average sentence length
    PUNCTUATION = "punctuation"      # Punctuation usage
    FORMALITY = "formality"          # Formal vs casual
    EMOJI_USAGE = "emoji_usage"      # Use of emojis
    GREETING_STYLE = "greeting_style"  # How they start conversations
    CLOSING_STYLE = "closing_style"    # How they end conversations
    FILLER_WORDS = "filler_words"      # Common filler words
    QUESTION_STYLE = "question_style"  # How they ask questions


@dataclass
class StylePattern:
    """A learned style pattern."""
    dimension: StyleDimension
    pattern: str
    frequency: int
    confidence: float
    examples: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "pattern": self.pattern,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "examples": self.examples[:3],  # Limit examples
            "last_seen": self.last_seen,
        }


@dataclass
class UserStyleProfile:
    """Complete user communication style profile."""
    # Vocabulary
    favorite_words: Dict[str, int] = field(default_factory=dict)
    unique_expressions: List[str] = field(default_factory=list)

    # Sentence structure
    avg_sentence_length: float = 0.0
    avg_word_length: float = 0.0
    uses_contractions: bool = True
    capitalization_style: str = "normal"  # normal, all_caps, no_caps

    # Punctuation
    uses_exclamations: bool = False
    uses_ellipsis: bool = False
    multiple_punctuation: bool = False

    # Tone
    formality_level: float = 0.5  # 0.0 = casual, 1.0 = formal
    emoji_frequency: float = 0.0  # emojis per message

    # Greetings/Closings
    common_greetings: List[str] = field(default_factory=list)
    common_closings: List[str] = field(default_factory=list)

    # Filler words
    filler_words: List[str] = field(default_factory=list)

    # Statistics
    total_messages_analyzed: int = 0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "favorite_words": dict(sorted(self.favorite_words.items(), key=lambda x: -x[1])[:20]),
            "unique_expressions": self.unique_expressions[:10],
            "avg_sentence_length": round(self.avg_sentence_length, 1),
            "avg_word_length": round(self.avg_word_length, 1),
            "uses_contractions": self.uses_contractions,
            "capitalization_style": self.capitalization_style,
            "uses_exclamations": self.uses_exclamations,
            "uses_ellipsis": self.uses_ellipsis,
            "formality_level": round(self.formality_level, 2),
            "emoji_frequency": round(self.emoji_frequency, 2),
            "common_greetings": self.common_greetings[:5],
            "common_closings": self.common_closings[:5],
            "filler_words": self.filler_words[:10],
            "total_messages_analyzed": self.total_messages_analyzed,
            "last_updated": self.last_updated,
        }

    def get_style_summary(self) -> str:
        """Get a natural language summary of the user's style."""
        parts = []

        # Formality
        if self.formality_level < 0.3:
            parts.append("casual and relaxed")
        elif self.formality_level > 0.7:
            parts.append("formal and professional")
        else:
            parts.append("moderately formal")

        # Sentence length
        if self.avg_sentence_length < 8:
            parts.append("uses short, direct sentences")
        elif self.avg_sentence_length > 15:
            parts.append("writes in longer, detailed sentences")

        # Punctuation
        if self.uses_exclamations:
            parts.append("often uses exclamation marks")
        if self.uses_ellipsis:
            parts.append("uses ellipsis frequently")

        # Emoji
        if self.emoji_frequency > 0.3:
            parts.append("frequently uses emojis")

        return ", ".join(parts) if parts else "standard communication style"


class StyleAnalyzer:
    """
    Analyzes user messages to extract communication style patterns.
    """

    # Common Italian contractions and casual forms
    CONTRACTIONS_IT = {"c'è", "dov'è", "cos'è", "com'è", "un'", "l'", "d'", "qu'"}

    # Common English contractions
    CONTRACTIONS_EN = {
        "i'm", "you're", "we're", "they're", "it's", "that's", "what's",
        "don't", "doesn't", "won't", "can't", "couldn't", "wouldn't",
        "i've", "you've", "we've", "they've", "i'll", "you'll", "we'll"
    }

    # Formal indicators
    FORMAL_WORDS = {
        # Italian
        "gentilmente", "cordialmente", "distinti saluti", "in merito",
        "le porgo", "egregio", "spettabile",
        # English
        "kindly", "regards", "sincerely", "concerning", "hereby",
        "pursuant", "accordingly", "furthermore"
    }

    # Casual indicators
    CASUAL_WORDS = {
        # Italian
        "ciao", "bella", "bello", "dai", "va be", "ok", "sì sì",
        "figurati", "tranquillo", "niente", "boh",
        # English
        "hey", "yo", "cool", "awesome", "yeah", "nah", "gonna",
        "wanna", "kinda", "sorta", "lol", "haha"
    }

    # Common greetings
    GREETINGS = {
        # Italian
        "ciao", "salve", "buongiorno", "buonasera", "buon pomeriggio",
        "hey", "ehi", "eccomi",
        # English
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "greetings"
    }

    # Emoji pattern
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "]+",
        flags=re.UNICODE
    )

    def analyze_message(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single message for style patterns.

        Args:
            text: User message text

        Returns:
            Dictionary of extracted style features
        """
        text_lower = text.lower()
        words = text.split()
        sentences = self._split_sentences(text)

        features = {
            "word_count": len(words),
            "char_count": len(text),
            "sentence_count": len(sentences),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
        }

        # Punctuation analysis
        features["exclamation_count"] = text.count("!")
        features["question_count"] = text.count("?")
        features["ellipsis_count"] = text.count("...") + text.count("…")
        features["multiple_punctuation"] = bool(re.search(r'[!?]{2,}', text))

        # Emoji analysis
        emojis = self.EMOJI_PATTERN.findall(text)
        features["emoji_count"] = len(emojis)

        # Capitalization
        if text.isupper():
            features["capitalization"] = "all_caps"
        elif text.islower():
            features["capitalization"] = "no_caps"
        else:
            features["capitalization"] = "normal"

        # Formality
        formal_count = sum(1 for w in self.FORMAL_WORDS if w in text_lower)
        casual_count = sum(1 for w in self.CASUAL_WORDS if w in text_lower)
        features["formal_indicators"] = formal_count
        features["casual_indicators"] = casual_count
        features["formality_score"] = self._calculate_formality(formal_count, casual_count, text)

        # Contractions
        contraction_count = sum(1 for w in words if w.lower() in self.CONTRACTIONS_EN | self.CONTRACTIONS_IT)
        features["contraction_count"] = contraction_count
        features["uses_contractions"] = contraction_count > 0

        # Greetings
        features["starts_with_greeting"] = any(text_lower.startswith(g) for g in self.GREETINGS)
        features["greeting"] = self._extract_greeting(text_lower)

        # Extract significant words (excluding common words)
        features["significant_words"] = self._extract_significant_words(words)

        return features

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_formality(
        self,
        formal_count: int,
        casual_count: int,
        text: str
    ) -> float:
        """Calculate formality score (0.0 = casual, 1.0 = formal)."""
        base = 0.5

        # Formal indicators
        base += formal_count * 0.1

        # Casual indicators
        base -= casual_count * 0.1

        # Short messages tend to be more casual
        if len(text) < 20:
            base -= 0.1

        # Proper capitalization suggests formality
        if text[0].isupper() if text else False:
            base += 0.05

        return max(0.0, min(1.0, base))

    def _extract_greeting(self, text_lower: str) -> Optional[str]:
        """Extract greeting from text if present."""
        for greeting in self.GREETINGS:
            if text_lower.startswith(greeting):
                # Find the end of the greeting
                match = re.match(rf'^({greeting}[,!]?\s*)', text_lower)
                if match:
                    return match.group(1).strip()
        return None

    def _extract_significant_words(
        self,
        words: List[str],
        min_length: int = 4
    ) -> List[str]:
        """Extract significant words (not common stopwords)."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "il", "lo", "la", "le", "un", "uno", "una", "di", "a", "da",
            "in", "con", "su", "per", "tra", "fra", "che", "chi",
            "sono", "sei", "è", "siamo", "siete", "hanno",
            "questo", "quello", "mio", "tuo", "suo", "nostro"
        }

        significant = []
        for word in words:
            clean = word.lower().strip(".,!?;:()[]{}\"'")
            if len(clean) >= min_length and clean not in stopwords and clean.isalpha():
                significant.append(clean)

        return significant


class DigitalTwin:
    """
    Digital Twin system that learns and mimics user communication style.

    Per Antonio Evo Unified Spec (v3.1):
    - Disabled by default, explicitly enabled, scope-limited
    - May approximate writing style and mimic structure/tone
    - May NEVER make decisions or act on behalf of user
    """

    # Mandatory disclosures per Antonio Evo spec
    DISCLOSURE_EN = "Digital Twin mode active. Output is a stylistic approximation."
    DISCLOSURE_IT = "Modalita Digital Twin attiva. L'output e un'approssimazione stilistica."

    def __init__(self, db_path: str = "data/evomemory.db"):
        """Initialize the Digital Twin system."""
        self.db_path = db_path
        self.analyzer = StyleAnalyzer()
        self.profile = UserStyleProfile()
        self._init_schema()
        self._load_profile()

        # Per spec: disabled by default
        self.enabled = False
        self.auto_respond_enabled = False
        self.min_messages_for_twin = 50  # Minimum messages before "speak as user"

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Initialize Digital Twin tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # User style profile
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS digital_twin_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

            # Word frequency tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS twin_vocabulary (
                    word TEXT PRIMARY KEY,
                    frequency INTEGER DEFAULT 1,
                    last_seen REAL NOT NULL
                )
            """)

            # Style patterns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS twin_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dimension TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.5,
                    examples TEXT,
                    last_seen REAL NOT NULL,
                    UNIQUE(dimension, pattern)
                )
            """)

            # Learned expressions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS twin_expressions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expression TEXT UNIQUE NOT NULL,
                    context TEXT,
                    frequency INTEGER DEFAULT 1,
                    created_at REAL NOT NULL
                )
            """)

            # Auto-response templates (user-approved)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS twin_auto_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_pattern TEXT NOT NULL,
                    response_template TEXT NOT NULL,
                    approved INTEGER DEFAULT 0,
                    usage_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                )
            """)

            conn.commit()

    def _load_profile(self):
        """Load user style profile from database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Load basic profile
            cursor.execute("SELECT * FROM digital_twin_profile")
            for row in cursor.fetchall():
                key, value = row["key"], row["value"]
                try:
                    parsed = json.loads(value)
                    if hasattr(self.profile, key):
                        setattr(self.profile, key, parsed)
                except (json.JSONDecodeError, TypeError):
                    if hasattr(self.profile, key):
                        setattr(self.profile, key, value)

            # Load vocabulary
            cursor.execute("""
                SELECT word, frequency FROM twin_vocabulary
                ORDER BY frequency DESC
                LIMIT 100
            """)
            self.profile.favorite_words = {
                row["word"]: row["frequency"]
                for row in cursor.fetchall()
            }

    def _save_profile(self):
        """Save user style profile to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Save scalar profile values
            for field in [
                "avg_sentence_length", "avg_word_length", "uses_contractions",
                "capitalization_style", "uses_exclamations", "uses_ellipsis",
                "multiple_punctuation", "formality_level", "emoji_frequency",
                "common_greetings", "common_closings", "filler_words",
                "unique_expressions", "total_messages_analyzed"
            ]:
                value = getattr(self.profile, field, None)
                if value is not None:
                    cursor.execute("""
                        INSERT OR REPLACE INTO digital_twin_profile
                        (key, value, updated_at)
                        VALUES (?, ?, ?)
                    """, (field, json.dumps(value), time.time()))

            conn.commit()

    def learn_from_message(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Learn from a user message to build their style profile.

        Args:
            text: User message text
            context: Optional context (e.g., "greeting", "question")

        Returns:
            Analysis results
        """
        if not text.strip():
            return {"learned": False}

        # Analyze message
        features = self.analyzer.analyze_message(text)

        # Update profile statistics
        n = self.profile.total_messages_analyzed
        self.profile.total_messages_analyzed += 1

        # Running averages
        if n > 0:
            self.profile.avg_sentence_length = (
                (self.profile.avg_sentence_length * n + features["avg_sentence_length"]) / (n + 1)
            )
            self.profile.avg_word_length = (
                (self.profile.avg_word_length * n + features["avg_word_length"]) / (n + 1)
            )
            self.profile.formality_level = (
                (self.profile.formality_level * n + features["formality_score"]) / (n + 1)
            )
            if features["word_count"] > 0:
                emoji_freq = features["emoji_count"] / features["word_count"]
                self.profile.emoji_frequency = (
                    (self.profile.emoji_frequency * n + emoji_freq) / (n + 1)
                )
        else:
            self.profile.avg_sentence_length = features["avg_sentence_length"]
            self.profile.avg_word_length = features["avg_word_length"]
            self.profile.formality_level = features["formality_score"]
            if features["word_count"] > 0:
                self.profile.emoji_frequency = features["emoji_count"] / features["word_count"]

        # Update boolean features
        if features["exclamation_count"] > 0:
            self.profile.uses_exclamations = True
        if features["ellipsis_count"] > 0:
            self.profile.uses_ellipsis = True
        if features["uses_contractions"]:
            self.profile.uses_contractions = True

        # Update vocabulary
        self._update_vocabulary(features["significant_words"])

        # Track greeting if present
        if features["greeting"]:
            self._add_pattern(StyleDimension.GREETING_STYLE, features["greeting"], text[:50])

        # Track unique expressions
        self._detect_unique_expressions(text)

        # Save profile periodically
        if self.profile.total_messages_analyzed % 10 == 0:
            self._save_profile()

        self.profile.last_updated = time.time()

        return {
            "learned": True,
            "total_messages": self.profile.total_messages_analyzed,
            "style_confidence": min(self.profile.total_messages_analyzed / self.min_messages_for_twin, 1.0),
        }

    def _update_vocabulary(self, words: List[str]):
        """Update vocabulary frequency counts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            for word in words:
                cursor.execute("""
                    INSERT INTO twin_vocabulary (word, frequency, last_seen)
                    VALUES (?, 1, ?)
                    ON CONFLICT(word) DO UPDATE SET
                        frequency = frequency + 1,
                        last_seen = excluded.last_seen
                """, (word.lower(), time.time()))

                # Update in-memory profile
                self.profile.favorite_words[word.lower()] = \
                    self.profile.favorite_words.get(word.lower(), 0) + 1

            conn.commit()

    def _add_pattern(
        self,
        dimension: StyleDimension,
        pattern: str,
        example: str
    ):
        """Add or update a style pattern."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if pattern exists
            cursor.execute("""
                SELECT id, examples FROM twin_patterns
                WHERE dimension = ? AND pattern = ?
            """, (dimension.value, pattern))
            row = cursor.fetchone()

            if row:
                # Update existing
                examples = json.loads(row["examples"]) if row["examples"] else []
                if example not in examples:
                    examples.append(example)
                    examples = examples[-5:]  # Keep last 5

                cursor.execute("""
                    UPDATE twin_patterns
                    SET frequency = frequency + 1,
                        confidence = MIN(1.0, confidence + 0.05),
                        examples = ?,
                        last_seen = ?
                    WHERE id = ?
                """, (json.dumps(examples), time.time(), row["id"]))
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO twin_patterns
                    (dimension, pattern, frequency, confidence, examples, last_seen)
                    VALUES (?, ?, 1, 0.5, ?, ?)
                """, (dimension.value, pattern, json.dumps([example]), time.time()))

            conn.commit()

    def _detect_unique_expressions(self, text: str):
        """Detect and store unique expressions/phrases."""
        # Look for quoted text
        quoted = re.findall(r'"([^"]+)"', text)
        for quote in quoted:
            if len(quote) > 5 and len(quote) < 50:
                self._store_expression(quote)

        # Look for repeated patterns (if same phrase appears multiple times)
        # This is a simplified approach
        words = text.split()
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            if len(phrase) > 10:
                # Check if this phrase appears often in vocabulary
                word_count = sum(
                    1 for w in phrase.split()
                    if w.lower() in self.profile.favorite_words
                    and self.profile.favorite_words[w.lower()] > 3
                )
                if word_count >= 2:
                    self._store_expression(phrase)

    def _store_expression(self, expression: str):
        """Store a unique expression."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO twin_expressions (expression, frequency, created_at)
                VALUES (?, 1, ?)
                ON CONFLICT(expression) DO UPDATE SET
                    frequency = frequency + 1
            """, (expression, time.time()))

            # Update profile
            if expression not in self.profile.unique_expressions:
                self.profile.unique_expressions.append(expression)
                self.profile.unique_expressions = self.profile.unique_expressions[-20:]

            conn.commit()

    def generate_style_prompt(self, language: str = "en") -> str:
        """
        Generate a prompt section describing the user's style.

        Per Antonio Evo Unified Spec (v3.1):
        - Must include mandatory disclosure
        - May approximate writing style
        - May mimic structure and tone

        This can be injected into LLM prompts when in "twin mode".
        """
        if not self.enabled:
            return ""

        if self.profile.total_messages_analyzed < self.min_messages_for_twin:
            return ""

        # Get appropriate disclosure
        disclosure = self.DISCLOSURE_IT if language == "it" else self.DISCLOSURE_EN

        parts = []
        parts.append(f"[{disclosure}]")
        parts.append("")
        parts.append("When responding as the user's digital twin, adopt these characteristics:")

        # Style summary
        style = self.profile.get_style_summary()
        parts.append(f"- Overall style: {style}")

        # Sentence length
        if self.profile.avg_sentence_length < 8:
            parts.append("- Keep sentences short and direct")
        elif self.profile.avg_sentence_length > 15:
            parts.append("- Use longer, more detailed sentences")

        # Formality
        if self.profile.formality_level < 0.3:
            parts.append("- Be casual and informal")
        elif self.profile.formality_level > 0.7:
            parts.append("- Maintain a formal, professional tone")

        # Contractions
        if self.profile.uses_contractions:
            parts.append("- Feel free to use contractions")
        else:
            parts.append("- Avoid contractions, use full forms")

        # Punctuation
        if self.profile.uses_exclamations:
            parts.append("- Can use exclamation marks for emphasis")
        if self.profile.uses_ellipsis:
            parts.append("- May use ellipsis (...) for pauses")

        # Favorite words
        top_words = list(self.profile.favorite_words.keys())[:10]
        if top_words:
            parts.append(f"- Commonly used words: {', '.join(top_words)}")

        # Unique expressions
        if self.profile.unique_expressions:
            parts.append(f"- Characteristic phrases: {', '.join(self.profile.unique_expressions[:5])}")

        # Greetings
        if self.profile.common_greetings:
            parts.append(f"- Typical greetings: {', '.join(self.profile.common_greetings[:3])}")

        # Per spec: constraints on what Digital Twin can/cannot do
        parts.append("")
        parts.append("CONSTRAINTS (per Antonio Evo Unified Spec):")
        parts.append("- You may approximate writing style and mimic tone")
        parts.append("- You may NOT make decisions on behalf of the user")
        parts.append("- You may NOT send messages autonomously")
        parts.append("- You may NOT act without explicit user approval")

        return "\n".join(parts)

    def is_ready(self) -> bool:
        """Check if enough data has been collected for twin mode."""
        return self.profile.total_messages_analyzed >= self.min_messages_for_twin

    def get_readiness(self) -> float:
        """Get readiness score (0.0 to 1.0)."""
        return min(self.profile.total_messages_analyzed / self.min_messages_for_twin, 1.0)

    def get_profile(self) -> UserStyleProfile:
        """Get current user style profile."""
        return self.profile

    def get_stats(self) -> Dict[str, Any]:
        """Get Digital Twin statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Vocabulary size
            cursor.execute("SELECT COUNT(*) FROM twin_vocabulary")
            vocab_size = cursor.fetchone()[0]

            # Pattern count
            cursor.execute("SELECT COUNT(*) FROM twin_patterns")
            pattern_count = cursor.fetchone()[0]

            # Expression count
            cursor.execute("SELECT COUNT(*) FROM twin_expressions")
            expression_count = cursor.fetchone()[0]

        return {
            "enabled": self.enabled,  # Per spec: disabled by default
            "version": "3.1",
            "ready": self.is_ready() and self.enabled,
            "readiness": round(self.get_readiness(), 2),
            "messages_analyzed": self.profile.total_messages_analyzed,
            "messages_needed": self.min_messages_for_twin,
            "vocabulary_size": vocab_size,
            "patterns_learned": pattern_count,
            "expressions_captured": expression_count,
            "style_summary": self.profile.get_style_summary(),
            "auto_respond_enabled": self.auto_respond_enabled,
            # Per Antonio Evo spec: mandatory disclosure
            "disclosure_en": self.DISCLOSURE_EN,
            "disclosure_it": self.DISCLOSURE_IT,
        }

    def enable_auto_respond(self, enabled: bool = True):
        """Enable or disable auto-respond mode."""
        self.auto_respond_enabled = enabled

    def reset(self):
        """Reset the Digital Twin (clear all learned data)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM digital_twin_profile")
            cursor.execute("DELETE FROM twin_vocabulary")
            cursor.execute("DELETE FROM twin_patterns")
            cursor.execute("DELETE FROM twin_expressions")
            cursor.execute("DELETE FROM twin_auto_responses")
            conn.commit()

        self.profile = UserStyleProfile()
        logger.info("Digital Twin reset complete")
