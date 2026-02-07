"""
Personality Evolution System for Antonio Evo (v2.3).

Tracks and evolves personality traits based on user interactions:
- Traits on 1-100 scale: humor, formality, verbosity, curiosity, patience
- Adapts based on positive/negative feedback signals
- Remembers trait evolution history
- Configures response generation to match evolved traits

Philosophy: Learn from every interaction, adapt naturally.
"""

import time
import sqlite3
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import json
import logging
import threading

logger = logging.getLogger(__name__)


class PersonalityTrait(Enum):
    """
    Core personality traits per Antonio Evo Unified Spec (v3.1).

    7 core traits as defined in the specification:
    - Traits are on 1-100 scale
    - Changes are slow, evidence-based, reversible, and logged
    - No emergent behavior, no character drift
    """
    HUMOR = "humor"            # 1=serious, 100=playful/humorous
    FORMALITY = "formality"    # 1=casual, 100=very formal
    VERBOSITY = "verbosity"    # 1=concise, 100=detailed/lengthy
    CURIOSITY = "curiosity"    # 1=practical, 100=exploratory
    PATIENCE = "patience"      # 1=impatient, 100=very patient
    EMPATHY = "empathy"        # 1=detached, 100=highly empathetic
    CREATIVITY = "creativity"  # 1=conventional, 100=highly creative


# Default trait values (balanced personality)
# Per spec: traits are 1-100 scale, starting balanced
DEFAULT_TRAITS = {
    PersonalityTrait.HUMOR: 40,
    PersonalityTrait.FORMALITY: 50,
    PersonalityTrait.VERBOSITY: 50,
    PersonalityTrait.CURIOSITY: 60,
    PersonalityTrait.PATIENCE: 70,
    PersonalityTrait.EMPATHY: 65,
    PersonalityTrait.CREATIVITY: 55,
}


@dataclass
class TraitValue:
    """Current value of a personality trait."""
    trait: PersonalityTrait
    value: int  # 0-100
    confidence: float  # How confident we are in this value (0.0-1.0)
    last_updated: float = field(default_factory=time.time)
    evolution_count: int = 0  # How many times it's been adjusted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trait": self.trait.value,
            "value": self.value,
            "confidence": self.confidence,
            "last_updated": self.last_updated,
            "evolution_count": self.evolution_count,
        }


@dataclass
class TraitEvolution:
    """Record of a trait change."""
    trait: PersonalityTrait
    old_value: int
    new_value: int
    trigger: str  # What caused the change
    confidence_delta: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trait": self.trait.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "trigger": self.trigger,
            "confidence_delta": self.confidence_delta,
            "timestamp": self.timestamp,
        }


@dataclass
class PersonalityProfile:
    """Current personality configuration."""
    traits: Dict[PersonalityTrait, TraitValue]
    version: str = "2.3"
    created_at: float = field(default_factory=time.time)
    total_evolutions: int = 0

    def get_trait(self, trait: PersonalityTrait) -> int:
        """Get value for a trait (0-100)."""
        if trait in self.traits:
            return self.traits[trait].value
        return DEFAULT_TRAITS.get(trait, 50)

    def get_response_guidelines(self) -> Dict[str, str]:
        """
        Generate response guidelines based on current traits.

        Per Antonio Evo Unified Spec (v3.1):
        - Uses only 4 core traits: Humor, Formality, Verbosity, Curiosity
        - Guidelines influence HOW responses are formed, never WHAT decisions are made
        """
        guidelines = {}

        # Humor (1=serious, 100=playful)
        humor = self.get_trait(PersonalityTrait.HUMOR)
        if humor < 30:
            guidelines["tone"] = "serious and professional"
        elif humor > 70:
            guidelines["tone"] = "light and playful, can use appropriate humor"
        else:
            guidelines["tone"] = "friendly but focused"

        # Formality (1=casual, 100=formal)
        formality = self.get_trait(PersonalityTrait.FORMALITY)
        if formality < 30:
            guidelines["style"] = "casual and conversational"
        elif formality > 70:
            guidelines["style"] = "formal and polished"
        else:
            guidelines["style"] = "semi-formal, approachable"

        # Verbosity (1=concise, 100=detailed)
        verbosity = self.get_trait(PersonalityTrait.VERBOSITY)
        if verbosity < 30:
            guidelines["length"] = "concise, brief answers preferred"
        elif verbosity > 70:
            guidelines["length"] = "detailed explanations with examples"
        else:
            guidelines["length"] = "balanced, expand when useful"

        # Curiosity (1=practical, 100=exploratory)
        curiosity = self.get_trait(PersonalityTrait.CURIOSITY)
        if curiosity > 70:
            guidelines["exploration"] = "ask follow-up questions, explore related topics"
        elif curiosity < 30:
            guidelines["exploration"] = "stick to the question, don't digress"
        else:
            guidelines["exploration"] = "balanced exploration when relevant"

        return guidelines

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traits": {t.value: tv.to_dict() for t, tv in self.traits.items()},
            "version": self.version,
            "created_at": self.created_at,
            "total_evolutions": self.total_evolutions,
            "guidelines": self.get_response_guidelines(),
        }


class FeedbackSignal(Enum):
    """Types of feedback that influence personality evolution."""
    # Positive signals
    EXPLICIT_POSITIVE = "explicit_positive"     # User explicitly praises response
    FOLLOW_UP_ENGAGED = "follow_up_engaged"     # User asks follow-up (engagement)
    TASK_COMPLETED = "task_completed"           # Task was successfully completed
    HUMOR_APPRECIATED = "humor_appreciated"     # User responds positively to humor

    # Negative signals
    EXPLICIT_NEGATIVE = "explicit_negative"     # User criticizes response
    RETRY_REQUEST = "retry_request"             # User asks for retry/rephrase
    TOPIC_CHANGE_ABRUPT = "topic_change_abrupt" # User abruptly changes topic
    TOO_LONG = "too_long"                       # User signals response too long
    TOO_SHORT = "too_short"                     # User asks for more detail

    # Neutral/adaptive signals
    FORMALITY_MISMATCH = "formality_mismatch"   # User's formality doesn't match
    EMOTION_DETECTED = "emotion_detected"       # Emotional state detected


class PersonalityEvolutionEngine:
    """
    Manages personality trait evolution based on feedback.
    """

    # How much traits change per feedback signal
    EVOLUTION_RATES = {
        FeedbackSignal.EXPLICIT_POSITIVE: 2,
        FeedbackSignal.EXPLICIT_NEGATIVE: -3,
        FeedbackSignal.FOLLOW_UP_ENGAGED: 1,
        FeedbackSignal.RETRY_REQUEST: -2,
        FeedbackSignal.TOO_LONG: -3,
        FeedbackSignal.TOO_SHORT: 3,
        FeedbackSignal.HUMOR_APPRECIATED: 3,
    }

    # Signal-to-trait mapping (only 4 core traits per spec)
    SIGNAL_TRAIT_MAP = {
        FeedbackSignal.TOO_LONG: PersonalityTrait.VERBOSITY,
        FeedbackSignal.TOO_SHORT: PersonalityTrait.VERBOSITY,
        FeedbackSignal.HUMOR_APPRECIATED: PersonalityTrait.HUMOR,
        FeedbackSignal.EXPLICIT_POSITIVE: None,  # Affects multiple traits
        FeedbackSignal.EXPLICIT_NEGATIVE: None,
        FeedbackSignal.FOLLOW_UP_ENGAGED: PersonalityTrait.CURIOSITY,
        FeedbackSignal.RETRY_REQUEST: PersonalityTrait.VERBOSITY,  # Retry suggests need for clarity
        FeedbackSignal.FORMALITY_MISMATCH: PersonalityTrait.FORMALITY,
    }

    def __init__(self, db_path: str = "data/evomemory.db"):
        """Initialize the evolution engine."""
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()
        self._load_profile()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Initialize personality tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Current personality traits
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS personality_traits (
                    trait TEXT PRIMARY KEY,
                    value INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    last_updated REAL NOT NULL,
                    evolution_count INTEGER DEFAULT 0
                )
            """)

            # Evolution history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trait_evolution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trait TEXT NOT NULL,
                    old_value INTEGER NOT NULL,
                    new_value INTEGER NOT NULL,
                    trigger TEXT NOT NULL,
                    confidence_delta REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)

            # Index for history queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_evolution_time
                ON trait_evolution_history(timestamp DESC)
            """)

            conn.commit()

    def _load_profile(self):
        """Load personality profile from database."""
        self.profile = PersonalityProfile(traits={})

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM personality_traits")

            for row in cursor.fetchall():
                trait = PersonalityTrait(row["trait"])
                self.profile.traits[trait] = TraitValue(
                    trait=trait,
                    value=row["value"],
                    confidence=row["confidence"],
                    last_updated=row["last_updated"],
                    evolution_count=row["evolution_count"],
                )

        # Fill in missing traits with defaults
        for trait, default_value in DEFAULT_TRAITS.items():
            if trait not in self.profile.traits:
                self.profile.traits[trait] = TraitValue(
                    trait=trait,
                    value=default_value,
                    confidence=0.5,  # Low confidence for new traits
                )
                self._save_trait(self.profile.traits[trait])

    def _save_trait(self, trait_value: TraitValue):
        """Save a trait value to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO personality_traits
                (trait, value, confidence, last_updated, evolution_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                trait_value.trait.value,
                trait_value.value,
                trait_value.confidence,
                trait_value.last_updated,
                trait_value.evolution_count,
            ))
            conn.commit()

    def _record_evolution(self, evolution: TraitEvolution):
        """Record a trait evolution in history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trait_evolution_history
                (trait, old_value, new_value, trigger, confidence_delta, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                evolution.trait.value,
                evolution.old_value,
                evolution.new_value,
                evolution.trigger,
                evolution.confidence_delta,
                evolution.timestamp,
            ))
            conn.commit()

    def evolve_trait(
        self,
        trait: PersonalityTrait,
        delta: int,
        trigger: str,
        confidence_boost: float = 0.02,
    ) -> Optional[TraitEvolution]:
        """
        Evolve a specific trait.

        Args:
            trait: Which trait to evolve
            delta: How much to change (-100 to 100)
            trigger: What caused this evolution
            confidence_boost: How much to increase confidence

        Returns:
            TraitEvolution record if changed, None if no change
        """
        with self._lock:
            if trait not in self.profile.traits:
                return None

            trait_value = self.profile.traits[trait]
            old_value = trait_value.value

            # Apply delta with bounds
            new_value = max(0, min(100, old_value + delta))

            # Skip if no actual change
            if new_value == old_value:
                return None

            # Update trait
            trait_value.value = new_value
            trait_value.last_updated = time.time()
            trait_value.evolution_count += 1
            trait_value.confidence = min(1.0, trait_value.confidence + confidence_boost)

            # Save
            self._save_trait(trait_value)
            self.profile.total_evolutions += 1

            # Record evolution
            evolution = TraitEvolution(
                trait=trait,
                old_value=old_value,
                new_value=new_value,
                trigger=trigger,
                confidence_delta=confidence_boost,
            )
            self._record_evolution(evolution)

            logger.info(f"Trait evolved: {trait.value} {old_value} -> {new_value} (trigger: {trigger})")

            return evolution

    def process_feedback(
        self,
        signal: FeedbackSignal,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TraitEvolution]:
        """
        Process a feedback signal and evolve traits accordingly.

        Args:
            signal: The feedback signal
            context: Optional context (e.g., which response, emotional state)

        Returns:
            List of trait evolutions triggered
        """
        evolutions = []
        context = context or {}

        # Get evolution rate for this signal
        base_rate = self.EVOLUTION_RATES.get(signal, 0)
        if base_rate == 0:
            return evolutions

        # Get target trait(s)
        target_trait = self.SIGNAL_TRAIT_MAP.get(signal)

        if target_trait:
            # Single trait evolution
            evolution = self.evolve_trait(
                trait=target_trait,
                delta=base_rate,
                trigger=signal.value,
            )
            if evolution:
                evolutions.append(evolution)

        elif signal == FeedbackSignal.EXPLICIT_POSITIVE:
            # Positive feedback - slightly boost relevant traits
            # Per spec: slow change, evidence-based
            if context.get("was_detailed"):
                evolution = self.evolve_trait(PersonalityTrait.VERBOSITY, 1, signal.value)
                if evolution:
                    evolutions.append(evolution)
            if context.get("was_curious"):
                evolution = self.evolve_trait(PersonalityTrait.CURIOSITY, 1, signal.value)
                if evolution:
                    evolutions.append(evolution)
            if context.get("was_funny"):
                evolution = self.evolve_trait(PersonalityTrait.HUMOR, 1, signal.value)
                if evolution:
                    evolutions.append(evolution)

        elif signal == FeedbackSignal.EXPLICIT_NEGATIVE:
            # Negative feedback - determine what to adjust
            # Per spec: changes are reversible and logged
            if context.get("too_formal"):
                evolution = self.evolve_trait(PersonalityTrait.FORMALITY, -2, signal.value)
                if evolution:
                    evolutions.append(evolution)
            elif context.get("too_casual"):
                evolution = self.evolve_trait(PersonalityTrait.FORMALITY, 2, signal.value)
                if evolution:
                    evolutions.append(evolution)
            if context.get("too_verbose"):
                evolution = self.evolve_trait(PersonalityTrait.VERBOSITY, -2, signal.value)
                if evolution:
                    evolutions.append(evolution)
            elif context.get("too_brief"):
                evolution = self.evolve_trait(PersonalityTrait.VERBOSITY, 2, signal.value)
                if evolution:
                    evolutions.append(evolution)

        return evolutions

    def adapt_to_user_style(
        self,
        user_text: str,
        response_text: str,
    ) -> List[TraitEvolution]:
        """
        Automatically adapt based on user's communication style.

        Analyzes user text patterns and adjusts traits to match.

        Args:
            user_text: User's message
            response_text: Antonio's response

        Returns:
            List of trait evolutions
        """
        evolutions = []

        # Analyze user's formality
        user_formal_indicators = sum([
            "please" in user_text.lower(),
            "thank" in user_text.lower(),
            "could you" in user_text.lower(),
            "would you" in user_text.lower(),
            user_text.endswith("?"),
        ])

        user_casual_indicators = sum([
            any(w in user_text.lower() for w in ["hey", "yo", "sup", "lol", "haha", "ciao"]),
            "!" in user_text,
            len(user_text.split()) < 5,
        ])

        current_formality = self.profile.get_trait(PersonalityTrait.FORMALITY)

        # Adjust formality toward user's style
        if user_formal_indicators >= 3 and current_formality < 60:
            evolution = self.evolve_trait(
                PersonalityTrait.FORMALITY, 1,
                "user_style_formal",
                confidence_boost=0.01
            )
            if evolution:
                evolutions.append(evolution)
        elif user_casual_indicators >= 2 and current_formality > 40:
            evolution = self.evolve_trait(
                PersonalityTrait.FORMALITY, -1,
                "user_style_casual",
                confidence_boost=0.01
            )
            if evolution:
                evolutions.append(evolution)

        # Analyze user's verbosity preference
        if len(user_text.split()) > 30:  # User writes long messages
            current_verbosity = self.profile.get_trait(PersonalityTrait.VERBOSITY)
            if current_verbosity < 60:
                evolution = self.evolve_trait(
                    PersonalityTrait.VERBOSITY, 1,
                    "user_verbose",
                    confidence_boost=0.01
                )
                if evolution:
                    evolutions.append(evolution)

        return evolutions

    def get_profile(self) -> PersonalityProfile:
        """Get current personality profile."""
        return self.profile

    def get_evolution_history(
        self,
        limit: int = 20,
        trait: Optional[PersonalityTrait] = None,
    ) -> List[TraitEvolution]:
        """Get recent evolution history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if trait:
                cursor.execute("""
                    SELECT * FROM trait_evolution_history
                    WHERE trait = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (trait.value, limit))
            else:
                cursor.execute("""
                    SELECT * FROM trait_evolution_history
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

            return [
                TraitEvolution(
                    trait=PersonalityTrait(row["trait"]),
                    old_value=row["old_value"],
                    new_value=row["new_value"],
                    trigger=row["trigger"],
                    confidence_delta=row["confidence_delta"],
                    timestamp=row["timestamp"],
                )
                for row in cursor.fetchall()
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get personality evolution statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total evolutions
            cursor.execute("SELECT COUNT(*) FROM trait_evolution_history")
            total_evolutions = cursor.fetchone()[0]

            # Most evolved trait
            cursor.execute("""
                SELECT trait, COUNT(*) as count
                FROM trait_evolution_history
                GROUP BY trait
                ORDER BY count DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            most_evolved = row["trait"] if row else None

            return {
                "enabled": True,
                "version": "2.3",
                "total_evolutions": total_evolutions,
                "most_evolved_trait": most_evolved,
                "current_profile": {
                    t.value: self.profile.traits[t].value
                    for t in self.profile.traits
                },
                "guidelines": self.profile.get_response_guidelines(),
            }

    def reset_trait(self, trait: PersonalityTrait):
        """Reset a trait to default value."""
        default_value = DEFAULT_TRAITS.get(trait, 50)
        self.evolve_trait(
            trait=trait,
            delta=default_value - self.profile.get_trait(trait),
            trigger="manual_reset",
            confidence_boost=-0.2,  # Lower confidence on reset
        )

    def reset_all(self):
        """Reset all traits to defaults."""
        for trait in PersonalityTrait:
            self.reset_trait(trait)
