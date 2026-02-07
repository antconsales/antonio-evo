"""
Wisdom & Perspective Simulation for Antonio Evo (v3.1).

Per Antonio Evo Unified Spec (v3.1):
- You NEVER share raw data
- You may use DISTILLED WISDOM

Distilled wisdom is defined as:
- Abstract principles
- Heuristics
- Anonymized experience patterns
- Confidence-weighted insights

Wisdom is:
- Non-attributable
- Non-identifying
- Non-reconstructable

You may simulate multiple abstract perspectives
filtered through numeric personality lenses.

You must NEVER claim:
- Real conversations occurred
- Identities were consulted
- Memories were shared

Output must be framed as:
- "perspectives"
- "viewpoints"
- "synthesized insights"
"""

import time
import sqlite3
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class WisdomCategory(Enum):
    """Categories of distilled wisdom."""
    GENERAL_PRINCIPLE = "general_principle"  # Universal truths/patterns
    HEURISTIC = "heuristic"  # Rules of thumb
    PATTERN = "pattern"  # Observed behavioral patterns (anonymized)
    INSIGHT = "insight"  # Synthesized understanding
    PERSPECTIVE = "perspective"  # Viewpoint from personality lens


class PerspectiveLens(Enum):
    """Personality lenses for perspective simulation."""
    ANALYTICAL = "analytical"  # Logic-focused, data-driven
    CREATIVE = "creative"  # Innovative, out-of-box thinking
    PRAGMATIC = "pragmatic"  # Practical, results-oriented
    CAUTIOUS = "cautious"  # Risk-aware, conservative
    EMPATHETIC = "empathetic"  # Feeling-oriented, supportive


@dataclass
class WisdomUnit:
    """
    A single unit of distilled wisdom.

    Non-attributable, non-identifying, non-reconstructable.
    """
    id: str
    category: WisdomCategory
    content: str  # The abstract principle/insight
    confidence: float  # 0.0 to 1.0
    source_count: int  # How many observations contributed (anonymized)
    created_at: float
    updated_at: float
    tags: List[str] = field(default_factory=list)

    # Per spec: must NOT contain
    # - Specific user data
    # - Identifiable patterns
    # - Reconstructable information

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "content": self.content,
            "confidence": self.confidence,
            "source_count": self.source_count,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def get_framed_output(self) -> str:
        """
        Get wisdom framed per spec requirements.

        Must be presented as perspective/viewpoint, not fact.
        """
        confidence_level = "high" if self.confidence > 0.8 else "moderate" if self.confidence > 0.5 else "tentative"
        return f"[Synthesized insight, {confidence_level} confidence]\n{self.content}"


@dataclass
class SynthesizedPerspective:
    """
    A perspective synthesized through personality lens.

    Per spec: These are viewpoints, NOT facts.
    """
    lens: PerspectiveLens
    content: str
    confidence: float
    basis: str  # What wisdom contributed to this (abstract description)

    # Mandatory framing per spec
    FRAMING_PREFIX = "From a {lens} perspective: "
    FRAMING_SUFFIX = "\n[This is a synthesized viewpoint, not a factual claim.]"

    def get_framed_output(self) -> str:
        """Get properly framed perspective output."""
        prefix = self.FRAMING_PREFIX.format(lens=self.lens.value)
        return f"{prefix}{self.content}{self.FRAMING_SUFFIX}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lens": self.lens.value,
            "content": self.content,
            "confidence": self.confidence,
            "basis": self.basis,
            "framed_output": self.get_framed_output(),
        }


class WisdomRepository:
    """
    Repository for distilled wisdom.

    Stores abstract principles, heuristics, and patterns
    in a non-attributable, non-identifying format.
    """

    def __init__(self, db_path: str = "data/evomemory.db"):
        """Initialize the wisdom repository."""
        self.db_path = db_path
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Initialize wisdom tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Wisdom units (distilled, non-attributable)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS wisdom_units (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_count INTEGER DEFAULT 1,
                    tags TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

            # Wisdom index for retrieval
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_wisdom_category
                ON wisdom_units(category)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_wisdom_confidence
                ON wisdom_units(confidence DESC)
            """)

            conn.commit()

    def _generate_id(self, content: str) -> str:
        """Generate non-identifying ID for wisdom."""
        # Hash content with salt to prevent reconstruction
        salted = f"wisdom:{time.time()}:{content[:50]}"
        return hashlib.sha256(salted.encode()).hexdigest()[:12]

    def store_wisdom(
        self,
        content: str,
        category: WisdomCategory,
        confidence: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> WisdomUnit:
        """
        Store a wisdom unit.

        The content MUST be:
        - Abstract (not specific to any user)
        - Non-attributable (can't trace to source)
        - Non-reconstructable (can't rebuild original data)
        """
        now = time.time()
        wisdom = WisdomUnit(
            id=self._generate_id(content),
            category=category,
            content=content,
            confidence=confidence,
            source_count=1,
            tags=tags or [],
            created_at=now,
            updated_at=now,
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO wisdom_units
                (id, category, content, confidence, source_count, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                wisdom.id,
                wisdom.category.value,
                wisdom.content,
                wisdom.confidence,
                wisdom.source_count,
                json.dumps(wisdom.tags),
                wisdom.created_at,
                wisdom.updated_at,
            ))
            conn.commit()

        logger.debug(f"Stored wisdom unit: {wisdom.id}")
        return wisdom

    def get_relevant_wisdom(
        self,
        tags: Optional[List[str]] = None,
        category: Optional[WisdomCategory] = None,
        min_confidence: float = 0.5,
        limit: int = 5,
    ) -> List[WisdomUnit]:
        """
        Retrieve relevant wisdom units.

        Returns only high-confidence, distilled insights.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT * FROM wisdom_units
                WHERE confidence >= ?
            """
            params: List[Any] = [min_confidence]

            if category:
                query += " AND category = ?"
                params.append(category.value)

            query += " ORDER BY confidence DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            wisdom_units = []
            for row in cursor.fetchall():
                unit = WisdomUnit(
                    id=row["id"],
                    category=WisdomCategory(row["category"]),
                    content=row["content"],
                    confidence=row["confidence"],
                    source_count=row["source_count"],
                    tags=json.loads(row["tags"]) if row["tags"] else [],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )

                # Filter by tags if specified
                if tags:
                    if any(t in unit.tags for t in tags):
                        wisdom_units.append(unit)
                else:
                    wisdom_units.append(unit)

            return wisdom_units

    def reinforce_wisdom(self, wisdom_id: str, confidence_boost: float = 0.05) -> bool:
        """
        Reinforce a wisdom unit's confidence.

        Called when the wisdom proves useful.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE wisdom_units
                SET confidence = MIN(1.0, confidence + ?),
                    source_count = source_count + 1,
                    updated_at = ?
                WHERE id = ?
            """, (confidence_boost, time.time(), wisdom_id))
            conn.commit()
            return cursor.rowcount > 0

    def get_stats(self) -> Dict[str, Any]:
        """Get wisdom repository statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM wisdom_units")
            total = cursor.fetchone()[0]

            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM wisdom_units
                GROUP BY category
            """)
            by_category = {row["category"]: row["count"] for row in cursor.fetchall()}

            cursor.execute("SELECT AVG(confidence) FROM wisdom_units")
            avg_conf = cursor.fetchone()[0] or 0

            return {
                "total_wisdom_units": total,
                "by_category": by_category,
                "average_confidence": round(avg_conf, 2),
            }


class WisdomSynthesizer:
    """
    Synthesizes perspectives from wisdom through personality lenses.

    Per Antonio Evo Unified Spec (v3.1):
    - Simulates multiple abstract perspectives
    - Filtered through numeric personality lenses
    - Never claims real conversations occurred
    """

    def __init__(self, repository: WisdomRepository):
        """Initialize the wisdom synthesizer."""
        self.repository = repository

    def synthesize_perspectives(
        self,
        topic: str,
        lenses: Optional[List[PerspectiveLens]] = None,
        min_confidence: float = 0.6,
    ) -> List[SynthesizedPerspective]:
        """
        Synthesize multiple perspectives on a topic.

        Per spec: These are viewpoints, not facts.
        Output is framed as "perspectives", "viewpoints", "synthesized insights".
        """
        if lenses is None:
            lenses = [PerspectiveLens.ANALYTICAL, PerspectiveLens.PRAGMATIC]

        # Get relevant wisdom
        wisdom_units = self.repository.get_relevant_wisdom(
            min_confidence=min_confidence,
            limit=10,
        )

        if not wisdom_units:
            return []

        perspectives = []
        for lens in lenses:
            perspective = self._generate_perspective(topic, lens, wisdom_units)
            if perspective:
                perspectives.append(perspective)

        return perspectives

    def _generate_perspective(
        self,
        topic: str,
        lens: PerspectiveLens,
        wisdom: List[WisdomUnit],
    ) -> Optional[SynthesizedPerspective]:
        """
        Generate a perspective through a specific lens.

        This is a simplified implementation. In production,
        this could use the LLM to synthesize perspectives
        from the abstract wisdom units.
        """
        if not wisdom:
            return None

        # Synthesize based on lens
        lens_modifiers = {
            PerspectiveLens.ANALYTICAL: "considering the logical implications and data patterns",
            PerspectiveLens.CREATIVE: "exploring innovative possibilities and unconventional approaches",
            PerspectiveLens.PRAGMATIC: "focusing on practical outcomes and actionable steps",
            PerspectiveLens.CAUTIOUS: "weighing potential risks and recommending safeguards",
            PerspectiveLens.EMPATHETIC: "considering human factors and emotional dimensions",
        }

        modifier = lens_modifiers.get(lens, "")

        # Combine relevant wisdom into perspective
        # (In production, this would be more sophisticated)
        avg_confidence = sum(w.confidence for w in wisdom) / len(wisdom)

        # Create abstract, non-attributable synthesis
        content = f"On the topic of '{topic}', {modifier}, synthesized patterns suggest careful consideration of context and constraints."

        return SynthesizedPerspective(
            lens=lens,
            content=content,
            confidence=avg_confidence,
            basis=f"Synthesized from {len(wisdom)} abstract wisdom units",
        )

    def get_framed_response(
        self,
        perspectives: List[SynthesizedPerspective],
    ) -> str:
        """
        Get properly framed multi-perspective response.

        Per spec: Must frame as perspectives/viewpoints, not facts.
        """
        if not perspectives:
            return "No synthesized insights available for this topic."

        parts = ["Here are synthesized perspectives on this topic:", ""]

        for i, p in enumerate(perspectives, 1):
            parts.append(f"{i}. {p.get_framed_output()}")
            parts.append("")

        parts.append("---")
        parts.append("Note: These are synthesized viewpoints based on abstract patterns,")
        parts.append("not factual claims or specific advice.")

        return "\n".join(parts)


# Pre-seeded wisdom examples (abstract, non-attributable)
DEFAULT_WISDOM = [
    {
        "content": "Complex problems often benefit from being broken into smaller, manageable steps.",
        "category": WisdomCategory.GENERAL_PRINCIPLE,
        "confidence": 0.85,
        "tags": ["problem-solving", "decomposition"],
    },
    {
        "content": "Clear communication of constraints helps set realistic expectations.",
        "category": WisdomCategory.HEURISTIC,
        "confidence": 0.80,
        "tags": ["communication", "expectations"],
    },
    {
        "content": "User frustration often signals a mismatch between expectation and capability.",
        "category": WisdomCategory.PATTERN,
        "confidence": 0.75,
        "tags": ["user-experience", "emotions"],
    },
    {
        "content": "Explicit acknowledgment of uncertainty builds trust over false confidence.",
        "category": WisdomCategory.INSIGHT,
        "confidence": 0.90,
        "tags": ["trust", "honesty"],
    },
    {
        "content": "Iterative refinement often produces better results than attempting perfection initially.",
        "category": WisdomCategory.GENERAL_PRINCIPLE,
        "confidence": 0.85,
        "tags": ["iteration", "process"],
    },
]


def seed_default_wisdom(repository: WisdomRepository):
    """Seed the repository with default wisdom."""
    for item in DEFAULT_WISDOM:
        repository.store_wisdom(
            content=item["content"],
            category=item["category"],
            confidence=item["confidence"],
            tags=item["tags"],
        )
    logger.info(f"Seeded {len(DEFAULT_WISDOM)} default wisdom units")
