"""
Memory Retriever for EvoMemory.

Handles context retrieval before classification:
- BM25 full-text search for relevant neurons
- Exact match lookup by input hash
- User preference loading
- Session context assembly
"""

import hashlib
import time
from typing import Optional, Dict, Any, List

from .neuron import (
    Neuron,
    MemoryContext,
    RetrievedNeuron,
    UserPreference,
    Mood,
)
from .storage import MemoryStorage


class MemoryRetriever:
    """
    Retrieves relevant memory context for incoming requests.

    Used BEFORE classification to inform:
    - Intent detection
    - Persona selection
    - Policy decisions
    """

    def __init__(
        self,
        storage: MemoryStorage,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize retriever.

        Args:
            storage: MemoryStorage instance
            config: Retrieval configuration
        """
        self.storage = storage
        self.config = config or {}

        # Config defaults
        self.max_neurons = self.config.get("max_neurons", 5)
        self.min_relevance = self.config.get("min_relevance", 0.3)
        self.min_confidence = self.config.get("min_confidence", 0.3)
        self.include_preferences = self.config.get("include_preferences", True)
        self.boost_on_access = self.config.get("boost_on_access", 0.02)

    def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        include_exact_match: bool = True,
    ) -> MemoryContext:
        """
        Retrieve memory context for a query.

        This is the main entry point, called before classification.

        Args:
            query: The user's input text
            session_id: Current session ID (optional)
            include_exact_match: Check for exact input match first

        Returns:
            MemoryContext with relevant neurons and preferences
        """
        start_time = time.time()

        relevant_neurons: List[RetrievedNeuron] = []
        preferences: Dict[str, UserPreference] = {}

        # 1. Check for exact match first (fastest)
        if include_exact_match:
            exact_match = self._find_exact_match(query)
            if exact_match:
                relevant_neurons.append(RetrievedNeuron(
                    neuron=exact_match,
                    relevance_score=1.0,
                    match_type="exact",
                ))
                # Boost confidence for exact match
                self.storage.increment_access(
                    exact_match.id,
                    confidence_boost=self.boost_on_access,
                )

        # 2. BM25 search for semantic matches
        if len(relevant_neurons) < self.max_neurons:
            bm25_results = self._search_bm25(
                query,
                limit=self.max_neurons - len(relevant_neurons),
            )

            for neuron, score in bm25_results:
                # Skip if already in results (exact match)
                if any(rn.neuron.id == neuron.id for rn in relevant_neurons):
                    continue

                if score >= self.min_relevance:
                    relevant_neurons.append(RetrievedNeuron(
                        neuron=neuron,
                        relevance_score=score,
                        match_type="semantic",
                    ))
                    # Small boost for semantic matches
                    self.storage.increment_access(
                        neuron.id,
                        confidence_boost=self.boost_on_access / 2,
                    )

        # 3. Load user preferences
        if self.include_preferences:
            preferences = self.storage.get_all_preferences()

        # 4. Get session stats if available
        session_neuron_count = 0
        if session_id:
            session_neurons = self.storage.get_by_session(session_id)
            session_neuron_count = len(session_neurons)

        # 5. Compute aggregated stats
        avg_confidence = 0.0
        dominant_mood = None

        if relevant_neurons:
            avg_confidence = sum(
                rn.neuron.confidence for rn in relevant_neurons
            ) / len(relevant_neurons)

            # Find dominant mood
            mood_counts: Dict[Mood, int] = {}
            for rn in relevant_neurons:
                mood_counts[rn.neuron.mood] = mood_counts.get(rn.neuron.mood, 0) + 1
            if mood_counts:
                dominant_mood = max(mood_counts, key=mood_counts.get)

        # 6. Build context
        elapsed_ms = int((time.time() - start_time) * 1000)

        return MemoryContext(
            relevant_neurons=relevant_neurons,
            preferences=preferences,
            session_id=session_id,
            session_neuron_count=session_neuron_count,
            avg_confidence=avg_confidence,
            dominant_mood=dominant_mood,
            has_relevant_memory=len(relevant_neurons) > 0,
            memory_retrieval_ms=elapsed_ms,
        )

    def _find_exact_match(self, query: str) -> Optional[Neuron]:
        """
        Find exact match by input hash.

        Args:
            query: Input text

        Returns:
            Neuron if exact match found, None otherwise
        """
        input_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        neuron = self.storage.get_by_input_hash(input_hash)

        if neuron and neuron.confidence >= self.min_confidence:
            return neuron
        return None

    def _search_bm25(
        self,
        query: str,
        limit: int = 5,
    ) -> List[tuple]:
        """
        Search using BM25 full-text search.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of (Neuron, score) tuples
        """
        # Prepare query for FTS5
        # Remove special characters that could break the query
        clean_query = self._prepare_fts_query(query)

        if not clean_query:
            return []

        try:
            results = self.storage.search_bm25(
                query=clean_query,
                limit=limit,
                min_confidence=self.min_confidence,
            )
            return results
        except Exception:
            # FTS query failed (invalid syntax, etc.)
            # Fall back to empty results
            return []

    def _prepare_fts_query(self, query: str) -> str:
        """
        Prepare query for FTS5 search.

        Handles special characters and creates proper query syntax.

        Args:
            query: Raw query string

        Returns:
            FTS5-safe query string
        """
        # Remove special FTS characters
        special_chars = ['"', "'", "(", ")", "*", ":", "^", "-", "+"]
        clean = query
        for char in special_chars:
            clean = clean.replace(char, " ")

        # Split into words and rejoin
        words = clean.split()
        if not words:
            return ""

        # Use OR matching for flexibility
        # Each word is matched independently
        return " OR ".join(words[:10])  # Limit to 10 terms

    def get_context_for_classification(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get simplified context for the classifier.

        Returns just the data needed for classification decisions.

        Args:
            query: User input
            session_id: Current session

        Returns:
            Dictionary with classification-relevant data
        """
        context = self.retrieve(query, session_id)

        # Extract just what the classifier needs
        return {
            "has_memory": context.has_relevant_memory,
            "best_match_confidence": (
                context.relevant_neurons[0].neuron.confidence
                if context.relevant_neurons else 0.0
            ),
            "best_match_persona": (
                context.relevant_neurons[0].neuron.persona.value
                if context.relevant_neurons else None
            ),
            "user_preferred_persona": (
                context.get_preference("default_persona")
            ),
            "avg_confidence": context.avg_confidence,
            "session_length": context.session_neuron_count,
        }

    def find_similar(
        self,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.5,
    ) -> List[RetrievedNeuron]:
        """
        Find neurons similar to a query.

        Useful for debugging and inspection.

        Args:
            query: Search query
            limit: Maximum results
            min_confidence: Minimum confidence threshold

        Returns:
            List of RetrievedNeuron objects
        """
        clean_query = self._prepare_fts_query(query)
        if not clean_query:
            return []

        results = self.storage.search_bm25(
            query=clean_query,
            limit=limit,
            min_confidence=min_confidence,
        )

        return [
            RetrievedNeuron(
                neuron=neuron,
                relevance_score=score,
                match_type="semantic",
            )
            for neuron, score in results
        ]
