"""
Web Search Service - Tavily API integration.

Provides web search capability for the LLM to answer questions
about current events, facts, and topics beyond its training data.

DESIGN:
- Uses Tavily API (free tier: 1000 searches/month)
- Detects queries that need web search via heuristics
- Returns formatted search context for LLM prompt injection
- API key stored in data/api_keys.json
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import requests

logger = logging.getLogger(__name__)

# Keywords that suggest a web search would be helpful
SEARCH_KEYWORDS_EN = {
    "search", "google", "look up", "find out", "latest", "current",
    "today", "yesterday", "news", "recent", "update", "2024", "2025", "2026",
    "what happened", "who won", "price of", "weather", "stock",
}

SEARCH_KEYWORDS_IT = {
    "cerca", "cercare", "google", "ricerca", "trovami", "ultimo", "ultimi",
    "attuale", "oggi", "ieri", "notizie", "recente", "aggiornamento",
    "cos'è successo", "chi ha vinto", "prezzo di", "meteo", "borsa",
}

SEARCH_KEYWORDS_FR = {
    "cherche", "recherche", "google", "trouve", "dernier", "actuel",
    "aujourd'hui", "hier", "nouvelles", "récent", "prix de", "météo",
}

SEARCH_KEYWORDS_ES = {
    "busca", "búsqueda", "google", "encuentra", "último", "actual",
    "hoy", "ayer", "noticias", "reciente", "precio de", "clima",
}

ALL_SEARCH_KEYWORDS = (
    SEARCH_KEYWORDS_EN | SEARCH_KEYWORDS_IT |
    SEARCH_KEYWORDS_FR | SEARCH_KEYWORDS_ES
)

# Question patterns that benefit from web search
QUESTION_PATTERNS = [
    r"(?:who|what|when|where|how)\s+(?:is|are|was|were|did|does|do|will|can)\b",
    r"(?:chi|cosa|quando|dove|come)\s+(?:è|sono|era|erano|ha|hanno)\b",
    r"(?:qui|que|quand|où|comment)\s+(?:est|sont|était|a|ont)\b",
    r"(?:quién|qué|cuándo|dónde|cómo)\s+(?:es|son|fue|ha|han)\b",
]


class WebSearchResult:
    """Result of a web search."""

    def __init__(
        self,
        success: bool,
        query: str = "",
        answer: str = "",
        results: List[Dict[str, Any]] = None,
        elapsed_ms: int = 0,
        error: Optional[str] = None,
    ):
        self.success = success
        self.query = query
        self.answer = answer
        self.results = results or []
        self.elapsed_ms = elapsed_ms
        self.error = error

    def to_context(self, max_results: int = 3) -> str:
        """Format search results as context for LLM prompt."""
        if not self.success or not self.results:
            return ""

        parts = ["\n\n--- WEB SEARCH RESULTS ---"]
        parts.append(f"Query: {self.query}\n")

        if self.answer:
            parts.append(f"Summary: {self.answer}\n")

        for i, result in enumerate(self.results[:max_results], 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")
            # Truncate content
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(f"[{i}] {title}")
            parts.append(f"    URL: {url}")
            parts.append(f"    {content}\n")

        parts.append("--- END SEARCH RESULTS ---")
        return "\n".join(parts)


class WebSearchService:
    """
    Web search via Tavily API.

    Provides search capability for the LLM pipeline.
    API key is loaded from data/api_keys.json.
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.auto_search = config.get("auto_search", False)
        self.max_results = config.get("max_results", 5)
        self.search_depth = config.get("search_depth", "basic")
        self.timeout = config.get("timeout", 15)
        self._api_key = config.get("api_key", "")
        self._api_keys_path = Path("data/api_keys.json")

    @property
    def api_key(self) -> str:
        """Get Tavily API key from config or stored keys."""
        if self._api_key:
            return self._api_key
        return self._load_api_key()

    def _load_api_key(self) -> str:
        """Load Tavily API key from data/api_keys.json."""
        try:
            if self._api_keys_path.exists():
                with open(self._api_keys_path, "r", encoding="utf-8") as f:
                    keys = json.load(f)
                for key in keys:
                    if key.get("name", "").lower() == "tavily":
                        return key.get("key", "")
        except Exception:
            pass
        return ""

    def is_available(self) -> bool:
        """Check if web search is configured and available."""
        return self.enabled and bool(self.api_key)

    def needs_search(self, text: str) -> bool:
        """
        Determine if a query would benefit from web search.

        Uses keyword matching and question pattern detection.
        """
        if not text or not self.is_available():
            return False

        text_lower = text.lower().strip()

        # Check for explicit search keywords
        for keyword in ALL_SEARCH_KEYWORDS:
            if keyword in text_lower:
                return True

        # Check for question patterns
        for pattern in QUESTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        return False

    def search(self, query: str) -> WebSearchResult:
        """
        Perform a web search via Tavily API.

        Args:
            query: Search query string

        Returns:
            WebSearchResult with search results
        """
        if not self.is_available():
            return WebSearchResult(
                success=False,
                query=query,
                error="Web search not configured (missing Tavily API key)",
            )

        start = time.time()

        try:
            response = requests.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "search_depth": self.search_depth,
                    "max_results": self.max_results,
                    "include_answer": True,
                    "include_raw_content": False,
                    "include_images": False,
                },
                timeout=self.timeout,
            )

            elapsed = int((time.time() - start) * 1000)

            if response.status_code == 401:
                return WebSearchResult(
                    success=False,
                    query=query,
                    elapsed_ms=elapsed,
                    error="Invalid Tavily API key",
                )

            if response.status_code == 429:
                return WebSearchResult(
                    success=False,
                    query=query,
                    elapsed_ms=elapsed,
                    error="Tavily rate limit exceeded",
                )

            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            answer = data.get("answer", "")

            logger.info(
                f"Web search: '{query}' -> {len(results)} results ({elapsed}ms)"
            )

            return WebSearchResult(
                success=True,
                query=query,
                answer=answer,
                results=results,
                elapsed_ms=elapsed,
            )

        except requests.Timeout:
            elapsed = int((time.time() - start) * 1000)
            return WebSearchResult(
                success=False,
                query=query,
                elapsed_ms=elapsed,
                error=f"Search timed out after {self.timeout}s",
            )

        except requests.ConnectionError:
            return WebSearchResult(
                success=False,
                query=query,
                error="Cannot connect to Tavily API",
            )

        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            return WebSearchResult(
                success=False,
                query=query,
                elapsed_ms=elapsed,
                error=f"Search error: {str(e)}",
            )


# Singleton
_web_search_service = None


def get_web_search_service(config: Dict[str, Any] = None) -> WebSearchService:
    """Get or create the singleton WebSearchService."""
    global _web_search_service
    if _web_search_service is None:
        _web_search_service = WebSearchService(config or {})
    return _web_search_service
