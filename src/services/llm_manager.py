"""
Multi-LLM Awareness Manager for Antonio Evo.

Manages multiple LLM providers with intelligent selection:
- Detects available LLMs (Ollama local, external APIs)
- Selects best LLM based on profile capabilities
- Handles fallback between local and external
- Tracks performance metrics

Philosophy: Local-first, fallback gracefully.
"""

import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import sqlite3

import requests

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Available LLM providers."""
    OLLAMA = "ollama"           # Local Ollama
    ANTHROPIC = "anthropic"     # Claude API
    OPENAI = "openai"           # OpenAI API
    GROQ = "groq"               # Groq API (fast inference)
    LOCAL_GGUF = "local_gguf"   # Local GGUF via llama.cpp


class LLMStatus(Enum):
    """LLM availability status."""
    AVAILABLE = "available"
    WARMING_UP = "warming_up"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


@dataclass
class LLMEndpoint:
    """Configuration for an LLM endpoint."""
    provider: LLMProvider
    name: str
    base_url: str
    model: str
    api_key: Optional[str] = None
    is_local: bool = True
    priority: int = 1  # Lower = higher priority
    max_context: int = 4096
    timeout: int = 120

    # Runtime state
    status: LLMStatus = LLMStatus.UNAVAILABLE
    last_check: Optional[datetime] = None
    last_error: Optional[str] = None
    avg_latency_ms: float = 0
    request_count: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.value,
            "name": self.name,
            "model": self.model,
            "is_local": self.is_local,
            "priority": self.priority,
            "status": self.status.value,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "request_count": self.request_count,
            "error_count": self.error_count,
        }


@dataclass
class LLMRequest:
    """A request to an LLM."""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    stop_sequences: List[str] = field(default_factory=list)
    context: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Response from an LLM."""
    success: bool
    text: str = ""
    provider: Optional[LLMProvider] = None
    model: str = ""
    latency_ms: int = 0
    tokens_used: int = 0
    error: Optional[str] = None
    was_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "provider": self.provider.value if self.provider else None,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
            "was_fallback": self.was_fallback,
        }


class LLMManager:
    """
    Multi-LLM Manager with intelligent selection.

    Features:
    - Automatic provider detection
    - Profile-aware selection
    - Latency-based routing
    - Automatic fallback
    - Performance tracking
    """

    def __init__(
        self,
        db_path: str = "data/evomemory.db",
        profile_capabilities: Optional[Any] = None,
    ):
        self.db_path = db_path
        self.profile_capabilities = profile_capabilities
        self.endpoints: Dict[str, LLMEndpoint] = {}
        self._lock = threading.Lock()
        self._initialized = False

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize performance tracking database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    tokens_used INTEGER,
                    success INTEGER NOT NULL,
                    error TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_metrics_provider
                ON llm_metrics(provider, timestamp)
            """)

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to initialize LLM metrics DB: {e}")

    def register_endpoint(self, endpoint: LLMEndpoint) -> None:
        """Register an LLM endpoint."""
        with self._lock:
            key = f"{endpoint.provider.value}:{endpoint.name}"
            self.endpoints[key] = endpoint
            logger.info(f"Registered LLM endpoint: {key}")

    def configure_from_env(self, config: Any) -> None:
        """
        Configure endpoints from environment/config.

        Args:
            config: ServiceConfig with LLM settings
        """
        # Register Ollama (local)
        if hasattr(config, 'llm_server'):
            self.register_endpoint(LLMEndpoint(
                provider=LLMProvider.OLLAMA,
                name="primary",
                base_url=config.llm_server,
                model=getattr(config, 'ollama_model', 'mistral'),
                is_local=True,
                priority=1,
                max_context=getattr(config, 'ollama_context_length', 4096),
                timeout=getattr(config, 'llm_timeout', 120),
            ))

        # Register external API fallback if configured
        if hasattr(config, 'external_api_key') and config.external_api_key:
            provider_map = {
                'anthropic': LLMProvider.ANTHROPIC,
                'openai': LLMProvider.OPENAI,
                'groq': LLMProvider.GROQ,
            }
            provider_name = getattr(config, 'external_provider', 'anthropic')
            provider = provider_map.get(provider_name, LLMProvider.ANTHROPIC)

            base_urls = {
                LLMProvider.ANTHROPIC: "https://api.anthropic.com",
                LLMProvider.OPENAI: "https://api.openai.com",
                LLMProvider.GROQ: "https://api.groq.com/openai",
            }

            self.register_endpoint(LLMEndpoint(
                provider=provider,
                name="fallback",
                base_url=base_urls.get(provider, ""),
                model=getattr(config, 'external_model', 'claude-3-haiku-20240307'),
                api_key=config.external_api_key,
                is_local=False,
                priority=10,  # Lower priority than local
                max_context=8192,
                timeout=60,
            ))

        self._initialized = True

    def check_availability(self, endpoint_key: Optional[str] = None) -> Dict[str, LLMStatus]:
        """
        Check availability of LLM endpoints.

        Args:
            endpoint_key: Optional specific endpoint to check

        Returns:
            Dict of endpoint keys to their status
        """
        results = {}

        endpoints_to_check = (
            {endpoint_key: self.endpoints[endpoint_key]}
            if endpoint_key and endpoint_key in self.endpoints
            else self.endpoints
        )

        for key, endpoint in endpoints_to_check.items():
            status = self._check_endpoint(endpoint)
            endpoint.status = status
            endpoint.last_check = datetime.now()
            results[key] = status

        return results

    def _check_endpoint(self, endpoint: LLMEndpoint) -> LLMStatus:
        """Check a single endpoint's availability."""
        try:
            if endpoint.provider == LLMProvider.OLLAMA:
                # Check Ollama
                response = requests.get(
                    f"{endpoint.base_url}/api/tags",
                    timeout=5
                )
                if response.status_code == 200:
                    # Check if specific model is available
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "").split(":")[0] for m in models]
                    if endpoint.model.split(":")[0] in model_names:
                        return LLMStatus.AVAILABLE
                    else:
                        logger.warning(f"Ollama model {endpoint.model} not found")
                        return LLMStatus.UNAVAILABLE
                return LLMStatus.UNAVAILABLE

            elif endpoint.provider == LLMProvider.ANTHROPIC:
                # Just verify we have an API key
                if endpoint.api_key:
                    return LLMStatus.AVAILABLE
                return LLMStatus.UNAVAILABLE

            elif endpoint.provider == LLMProvider.OPENAI:
                if endpoint.api_key:
                    return LLMStatus.AVAILABLE
                return LLMStatus.UNAVAILABLE

            elif endpoint.provider == LLMProvider.GROQ:
                if endpoint.api_key:
                    return LLMStatus.AVAILABLE
                return LLMStatus.UNAVAILABLE

            else:
                return LLMStatus.UNAVAILABLE

        except requests.exceptions.Timeout:
            endpoint.last_error = "Connection timeout"
            return LLMStatus.UNAVAILABLE
        except requests.exceptions.ConnectionError:
            endpoint.last_error = "Connection refused"
            return LLMStatus.UNAVAILABLE
        except Exception as e:
            endpoint.last_error = str(e)
            return LLMStatus.ERROR

    def select_endpoint(
        self,
        prefer_local: bool = True,
        required_context: int = 0,
    ) -> Optional[LLMEndpoint]:
        """
        Select the best available endpoint.

        Args:
            prefer_local: Prefer local endpoints over external
            required_context: Minimum context window required

        Returns:
            Best available endpoint or None
        """
        # Filter available endpoints
        available = [
            ep for ep in self.endpoints.values()
            if ep.status == LLMStatus.AVAILABLE
            and ep.max_context >= required_context
        ]

        if not available:
            # Try to check availability first
            self.check_availability()
            available = [
                ep for ep in self.endpoints.values()
                if ep.status == LLMStatus.AVAILABLE
                and ep.max_context >= required_context
            ]

        if not available:
            return None

        # Check profile preferences
        if self.profile_capabilities:
            # If profile is external-primary, flip preference
            if getattr(self.profile_capabilities, 'external_llm_primary', False):
                prefer_local = False
            # If profile doesn't allow external fallback, filter them out
            if not getattr(self.profile_capabilities, 'external_llm_fallback', True):
                available = [ep for ep in available if ep.is_local]

        if not available:
            return None

        # Sort by:
        # 1. Local preference (if prefer_local)
        # 2. Priority
        # 3. Average latency
        def sort_key(ep: LLMEndpoint) -> tuple:
            local_score = 0 if (prefer_local == ep.is_local) else 1
            return (local_score, ep.priority, ep.avg_latency_ms)

        available.sort(key=sort_key)
        return available[0]

    def generate(
        self,
        request: LLMRequest,
        prefer_local: bool = True,
        allow_fallback: bool = True,
    ) -> LLMResponse:
        """
        Generate response from best available LLM.

        Args:
            request: The LLM request
            prefer_local: Prefer local endpoints
            allow_fallback: Allow fallback to other endpoints on failure

        Returns:
            LLM response
        """
        # Select primary endpoint
        endpoint = self.select_endpoint(
            prefer_local=prefer_local,
            required_context=len(request.prompt) // 4,  # Rough token estimate
        )

        if not endpoint:
            return LLMResponse(
                success=False,
                error="No LLM endpoints available",
            )

        # Try primary endpoint
        response = self._call_endpoint(endpoint, request)

        if response.success:
            return response

        # Try fallback if allowed
        if allow_fallback:
            fallback_endpoints = [
                ep for ep in self.endpoints.values()
                if ep.status == LLMStatus.AVAILABLE
                and ep != endpoint
            ]

            for fallback in sorted(fallback_endpoints, key=lambda x: x.priority):
                logger.info(f"Falling back to {fallback.provider.value}:{fallback.name}")
                response = self._call_endpoint(fallback, request)
                if response.success:
                    response.was_fallback = True
                    return response

        return response

    def _call_endpoint(self, endpoint: LLMEndpoint, request: LLMRequest) -> LLMResponse:
        """Call a specific endpoint."""
        start_time = time.time()

        try:
            if endpoint.provider == LLMProvider.OLLAMA:
                return self._call_ollama(endpoint, request, start_time)
            elif endpoint.provider == LLMProvider.ANTHROPIC:
                return self._call_anthropic(endpoint, request, start_time)
            elif endpoint.provider == LLMProvider.OPENAI:
                return self._call_openai(endpoint, request, start_time)
            elif endpoint.provider == LLMProvider.GROQ:
                return self._call_groq(endpoint, request, start_time)
            else:
                return LLMResponse(
                    success=False,
                    error=f"Unsupported provider: {endpoint.provider.value}",
                )
        except Exception as e:
            self._record_metric(endpoint, 0, 0, False, str(e))
            endpoint.error_count += 1
            return LLMResponse(
                success=False,
                error=str(e),
                provider=endpoint.provider,
                model=endpoint.model,
            )

    def _call_ollama(
        self,
        endpoint: LLMEndpoint,
        request: LLMRequest,
        start_time: float,
    ) -> LLMResponse:
        """Call Ollama API."""
        try:
            payload = {
                "model": endpoint.model,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "num_predict": request.max_tokens,
                    "temperature": request.temperature,
                },
            }

            if request.system_prompt:
                payload["system"] = request.system_prompt

            response = requests.post(
                f"{endpoint.base_url}/api/generate",
                json=payload,
                timeout=endpoint.timeout,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "")
                tokens = data.get("eval_count", len(text) // 4)

                self._record_metric(endpoint, latency_ms, tokens, True)
                self._update_latency(endpoint, latency_ms)

                return LLMResponse(
                    success=True,
                    text=text,
                    provider=endpoint.provider,
                    model=endpoint.model,
                    latency_ms=latency_ms,
                    tokens_used=tokens,
                )
            else:
                error = f"Ollama error: {response.status_code}"
                self._record_metric(endpoint, latency_ms, 0, False, error)
                return LLMResponse(
                    success=False,
                    error=error,
                    provider=endpoint.provider,
                    model=endpoint.model,
                    latency_ms=latency_ms,
                )

        except requests.exceptions.Timeout:
            latency_ms = int((time.time() - start_time) * 1000)
            self._record_metric(endpoint, latency_ms, 0, False, "Timeout")
            return LLMResponse(
                success=False,
                error="Request timeout",
                provider=endpoint.provider,
                model=endpoint.model,
                latency_ms=latency_ms,
            )

    def _call_anthropic(
        self,
        endpoint: LLMEndpoint,
        request: LLMRequest,
        start_time: float,
    ) -> LLMResponse:
        """Call Anthropic Claude API."""
        try:
            headers = {
                "x-api-key": endpoint.api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            }

            messages = [{"role": "user", "content": request.prompt}]

            payload = {
                "model": endpoint.model,
                "max_tokens": request.max_tokens,
                "messages": messages,
            }

            if request.system_prompt:
                payload["system"] = request.system_prompt

            response = requests.post(
                f"{endpoint.base_url}/v1/messages",
                headers=headers,
                json=payload,
                timeout=endpoint.timeout,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()
                text = data.get("content", [{}])[0].get("text", "")
                usage = data.get("usage", {})
                tokens = usage.get("output_tokens", 0) + usage.get("input_tokens", 0)

                self._record_metric(endpoint, latency_ms, tokens, True)
                self._update_latency(endpoint, latency_ms)

                return LLMResponse(
                    success=True,
                    text=text,
                    provider=endpoint.provider,
                    model=endpoint.model,
                    latency_ms=latency_ms,
                    tokens_used=tokens,
                )
            else:
                error = f"Anthropic error: {response.status_code}"
                if response.status_code == 429:
                    endpoint.status = LLMStatus.RATE_LIMITED
                self._record_metric(endpoint, latency_ms, 0, False, error)
                return LLMResponse(
                    success=False,
                    error=error,
                    provider=endpoint.provider,
                    model=endpoint.model,
                    latency_ms=latency_ms,
                )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            self._record_metric(endpoint, latency_ms, 0, False, str(e))
            return LLMResponse(
                success=False,
                error=str(e),
                provider=endpoint.provider,
                model=endpoint.model,
                latency_ms=latency_ms,
            )

    def _call_openai(
        self,
        endpoint: LLMEndpoint,
        request: LLMRequest,
        start_time: float,
    ) -> LLMResponse:
        """Call OpenAI API."""
        try:
            headers = {
                "Authorization": f"Bearer {endpoint.api_key}",
                "Content-Type": "application/json",
            }

            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

            payload = {
                "model": endpoint.model,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "messages": messages,
            }

            response = requests.post(
                f"{endpoint.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=endpoint.timeout,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens", 0)

                self._record_metric(endpoint, latency_ms, tokens, True)
                self._update_latency(endpoint, latency_ms)

                return LLMResponse(
                    success=True,
                    text=text,
                    provider=endpoint.provider,
                    model=endpoint.model,
                    latency_ms=latency_ms,
                    tokens_used=tokens,
                )
            else:
                error = f"OpenAI error: {response.status_code}"
                self._record_metric(endpoint, latency_ms, 0, False, error)
                return LLMResponse(
                    success=False,
                    error=error,
                    provider=endpoint.provider,
                    model=endpoint.model,
                    latency_ms=latency_ms,
                )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            self._record_metric(endpoint, latency_ms, 0, False, str(e))
            return LLMResponse(
                success=False,
                error=str(e),
                provider=endpoint.provider,
                model=endpoint.model,
                latency_ms=latency_ms,
            )

    def _call_groq(
        self,
        endpoint: LLMEndpoint,
        request: LLMRequest,
        start_time: float,
    ) -> LLMResponse:
        """Call Groq API (OpenAI-compatible)."""
        # Groq uses OpenAI-compatible API
        return self._call_openai(endpoint, request, start_time)

    def _update_latency(self, endpoint: LLMEndpoint, latency_ms: int) -> None:
        """Update rolling average latency."""
        endpoint.request_count += 1
        # Exponential moving average
        alpha = 0.2
        endpoint.avg_latency_ms = (
            alpha * latency_ms + (1 - alpha) * endpoint.avg_latency_ms
        )

    def _record_metric(
        self,
        endpoint: LLMEndpoint,
        latency_ms: int,
        tokens: int,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Record performance metric to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO llm_metrics
                (provider, model, latency_ms, tokens_used, success, error, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                endpoint.provider.value,
                endpoint.model,
                latency_ms,
                tokens,
                1 if success else 0,
                error,
                datetime.now().isoformat(),
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to record LLM metric: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get LLM manager statistics."""
        endpoints_info = {
            key: ep.to_dict()
            for key, ep in self.endpoints.items()
        }

        # Get recent metrics from database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get success rate by provider (last 24h)
            cursor.execute("""
                SELECT provider,
                       COUNT(*) as total,
                       SUM(success) as successes,
                       AVG(latency_ms) as avg_latency
                FROM llm_metrics
                WHERE timestamp > datetime('now', '-1 day')
                GROUP BY provider
            """)

            provider_stats = {}
            for row in cursor.fetchall():
                provider_stats[row[0]] = {
                    "total_requests": row[1],
                    "success_rate": row[2] / row[1] if row[1] > 0 else 0,
                    "avg_latency_ms": round(row[3], 1) if row[3] else 0,
                }

            conn.close()

            return {
                "version": "1.0",
                "initialized": self._initialized,
                "endpoints": endpoints_info,
                "provider_stats_24h": provider_stats,
            }

        except Exception as e:
            return {
                "version": "1.0",
                "initialized": self._initialized,
                "endpoints": endpoints_info,
                "error": str(e),
            }

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        self.check_availability()

        return [
            {
                "provider": ep.provider.value,
                "name": ep.name,
                "model": ep.model,
                "is_local": ep.is_local,
                "status": ep.status.value,
                "avg_latency_ms": round(ep.avg_latency_ms, 1),
            }
            for ep in self.endpoints.values()
        ]


# Singleton instance
_llm_manager: Optional[LLMManager] = None


def get_llm_manager(
    db_path: str = "data/evomemory.db",
    profile_capabilities: Optional[Any] = None,
) -> LLMManager:
    """Get or create the LLM manager singleton."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager(db_path, profile_capabilities)
    return _llm_manager
