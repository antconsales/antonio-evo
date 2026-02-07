"""
Ollama Warm-up Service

Preloads the LLM model into memory at startup and keeps it active
with periodic requests. This eliminates the ~10-15s cold start
delay on first request.

Critical for Raspberry Pi 5 where model loading is slow.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaWarmup:
    """
    Ollama model preloading and keep-alive service.

    Features:
    - Sends initial request to load model into memory
    - Background thread sends periodic keep-alive requests
    - Prevents model unloading due to inactivity
    - Thread-safe status monitoring
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral",
        warmup_prompt: str = "Ciao, come posso aiutarti?",
        keepalive_minutes: int = 5
    ):
        self.base_url = base_url
        self.model = model
        self.warmup_prompt = warmup_prompt
        self.keepalive_seconds = keepalive_minutes * 60

        self._warmed_up = False
        self._warmup_time: Optional[float] = None
        self._last_keepalive: Optional[float] = None
        self._keepalive_count = 0

        self._stop_event = threading.Event()
        self._keepalive_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def warmup(self, timeout: int = 120) -> bool:
        """
        Send initial request to load model into memory.

        This forces Ollama to load the model from disk into RAM/GPU,
        which takes 10-15 seconds on Raspberry Pi 5 with Mistral 7B.

        Args:
            timeout: Maximum time to wait for model loading

        Returns:
            True if model is loaded, False otherwise
        """
        logger.info(f"Warming up Ollama model: {self.model}")
        start_time = time.time()

        try:
            # First, check if Ollama is running
            if not self._check_ollama_health():
                logger.warning("Ollama server not available")
                return False

            # Check if model exists, pull if not
            if not self._check_model_exists():
                logger.info(f"Model {self.model} not found, pulling...")
                if not self._pull_model(timeout):
                    logger.error(f"Failed to pull model {self.model}")
                    return False

            # Send warmup request
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": self.warmup_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 10,  # Short response for speed
                    }
                },
                timeout=timeout
            )

            if response.status_code == 200:
                elapsed = time.time() - start_time
                with self._lock:
                    self._warmed_up = True
                    self._warmup_time = elapsed

                logger.info(
                    f"Ollama warmup complete in {elapsed:.1f}s"
                )
                return True
            else:
                logger.warning(
                    f"Warmup request failed: {response.status_code} - {response.text}"
                )
                return False

        except requests.Timeout:
            logger.warning(f"Warmup timeout after {timeout}s")
            return False
        except requests.ConnectionError as e:
            logger.warning(f"Cannot connect to Ollama: {e}")
            return False
        except Exception as e:
            logger.error(f"Warmup error: {e}")
            return False

    def _check_ollama_health(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def _check_model_exists(self) -> bool:
        """Check if the model is already pulled."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                for m in models:
                    name = m.get("name", "")
                    # Match model name (with or without tag)
                    if name == self.model or name.startswith(f"{self.model}:"):
                        return True
            return False
        except:
            return False

    def _pull_model(self, timeout: int = 600) -> bool:
        """Pull model from Ollama registry."""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model, "stream": False},
                timeout=timeout
            )
            return response.status_code == 200
        except:
            return False

    def start_keepalive(self) -> None:
        """
        Start background keep-alive thread.

        Ollama unloads models after OLLAMA_KEEP_ALIVE period (default 5m).
        This thread sends minimal requests to prevent unloading.
        """
        if self._keepalive_thread is not None:
            logger.warning("Keep-alive thread already running")
            return

        self._stop_event.clear()

        def keepalive_loop():
            logger.info(
                f"Starting keep-alive thread (interval: {self.keepalive_seconds}s)"
            )

            while not self._stop_event.is_set():
                # Wait for interval (interruptible)
                if self._stop_event.wait(self.keepalive_seconds):
                    break  # Stop event was set

                try:
                    # Minimal request to keep model loaded
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": " ",  # Minimal prompt
                            "stream": False,
                            "options": {"num_predict": 1}
                        },
                        timeout=30
                    )

                    with self._lock:
                        self._last_keepalive = time.time()
                        if response.status_code == 200:
                            self._keepalive_count += 1

                    logger.debug(
                        f"Keep-alive ping sent (total: {self._keepalive_count})"
                    )

                except requests.RequestException as e:
                    logger.warning(f"Keep-alive failed: {e}")
                except Exception as e:
                    logger.error(f"Keep-alive error: {e}")

            logger.info("Keep-alive thread stopped")

        self._keepalive_thread = threading.Thread(
            target=keepalive_loop,
            name="ollama-keepalive",
            daemon=True
        )
        self._keepalive_thread.start()

    def stop_keepalive(self) -> None:
        """Stop the keep-alive thread."""
        if self._keepalive_thread is None:
            return

        logger.info("Stopping keep-alive thread...")
        self._stop_event.set()

        # Wait for thread to finish
        self._keepalive_thread.join(timeout=5)
        self._keepalive_thread = None

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of the warmup service."""
        with self._lock:
            return {
                "model": self.model,
                "base_url": self.base_url,
                "warmed_up": self._warmed_up,
                "warmup_time_seconds": self._warmup_time,
                "keepalive_running": self._keepalive_thread is not None,
                "keepalive_interval_seconds": self.keepalive_seconds,
                "keepalive_count": self._keepalive_count,
                "last_keepalive": self._last_keepalive,
            }

    def is_model_loaded(self) -> bool:
        """Check if model is currently loaded in Ollama."""
        try:
            # Send a minimal request and check response time
            start = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": " ",
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=30
            )
            elapsed = time.time() - start

            # If response is fast (<2s), model is loaded
            # If slow (>5s), model is being loaded from disk
            return response.status_code == 200 and elapsed < 5.0

        except:
            return False

    def __enter__(self):
        """Context manager entry."""
        self.warmup()
        self.start_keepalive()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_keepalive()
        return False
