"""
TTS HTTP Handler - Text to speech via piper-http server

Connects to a persistent piper HTTP server instead of spawning
subprocess for each request. Target latency: ~500ms vs ~2-3s cold start.

Server: Wyoming Piper (rhasspy/wyoming-piper)
"""

import os
import hashlib
import logging
from typing import Dict, Any, Optional

import requests

from .base import BaseHandler
from ..models.request import Request
from ..models.response import Response, ResponseMeta

logger = logging.getLogger(__name__)


class TTSHTTPHandler(BaseHandler):
    """
    Piper HTTP client for text-to-speech.

    Sends text to a persistent HTTP server for synthesis.
    Falls back to subprocess TTSHandler if server unavailable.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.server_url = config.get("server_url", "http://localhost:8804")
        self.timeout = config.get("timeout", 10)
        self.voice = config.get("voice", "it_IT-riccardo-x_low")
        self.speed = config.get("speed", 1.0)
        self.output_dir = config.get("output_dir", "output/tts")

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Lazy-loaded fallback handler
        self._fallback_handler: Optional["TTSHandler"] = None
        self._use_fallback = config.get("enable_fallback", True)

    def _get_fallback(self) -> Optional["TTSHandler"]:
        """Lazy-load subprocess fallback handler."""
        if not self._use_fallback:
            return None

        if self._fallback_handler is None:
            try:
                from .tts import TTSHandler
                self._fallback_handler = TTSHandler(self.config)
                if not self._fallback_handler.is_available():
                    self._fallback_handler = None
            except ImportError:
                pass

        return self._fallback_handler

    def process(self, request: Request) -> Response:
        """Convert text to speech via HTTP server."""

        text = request.text

        if not text:
            return Response.error_response(
                error="No text provided",
                code="MISSING_TEXT"
            )

        # Generate unique output filename
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        output_path = os.path.join(self.output_dir, f"{text_hash}.wav")

        # Check cache
        if os.path.exists(output_path):
            return Response(
                success=True,
                audio_path=output_path,
                text=text,
                output={
                    "audio_path": output_path,
                    "voice": self.voice,
                    "text_length": len(text),
                    "cached": True
                },
                meta=ResponseMeta()
            )

        # Try HTTP server first
        result = self._synthesize_http(text, output_path)
        if result is not None:
            return result

        # Fall back to subprocess if HTTP fails
        logger.warning("HTTP synthesis failed, trying fallback...")
        fallback = self._get_fallback()
        if fallback:
            return fallback.process(request)

        return Response.error_response(
            error="TTS server unavailable and no fallback configured",
            code="TTS_UNAVAILABLE"
        )

    def _synthesize_http(self, text: str, output_path: str) -> Optional[Response]:
        """
        Send text to piper HTTP server.

        Wyoming Piper API:
        POST /synthesize
        - text: text to synthesize (form data)
        Returns: audio/wav stream

        Alternative (OpenAI-compatible):
        POST /v1/audio/speech
        - input: text
        - voice: voice name
        """
        try:
            # Try Wyoming Piper API first (native)
            result = self._try_wyoming_api(text, output_path)
            if result:
                return result

            # Try OpenAI-compatible API as fallback
            result = self._try_openai_api(text, output_path)
            if result:
                return result

            return None

        except requests.Timeout:
            logger.warning(f"HTTP synthesis timeout after {self.timeout}s")
            return None
        except requests.ConnectionError as e:
            logger.warning(f"Cannot connect to TTS server: {e}")
            return None
        except requests.RequestException as e:
            logger.warning(f"HTTP synthesis error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected synthesis error: {e}")
            return None

    def _try_wyoming_api(self, text: str, output_path: str) -> Optional[Response]:
        """Try Wyoming Piper native API."""
        try:
            # Wyoming Piper uses WebSocket, but some wrappers provide HTTP
            # Try common HTTP endpoints
            for endpoint in ["/api/tts", "/synthesize", "/tts"]:
                url = f"{self.server_url}{endpoint}"
                try:
                    response = requests.post(
                        url,
                        data={"text": text},
                        timeout=self.timeout
                    )
                    if response.status_code == 200:
                        with open(output_path, "wb") as f:
                            f.write(response.content)
                        return self._build_success_response(text, output_path)
                except requests.RequestException:
                    continue

            return None

        except Exception:
            return None

    def _try_openai_api(self, text: str, output_path: str) -> Optional[Response]:
        """Try OpenAI-compatible TTS API."""
        try:
            url = f"{self.server_url}/v1/audio/speech"
            response = requests.post(
                url,
                json={
                    "input": text,
                    "voice": self.voice,
                    "model": "tts-1",  # Required by OpenAI API format
                    "response_format": "wav",
                    "speed": self.speed
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return self._build_success_response(text, output_path)

            return None

        except Exception:
            return None

    def _build_success_response(self, text: str, output_path: str) -> Response:
        """Build successful TTS response."""
        return Response(
            success=True,
            audio_path=output_path,
            text=text,
            output={
                "audio_path": output_path,
                "voice": self.voice,
                "text_length": len(text),
                "source": "http"
            },
            meta=ResponseMeta()
        )

    def is_available(self) -> bool:
        """Check if HTTP server is available."""
        try:
            # Try health endpoint
            for endpoint in ["/health", "/api/health", "/"]:
                try:
                    response = requests.get(
                        f"{self.server_url}{endpoint}",
                        timeout=5
                    )
                    if response.status_code in (200, 404):
                        return True
                except:
                    continue

            # Check if fallback is available
            fallback = self._get_fallback()
            return fallback is not None and fallback.is_available()

        except:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of the handler."""
        http_available = False
        fallback_available = False

        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            http_available = response.status_code == 200
        except:
            pass

        fallback = self._get_fallback()
        if fallback:
            fallback_available = fallback.is_available()

        return {
            "handler": "TTSHTTPHandler",
            "server_url": self.server_url,
            "http_available": http_available,
            "fallback_available": fallback_available,
            "voice": self.voice,
            "speed": self.speed,
            "timeout": self.timeout
        }
