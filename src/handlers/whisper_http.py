"""
Whisper HTTP Handler - Speech to text via faster-whisper HTTP server

Connects to a persistent faster-whisper server instead of spawning
subprocess for each request. This dramatically reduces latency
(~1.5s vs ~5-8s cold start).

Server: https://github.com/fedirz/faster-whisper-server
"""

import os
import tempfile
import logging
from typing import Dict, Any, Optional

import requests

from .base import BaseHandler
from ..models.request import Request
from ..models.response import Response, ResponseMeta

logger = logging.getLogger(__name__)


class WhisperHTTPHandler(BaseHandler):
    """
    faster-whisper HTTP client for speech-to-text.

    Sends audio to a persistent HTTP server for transcription.
    Falls back to subprocess WhisperHandler if server unavailable.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.server_url = config.get("server_url", "http://localhost:8803")
        self.timeout = config.get("timeout", 30)
        self.language = config.get("language", "auto")
        self.model = config.get("model", "base")

        # Lazy-loaded fallback handler
        self._fallback_handler: Optional["WhisperHandler"] = None
        self._use_fallback = config.get("enable_fallback", True)

    def _get_fallback(self) -> Optional["WhisperHandler"]:
        """Lazy-load subprocess fallback handler."""
        if not self._use_fallback:
            return None

        if self._fallback_handler is None:
            try:
                from .whisper import WhisperHandler
                self._fallback_handler = WhisperHandler(self.config)
                if not self._fallback_handler.is_available():
                    self._fallback_handler = None
            except ImportError:
                pass

        return self._fallback_handler

    def process(self, request: Request) -> Response:
        """Transcribe audio to text via HTTP server."""

        audio_path = request.audio_path
        audio_bytes = request.audio_bytes

        if not audio_path and not audio_bytes:
            return Response.error_response(
                error="No audio provided",
                code="MISSING_AUDIO"
            )

        # Prepare audio data
        temp_file = None
        if audio_bytes and not audio_path:
            try:
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                )
                temp_file.write(audio_bytes)
                temp_file.close()
                audio_path = temp_file.name
            except IOError as e:
                return Response.error_response(
                    error=f"Failed to write temp audio file: {e}",
                    code="IO_ERROR"
                )

        try:
            # Try HTTP server first
            result = self._transcribe_http(audio_path)
            if result is not None:
                return result

            # Fall back to subprocess if HTTP fails
            logger.warning("HTTP transcription failed, trying fallback...")
            fallback = self._get_fallback()
            if fallback:
                return fallback.process(request)

            return Response.error_response(
                error="ASR server unavailable and no fallback configured",
                code="ASR_UNAVAILABLE"
            )

        finally:
            # Cleanup temp file
            if temp_file and os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    def _transcribe_http(self, audio_path: str) -> Optional[Response]:
        """
        Send audio to faster-whisper server.

        faster-whisper-server API:
        POST /v1/audio/transcriptions
        - file: audio file (multipart/form-data)
        - language: language code (optional, "auto" for detection)
        - response_format: "json", "text", "verbose_json"
        """
        try:
            # Build the transcription URL
            # faster-whisper-server uses OpenAI-compatible API
            url = f"{self.server_url}/v1/audio/transcriptions"

            with open(audio_path, "rb") as audio_file:
                files = {
                    "file": ("audio.wav", audio_file, "audio/wav")
                }
                data = {
                    "response_format": "json"
                }

                # Add language if not auto
                if self.language and self.language != "auto":
                    data["language"] = self.language

                response = requests.post(
                    url,
                    files=files,
                    data=data,
                    timeout=self.timeout
                )

            if response.status_code == 200:
                result = response.json()
                transcription = result.get("text", "").strip()

                return Response(
                    success=True,
                    text=transcription,
                    output={
                        "transcription": transcription,
                        "language": result.get("language", self.language),
                        "source": "http"
                    },
                    meta=ResponseMeta()
                )

            logger.warning(
                f"HTTP transcription failed: {response.status_code} - {response.text}"
            )
            return None

        except requests.Timeout:
            logger.warning(f"HTTP transcription timeout after {self.timeout}s")
            return None
        except requests.ConnectionError as e:
            logger.warning(f"Cannot connect to ASR server: {e}")
            return None
        except requests.RequestException as e:
            logger.warning(f"HTTP transcription error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected transcription error: {e}")
            return None

    def is_available(self) -> bool:
        """Check if HTTP server is available."""
        try:
            # Try health endpoint first
            response = requests.get(
                f"{self.server_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                return True

            # Try root endpoint as fallback
            response = requests.get(
                self.server_url,
                timeout=5
            )
            return response.status_code in (200, 404)

        except requests.RequestException:
            # Check if fallback is available
            fallback = self._get_fallback()
            return fallback is not None and fallback.is_available()

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
            "handler": "WhisperHTTPHandler",
            "server_url": self.server_url,
            "http_available": http_available,
            "fallback_available": fallback_available,
            "language": self.language,
            "model": self.model,
            "timeout": self.timeout
        }
