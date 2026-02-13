"""
TTS Handler - Text to speech via Piper

WHY PIPER:
- Lightweight
- Real-time on CPU
- Italian voices available
- Minimal dependencies
"""

import subprocess
import os
import hashlib
from typing import Dict, Any

from .base import BaseHandler
from ..models.request import Request
from ..models.response import Response, ResponseMeta


class TTSHandler(BaseHandler):
    """
    Piper TTS for text-to-speech.

    This handler ONLY generates audio.
    No reasoning, no interpretation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.piper_path = config.get("piper_path", "piper")
        self.voice = config.get("voice", "en_US-lessac-medium")
        self.output_dir = config.get("output_dir", "output/tts")
        self.timeout = config.get("timeout", 30)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self, request: Request) -> Response:
        """Convert text to speech."""

        text = request.text

        if not text:
            return Response.error_response(
                error="No text provided",
                code="MISSING_TEXT"
            )

        # Generate unique output filename
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        output_path = os.path.join(self.output_dir, f"{text_hash}.wav")

        try:
            # Run Piper
            # Piper reads from stdin
            # Strip emoji/non-ASCII that Piper can't pronounce
            clean_text = text.encode('ascii', 'ignore').decode('ascii')
            result = subprocess.run(
                [
                    self.piper_path,
                    "--model", self.voice,
                    "--output_file", output_path
                ],
                input=clean_text.encode('utf-8'),
                capture_output=True,
                timeout=self.timeout
            )

            if result.returncode != 0:
                return Response.error_response(
                    error=f"Piper error: {result.stderr}",
                    code="PIPER_ERROR"
                )

            # Verify output exists
            if not os.path.exists(output_path):
                return Response.error_response(
                    error="TTS output file not created",
                    code="OUTPUT_MISSING"
                )

            return Response(
                success=True,
                audio_path=output_path,
                text=text,
                output={
                    "audio_path": output_path,
                    "voice": self.voice,
                    "text_length": len(text)
                },
                meta=ResponseMeta()
            )

        except subprocess.TimeoutExpired:
            return Response.error_response(
                error="TTS timeout",
                code="TIMEOUT"
            )
        except FileNotFoundError:
            return Response.error_response(
                error=f"Piper binary not found at: {self.piper_path}",
                code="BINARY_NOT_FOUND"
            )

    def is_available(self) -> bool:
        """Check if Piper binary exists."""
        try:
            result = subprocess.run(
                [self.piper_path, "--help"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
