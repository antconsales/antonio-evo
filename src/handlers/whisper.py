"""
Whisper Handler - Speech to text via whisper.cpp

OUTPUT: Always plain text.
No reasoning happens here - just transcription.
"""

import subprocess
import tempfile
import os
from typing import Dict, Any

from .base import BaseHandler
from ..models.request import Request
from ..models.response import Response, ResponseMeta


class WhisperHandler(BaseHandler):
    """
    whisper.cpp for speech-to-text.

    This handler ONLY transcribes audio.
    It does not interpret, reason, or respond.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.whisper_path = config.get("whisper_path", "whisper")
        self.model_path = config.get("model_path", "models/ggml-base.bin")
        self.language = config.get("language", "auto")
        self.timeout = config.get("timeout", 120)

    def process(self, request: Request) -> Response:
        """Transcribe audio to text."""

        audio_path = request.audio_path
        audio_bytes = request.audio_bytes

        if not audio_path and not audio_bytes:
            return Response.error_response(
                error="No audio provided",
                code="MISSING_AUDIO"
            )

        temp_file = None

        # If bytes provided, write to temp file
        if audio_bytes and not audio_path:
            try:
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.write(audio_bytes)
                temp_file.close()
                audio_path = temp_file.name
            except IOError as e:
                return Response.error_response(
                    error=f"Failed to write temp audio file: {e}",
                    code="IO_ERROR"
                )

        try:
            # Run whisper.cpp
            cmd = [
                self.whisper_path,
                "-m", self.model_path,
                "-f", audio_path,
                "-l", self.language,
                "--no-timestamps",
                "-otxt"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode != 0:
                return Response.error_response(
                    error=f"Whisper error: {result.stderr}",
                    code="WHISPER_ERROR"
                )

            # Read output
            txt_path = audio_path + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path) as f:
                    transcription = f.read().strip()
                os.remove(txt_path)
            else:
                transcription = result.stdout.strip()

            return Response(
                success=True,
                text=transcription,
                output={"transcription": transcription, "language": self.language},
                meta=ResponseMeta()
            )

        except subprocess.TimeoutExpired:
            return Response.error_response(
                error="Transcription timeout",
                code="TIMEOUT"
            )
        except FileNotFoundError:
            return Response.error_response(
                error=f"Whisper binary not found at: {self.whisper_path}",
                code="BINARY_NOT_FOUND"
            )
        finally:
            # Cleanup temp file
            if temp_file and os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    def is_available(self) -> bool:
        """Check if whisper binary exists."""
        try:
            result = subprocess.run(
                [self.whisper_path, "--help"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
