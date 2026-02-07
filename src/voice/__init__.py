"""
Voice module for Antonio Local Orchestrator.

Provides optional voice input (speech-to-text) and voice output (text-to-speech)
as I/O adapters. Voice is not intelligence - it's just another way to
input text and receive text output.

CORE PRINCIPLE:
Voice is I/O only. Text remains the source of truth.
Transcribed speech enters the same pipeline as typed text.
TTS receives only final formatted text from ResponseFormatter.

FEATURES:
- Voice Input: Push-to-talk speech-to-text via Whisper (local, CPU-only)
- Voice Output: Text-to-speech via Piper (local, CPU-only)

DISABLED BY DEFAULT:
All voice features must be explicitly enabled in config/voice.json.
"""

from .config import VoiceConfig, load_voice_config
from .input import (
    VoiceInput,
    VoiceInputResult,
    VoiceInputError,
    VoiceInputErrorCode,
)
from .output import (
    VoiceOutput,
    VoiceOutputResult,
    VoiceOutputError,
    VoiceOutputErrorCode,
)

__all__ = [
    # Config
    "VoiceConfig",
    "load_voice_config",
    # Input
    "VoiceInput",
    "VoiceInputResult",
    "VoiceInputError",
    "VoiceInputErrorCode",
    # Output
    "VoiceOutput",
    "VoiceOutputResult",
    "VoiceOutputError",
    "VoiceOutputErrorCode",
]
