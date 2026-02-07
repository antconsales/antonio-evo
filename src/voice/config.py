"""
Voice Configuration - Settings for voice input and output.

All voice features are disabled by default.
Configuration is loaded from config/voice.json.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class VoiceInputConfig:
    """
    Configuration for voice input (speech-to-text).

    Attributes:
        enabled: Whether voice input is enabled
        engine: STT engine to use (whisper)
        model: Whisper model size (tiny, base, small, medium, large)
        language: Language code or "auto" for detection
        timeout_seconds: Max time to wait for speech
        max_duration_seconds: Max recording duration
        sample_rate: Audio sample rate in Hz
    """
    enabled: bool = False
    engine: str = "whisper"
    model: str = "base"
    language: str = "auto"
    timeout_seconds: int = 30
    max_duration_seconds: int = 60
    sample_rate: int = 16000

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceInputConfig":
        """Create config from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            engine=data.get("engine", "whisper"),
            model=data.get("model", "base"),
            language=data.get("language", "auto"),
            timeout_seconds=data.get("timeout_seconds", 30),
            max_duration_seconds=data.get("max_duration_seconds", 60),
            sample_rate=data.get("sample_rate", 16000),
        )


@dataclass
class VoiceOutputConfig:
    """
    Configuration for voice output (text-to-speech).

    Attributes:
        enabled: Whether voice output is enabled
        engine: TTS engine to use (piper)
        voice: Voice model to use
        speed: Speech speed multiplier
    """
    enabled: bool = False
    engine: str = "piper"
    voice: str = "en_US-lessac-medium"
    speed: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceOutputConfig":
        """Create config from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            engine=data.get("engine", "piper"),
            voice=data.get("voice", "en_US-lessac-medium"),
            speed=data.get("speed", 1.0),
        )


@dataclass
class VoiceConfig:
    """
    Complete voice configuration.

    Combines input and output settings.
    """
    input: VoiceInputConfig = field(default_factory=VoiceInputConfig)
    output: VoiceOutputConfig = field(default_factory=VoiceOutputConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceConfig":
        """Create config from dictionary."""
        return cls(
            input=VoiceInputConfig.from_dict(data.get("input", {})),
            output=VoiceOutputConfig.from_dict(data.get("output", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input": {
                "enabled": self.input.enabled,
                "engine": self.input.engine,
                "model": self.input.model,
                "language": self.input.language,
                "timeout_seconds": self.input.timeout_seconds,
                "max_duration_seconds": self.input.max_duration_seconds,
                "sample_rate": self.input.sample_rate,
            },
            "output": {
                "enabled": self.output.enabled,
                "engine": self.output.engine,
                "voice": self.output.voice,
                "speed": self.output.speed,
            },
        }


def load_voice_config(config_path: str = "config/voice.json") -> VoiceConfig:
    """
    Load voice configuration from file.

    Args:
        config_path: Path to voice.json

    Returns:
        VoiceConfig with loaded settings or defaults

    Note:
        Returns default (disabled) config if file not found or invalid.
        Never raises exceptions.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return VoiceConfig.from_dict(data)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        # Return defaults (all disabled)
        return VoiceConfig()
    except Exception:
        # Catch-all for unexpected errors
        return VoiceConfig()
