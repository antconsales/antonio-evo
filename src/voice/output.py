"""
Voice Output - Text-to-speech for Antonio.

Provides local text-to-speech using Piper TTS.
All processing is local, CPU-only, and offline.

CORE PRINCIPLE:
Voice output is a derivative representation of text.
Text is the canonical source of truth.

PIPELINE:
Internal Response → ResponseFormatter → Final Text → TTS → Audio

RULES:
- TTS receives ONLY final formatted text
- TTS does NOT access internal responses
- TTS failures do NOT fail the request
- Audio is ephemeral (not stored by default)
"""

import io
import os
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import VoiceOutputConfig


class VoiceOutputErrorCode(Enum):
    """Error codes for voice output operations."""
    NOT_ENABLED = "VOICE_OUTPUT_NOT_ENABLED"
    PIPER_UNAVAILABLE = "VOICE_PIPER_UNAVAILABLE"
    TTS_FAILED = "VOICE_TTS_FAILED"
    PLAYBACK_FAILED = "VOICE_PLAYBACK_FAILED"
    EMPTY_TEXT = "VOICE_EMPTY_TEXT"
    AUDIO_SYSTEM_UNAVAILABLE = "VOICE_AUDIO_SYSTEM_UNAVAILABLE"


class VoiceOutputError(Exception):
    """Exception for voice output errors."""

    def __init__(self, code: VoiceOutputErrorCode, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code.value}: {message}")


@dataclass
class VoiceOutputResult:
    """
    Result of voice output operation.

    Attributes:
        success: Whether TTS and playback succeeded
        audio_data: Generated audio bytes (if requested)
        error_code: Error code (if failed)
        error_message: User-safe error message (if failed)
        duration_seconds: Duration of generated audio
    """
    success: bool
    audio_data: Optional[bytes] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "success": self.success,
            "duration_seconds": self.duration_seconds,
        }
        if not self.success:
            result["error_code"] = self.error_code
            result["error_message"] = self.error_message
        return result


# User-safe error messages (no internal details)
USER_ERROR_MESSAGES = {
    VoiceOutputErrorCode.NOT_ENABLED: "Voice output is not enabled.",
    VoiceOutputErrorCode.PIPER_UNAVAILABLE: "Voice output is unavailable. Piper is not installed.",
    VoiceOutputErrorCode.TTS_FAILED: "Could not generate speech. Please try again.",
    VoiceOutputErrorCode.PLAYBACK_FAILED: "Could not play audio. Check your audio settings.",
    VoiceOutputErrorCode.EMPTY_TEXT: "No text to speak.",
    VoiceOutputErrorCode.AUDIO_SYSTEM_UNAVAILABLE: "Audio playback is not available.",
}


def _check_piper_available() -> Tuple[bool, Optional[str]]:
    """
    Check if Piper TTS is available.

    Returns:
        Tuple of (available, executable_path)
    """
    # Try common piper executables
    piper_names = ["piper", "piper-tts", "piper.exe"]

    for name in piper_names:
        try:
            result = subprocess.run(
                [name, "--help"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0 or b"piper" in result.stdout.lower() or b"piper" in result.stderr.lower():
                return True, name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        except Exception:
            continue

    # Try Python piper-tts module
    try:
        import piper
        return True, "python-piper"
    except ImportError:
        pass

    return False, None


def _check_audio_playback_available() -> bool:
    """
    Check if audio playback is available.

    Returns:
        True if audio can be played
    """
    # Try pygame
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.quit()
        return True
    except (ImportError, Exception):
        pass

    # Try simpleaudio
    try:
        import simpleaudio
        return True
    except ImportError:
        pass

    # Try sounddevice
    try:
        import sounddevice
        return True
    except ImportError:
        pass

    # Try playsound
    try:
        import playsound
        return True
    except ImportError:
        pass

    # Try system commands
    import platform
    system = platform.system()

    if system == "Windows":
        # Windows has built-in audio playback
        return True
    elif system == "Darwin":
        # macOS has afplay
        try:
            subprocess.run(["which", "afplay"], capture_output=True, check=True)
            return True
        except Exception:
            pass
    elif system == "Linux":
        # Check for aplay or paplay
        for cmd in ["aplay", "paplay"]:
            try:
                subprocess.run(["which", cmd], capture_output=True, check=True)
                return True
            except Exception:
                continue

    return False


class VoiceOutput:
    """
    Voice output handler using local Piper TTS.

    Converts formatted text responses to audio.
    Text is the canonical source - audio is derivative.

    Usage:
        voice = VoiceOutput(config)

        if voice.is_available():
            # Speak the text
            result = voice.speak("Hello, world!")

            # Or get audio data
            result = voice.generate_audio("Hello, world!")
            if result.success:
                audio_bytes = result.audio_data
    """

    def __init__(self, config: Optional[VoiceOutputConfig] = None):
        """
        Initialize voice output.

        Args:
            config: Voice output configuration (uses defaults if None)
        """
        self._config = config or VoiceOutputConfig()
        self._piper_path: Optional[str] = None
        self._model_path: Optional[str] = None

    def is_available(self) -> bool:
        """
        Check if voice output is available.

        Returns:
            True if voice output can be used

        Note:
            Returns False if not enabled in config.
        """
        if not self._config.enabled:
            return False

        piper_ok, self._piper_path = _check_piper_available()
        if not piper_ok:
            return False

        return True

    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check all voice output dependencies.

        Returns:
            Dictionary with dependency status
        """
        piper_ok, piper_path = _check_piper_available()
        playback_ok = _check_audio_playback_available()

        return {
            "enabled": self._config.enabled,
            "piper_available": piper_ok,
            "piper_path": piper_path,
            "playback_available": playback_ok,
            "ready": self._config.enabled and piper_ok,
        }

    def speak(self, text: str) -> VoiceOutputResult:
        """
        Speak text aloud.

        Generates audio and plays it immediately.

        Args:
            text: The formatted text to speak

        Returns:
            VoiceOutputResult indicating success/failure

        Note:
            Never raises exceptions. Failures are returned as results.
        """
        if not self._config.enabled:
            return VoiceOutputResult(
                success=False,
                error_code=VoiceOutputErrorCode.NOT_ENABLED.value,
                error_message=USER_ERROR_MESSAGES[VoiceOutputErrorCode.NOT_ENABLED],
            )

        if not text or not text.strip():
            return VoiceOutputResult(
                success=False,
                error_code=VoiceOutputErrorCode.EMPTY_TEXT.value,
                error_message=USER_ERROR_MESSAGES[VoiceOutputErrorCode.EMPTY_TEXT],
            )

        # Generate audio
        result = self.generate_audio(text)
        if not result.success:
            return result

        # Play audio
        play_result = self._play_audio(result.audio_data)
        if not play_result.success:
            # Return the play error but include the audio data
            play_result.audio_data = result.audio_data
            play_result.duration_seconds = result.duration_seconds
            return play_result

        return result

    def generate_audio(self, text: str) -> VoiceOutputResult:
        """
        Generate audio from text without playing.

        Args:
            text: The formatted text to convert

        Returns:
            VoiceOutputResult with audio_data if successful
        """
        if not self._config.enabled:
            return VoiceOutputResult(
                success=False,
                error_code=VoiceOutputErrorCode.NOT_ENABLED.value,
                error_message=USER_ERROR_MESSAGES[VoiceOutputErrorCode.NOT_ENABLED],
            )

        if not text or not text.strip():
            return VoiceOutputResult(
                success=False,
                error_code=VoiceOutputErrorCode.EMPTY_TEXT.value,
                error_message=USER_ERROR_MESSAGES[VoiceOutputErrorCode.EMPTY_TEXT],
            )

        piper_ok, self._piper_path = _check_piper_available()
        if not piper_ok:
            return VoiceOutputResult(
                success=False,
                error_code=VoiceOutputErrorCode.PIPER_UNAVAILABLE.value,
                error_message=USER_ERROR_MESSAGES[VoiceOutputErrorCode.PIPER_UNAVAILABLE],
            )

        # Clean text for TTS
        clean_text = self._prepare_text_for_tts(text)

        if self._piper_path == "python-piper":
            return self._generate_with_python_piper(clean_text)
        else:
            return self._generate_with_piper_cli(clean_text)

    def _prepare_text_for_tts(self, text: str) -> str:
        """
        Prepare text for TTS processing.

        Cleans and normalizes text for better speech synthesis.
        """
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Remove markdown-style formatting that shouldn't be spoken
        # (basic cleanup only - complex markdown would need more processing)
        text = text.replace("```", " ")
        text = text.replace("`", "")
        text = text.replace("**", "")
        text = text.replace("__", "")
        text = text.replace("##", "")
        text = text.replace("#", "")

        return text.strip()

    def _generate_with_piper_cli(self, text: str) -> VoiceOutputResult:
        """
        Generate audio using Piper CLI.

        Creates a temporary WAV file, runs piper, reads the audio.
        """
        temp_input = None
        temp_output = None

        try:
            # Write text to temp file (piper reads from stdin or file)
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(text)
                temp_input = f.name

            # Create temp output file
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as f:
                temp_output = f.name

            # Build piper command
            cmd = [
                self._piper_path,
                "--model", self._config.voice,
                "--output_file", temp_output,
            ]

            # Add speed if not default
            if self._config.speed != 1.0:
                cmd.extend(["--length_scale", str(1.0 / self._config.speed)])

            # Run piper with text as stdin
            with open(temp_input, "r", encoding="utf-8") as f:
                result = subprocess.run(
                    cmd,
                    stdin=f,
                    capture_output=True,
                    timeout=60,
                )

            if result.returncode != 0:
                return VoiceOutputResult(
                    success=False,
                    error_code=VoiceOutputErrorCode.TTS_FAILED.value,
                    error_message=USER_ERROR_MESSAGES[VoiceOutputErrorCode.TTS_FAILED],
                )

            # Read generated audio
            if os.path.exists(temp_output):
                with open(temp_output, "rb") as f:
                    audio_data = f.read()

                # Estimate duration from WAV file
                duration = self._estimate_wav_duration(audio_data)

                return VoiceOutputResult(
                    success=True,
                    audio_data=audio_data,
                    duration_seconds=duration,
                )
            else:
                return VoiceOutputResult(
                    success=False,
                    error_code=VoiceOutputErrorCode.TTS_FAILED.value,
                    error_message=USER_ERROR_MESSAGES[VoiceOutputErrorCode.TTS_FAILED],
                )

        except subprocess.TimeoutExpired:
            return VoiceOutputResult(
                success=False,
                error_code=VoiceOutputErrorCode.TTS_FAILED.value,
                error_message="Speech generation timed out.",
            )
        except Exception:
            return VoiceOutputResult(
                success=False,
                error_code=VoiceOutputErrorCode.TTS_FAILED.value,
                error_message=USER_ERROR_MESSAGES[VoiceOutputErrorCode.TTS_FAILED],
            )
        finally:
            # Clean up temp files
            for path in [temp_input, temp_output]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception:
                        pass

    def _generate_with_python_piper(self, text: str) -> VoiceOutputResult:
        """
        Generate audio using Python piper module.
        """
        try:
            import piper

            # Load voice model
            voice = piper.PiperVoice.load(self._config.voice)

            # Generate audio
            audio_data = b""
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(voice.config.sample_rate)

                    # Synthesize
                    for audio_bytes in voice.synthesize_stream_raw(text):
                        wav.writeframes(audio_bytes)

                audio_data = wav_io.getvalue()

            duration = self._estimate_wav_duration(audio_data)

            return VoiceOutputResult(
                success=True,
                audio_data=audio_data,
                duration_seconds=duration,
            )

        except Exception:
            return VoiceOutputResult(
                success=False,
                error_code=VoiceOutputErrorCode.TTS_FAILED.value,
                error_message=USER_ERROR_MESSAGES[VoiceOutputErrorCode.TTS_FAILED],
            )

    def _estimate_wav_duration(self, audio_data: bytes) -> float:
        """
        Estimate duration of WAV audio data.
        """
        try:
            with io.BytesIO(audio_data) as wav_io:
                with wave.open(wav_io, "rb") as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    if rate > 0:
                        return frames / rate
        except Exception:
            pass
        return 0.0

    def _play_audio(self, audio_data: bytes) -> VoiceOutputResult:
        """
        Play audio data.

        Tries multiple playback methods.
        """
        if not audio_data:
            return VoiceOutputResult(
                success=False,
                error_code=VoiceOutputErrorCode.PLAYBACK_FAILED.value,
                error_message="No audio data to play.",
            )

        # Try pygame first
        try:
            return self._play_with_pygame(audio_data)
        except Exception:
            pass

        # Try simpleaudio
        try:
            return self._play_with_simpleaudio(audio_data)
        except Exception:
            pass

        # Try sounddevice
        try:
            return self._play_with_sounddevice(audio_data)
        except Exception:
            pass

        # Try system command (write to temp file)
        try:
            return self._play_with_system_command(audio_data)
        except Exception:
            pass

        return VoiceOutputResult(
            success=False,
            error_code=VoiceOutputErrorCode.AUDIO_SYSTEM_UNAVAILABLE.value,
            error_message=USER_ERROR_MESSAGES[VoiceOutputErrorCode.AUDIO_SYSTEM_UNAVAILABLE],
        )

    def _play_with_pygame(self, audio_data: bytes) -> VoiceOutputResult:
        """Play audio using pygame."""
        import pygame

        pygame.mixer.init()
        try:
            sound = pygame.mixer.Sound(io.BytesIO(audio_data))
            sound.play()

            # Wait for playback to complete
            while pygame.mixer.get_busy():
                pygame.time.wait(100)

            duration = sound.get_length()
            return VoiceOutputResult(
                success=True,
                audio_data=audio_data,
                duration_seconds=duration,
            )
        finally:
            pygame.mixer.quit()

    def _play_with_simpleaudio(self, audio_data: bytes) -> VoiceOutputResult:
        """Play audio using simpleaudio."""
        import simpleaudio

        with io.BytesIO(audio_data) as wav_io:
            wav = wave.open(wav_io, "rb")
            play_obj = simpleaudio.play_buffer(
                wav.readframes(wav.getnframes()),
                num_channels=wav.getnchannels(),
                bytes_per_sample=wav.getsampwidth(),
                sample_rate=wav.getframerate(),
            )
            play_obj.wait_done()

        duration = self._estimate_wav_duration(audio_data)
        return VoiceOutputResult(
            success=True,
            audio_data=audio_data,
            duration_seconds=duration,
        )

    def _play_with_sounddevice(self, audio_data: bytes) -> VoiceOutputResult:
        """Play audio using sounddevice."""
        import sounddevice as sd
        import numpy as np

        with io.BytesIO(audio_data) as wav_io:
            with wave.open(wav_io, "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                audio_array = np.frombuffer(frames, dtype=np.int16)
                sd.play(audio_array, wav.getframerate())
                sd.wait()

        duration = self._estimate_wav_duration(audio_data)
        return VoiceOutputResult(
            success=True,
            audio_data=audio_data,
            duration_seconds=duration,
        )

    def _play_with_system_command(self, audio_data: bytes) -> VoiceOutputResult:
        """Play audio using system command."""
        import platform

        temp_file = None
        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as f:
                f.write(audio_data)
                temp_file = f.name

            system = platform.system()

            if system == "Windows":
                # Use Windows built-in player
                import winsound
                winsound.PlaySound(temp_file, winsound.SND_FILENAME)
            elif system == "Darwin":
                # macOS
                subprocess.run(["afplay", temp_file], check=True)
            else:
                # Linux - try aplay or paplay
                try:
                    subprocess.run(["aplay", temp_file], check=True)
                except FileNotFoundError:
                    subprocess.run(["paplay", temp_file], check=True)

            duration = self._estimate_wav_duration(audio_data)
            return VoiceOutputResult(
                success=True,
                audio_data=audio_data,
                duration_seconds=duration,
            )

        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
