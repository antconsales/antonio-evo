"""
Voice Input - Speech-to-text for Antonio.

Provides push-to-talk voice input using local Whisper for transcription.
All processing is local, CPU-only, and offline.

CORE PRINCIPLE:
Voice input produces plain text that enters the existing pipeline
through the Normalizer, exactly like typed text.

ACTIVATION:
Push-to-talk only. No always-on listening. No wake word.

PRIVACY:
- Audio is processed in memory only
- No audio written to disk by default
- No audio in logs
- Only transcribed text may appear in audit logs (truncated)
"""

import io
import os
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import VoiceInputConfig


class VoiceInputErrorCode(Enum):
    """Error codes for voice input operations."""
    NOT_ENABLED = "VOICE_NOT_ENABLED"
    NO_MICROPHONE = "VOICE_NO_MICROPHONE"
    RECORDING_FAILED = "VOICE_RECORDING_FAILED"
    TRANSCRIPTION_FAILED = "VOICE_TRANSCRIPTION_FAILED"
    WHISPER_UNAVAILABLE = "VOICE_WHISPER_UNAVAILABLE"
    TIMEOUT = "VOICE_TIMEOUT"
    NO_SPEECH = "VOICE_NO_SPEECH"
    AUDIO_TOO_SHORT = "VOICE_AUDIO_TOO_SHORT"
    AUDIO_TOO_LONG = "VOICE_AUDIO_TOO_LONG"
    INVALID_AUDIO = "VOICE_INVALID_AUDIO"


class VoiceInputError(Exception):
    """Exception for voice input errors."""

    def __init__(self, code: VoiceInputErrorCode, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code.value}: {message}")


@dataclass
class VoiceInputResult:
    """
    Result of voice input operation.

    Attributes:
        success: Whether transcription succeeded
        text: Transcribed text (if successful)
        error_code: Error code (if failed)
        error_message: User-safe error message (if failed)
        duration_seconds: Duration of recorded audio
    """
    success: bool
    text: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "success": self.success,
            "duration_seconds": self.duration_seconds,
        }
        if self.success:
            result["text"] = self.text
        else:
            result["error_code"] = self.error_code
            result["error_message"] = self.error_message
        return result


# User-safe error messages (no internal details)
USER_ERROR_MESSAGES = {
    VoiceInputErrorCode.NOT_ENABLED: "Voice input is not enabled.",
    VoiceInputErrorCode.NO_MICROPHONE: "No microphone detected. Please connect a microphone.",
    VoiceInputErrorCode.RECORDING_FAILED: "Failed to record audio. Please try again.",
    VoiceInputErrorCode.TRANSCRIPTION_FAILED: "Could not transcribe audio. Please try again.",
    VoiceInputErrorCode.WHISPER_UNAVAILABLE: "Voice input is unavailable. Whisper is not installed.",
    VoiceInputErrorCode.TIMEOUT: "Recording timed out. Please try again.",
    VoiceInputErrorCode.NO_SPEECH: "No speech detected. Please speak clearly.",
    VoiceInputErrorCode.AUDIO_TOO_SHORT: "Recording was too short. Please speak longer.",
    VoiceInputErrorCode.AUDIO_TOO_LONG: "Recording was too long. Please keep it brief.",
    VoiceInputErrorCode.INVALID_AUDIO: "Audio format not recognized.",
}


def _check_whisper_available() -> Tuple[bool, Optional[str]]:
    """
    Check if Whisper is available for transcription.

    Looks for whisper CLI tool (whisper.cpp or openai-whisper).

    Returns:
        Tuple of (available, executable_path)
    """
    # Try common whisper executables
    whisper_names = ["whisper", "whisper-cli", "whisper.cpp", "main"]

    for name in whisper_names:
        try:
            result = subprocess.run(
                [name, "--help"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0 or b"whisper" in result.stdout.lower() or b"whisper" in result.stderr.lower():
                return True, name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        except Exception:
            continue

    # Try Python whisper module
    try:
        import whisper
        return True, "python-whisper"
    except ImportError:
        pass

    return False, None


def _check_audio_available() -> bool:
    """
    Check if audio recording is available.

    Returns:
        True if audio recording is possible
    """
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        device_count = pa.get_device_count()
        pa.terminate()
        return device_count > 0
    except ImportError:
        pass
    except Exception:
        pass

    # Try sounddevice as fallback
    try:
        import sounddevice
        devices = sounddevice.query_devices()
        return len(devices) > 0
    except ImportError:
        pass
    except Exception:
        pass

    return False


class VoiceInput:
    """
    Voice input handler using local Whisper.

    Provides push-to-talk voice input that transcribes speech to text.
    The resulting text is processed exactly like typed input.

    Usage:
        voice = VoiceInput(config)

        if voice.is_available():
            # Push-to-talk: record while key held
            result = voice.record_and_transcribe()

            if result.success:
                # Submit result.text to pipeline
                process(result.text)

        # Or from audio bytes (for API)
        result = voice.transcribe_audio(audio_bytes)
    """

    def __init__(self, config: Optional[VoiceInputConfig] = None):
        """
        Initialize voice input.

        Args:
            config: Voice input configuration (uses defaults if None)
        """
        self._config = config or VoiceInputConfig()
        self._whisper_path: Optional[str] = None
        self._recording = False
        self._stop_recording = threading.Event()

    def is_available(self) -> bool:
        """
        Check if voice input is available.

        Returns:
            True if voice input can be used

        Note:
            Returns False if not enabled in config.
        """
        if not self._config.enabled:
            return False

        whisper_ok, self._whisper_path = _check_whisper_available()
        if not whisper_ok:
            return False

        return True

    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check all voice input dependencies.

        Returns:
            Dictionary with dependency status
        """
        whisper_ok, whisper_path = _check_whisper_available()
        audio_ok = _check_audio_available()

        return {
            "enabled": self._config.enabled,
            "whisper_available": whisper_ok,
            "whisper_path": whisper_path,
            "audio_available": audio_ok,
            "ready": self._config.enabled and whisper_ok and audio_ok,
        }

    def record_and_transcribe(
        self,
        on_recording_start: Optional[Callable[[], None]] = None,
        on_recording_stop: Optional[Callable[[], None]] = None,
    ) -> VoiceInputResult:
        """
        Record audio and transcribe to text (push-to-talk).

        This is a blocking call that:
        1. Starts recording
        2. Records until stop_recording() is called or timeout
        3. Transcribes the audio
        4. Returns the text

        Args:
            on_recording_start: Callback when recording starts
            on_recording_stop: Callback when recording stops

        Returns:
            VoiceInputResult with transcribed text or error
        """
        if not self._config.enabled:
            return VoiceInputResult(
                success=False,
                error_code=VoiceInputErrorCode.NOT_ENABLED.value,
                error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.NOT_ENABLED],
            )

        # Check dependencies
        whisper_ok, self._whisper_path = _check_whisper_available()
        if not whisper_ok:
            return VoiceInputResult(
                success=False,
                error_code=VoiceInputErrorCode.WHISPER_UNAVAILABLE.value,
                error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.WHISPER_UNAVAILABLE],
            )

        # Record audio
        audio_data, duration, error = self._record_audio(
            on_start=on_recording_start,
            on_stop=on_recording_stop,
        )

        if error:
            return error

        # Transcribe
        return self._transcribe_audio_data(audio_data, duration)

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
    ) -> VoiceInputResult:
        """
        Transcribe audio bytes to text.

        Used for API endpoint where audio is received as bytes.

        Args:
            audio_bytes: Raw audio data (WAV format expected)
            sample_rate: Audio sample rate

        Returns:
            VoiceInputResult with transcribed text or error
        """
        if not self._config.enabled:
            return VoiceInputResult(
                success=False,
                error_code=VoiceInputErrorCode.NOT_ENABLED.value,
                error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.NOT_ENABLED],
            )

        whisper_ok, self._whisper_path = _check_whisper_available()
        if not whisper_ok:
            return VoiceInputResult(
                success=False,
                error_code=VoiceInputErrorCode.WHISPER_UNAVAILABLE.value,
                error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.WHISPER_UNAVAILABLE],
            )

        if not audio_bytes:
            return VoiceInputResult(
                success=False,
                error_code=VoiceInputErrorCode.INVALID_AUDIO.value,
                error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.INVALID_AUDIO],
            )

        # Estimate duration from audio size (16-bit mono at sample_rate)
        duration = len(audio_bytes) / (sample_rate * 2)

        if duration < 0.5:
            return VoiceInputResult(
                success=False,
                error_code=VoiceInputErrorCode.AUDIO_TOO_SHORT.value,
                error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.AUDIO_TOO_SHORT],
                duration_seconds=duration,
            )

        if duration > self._config.max_duration_seconds:
            return VoiceInputResult(
                success=False,
                error_code=VoiceInputErrorCode.AUDIO_TOO_LONG.value,
                error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.AUDIO_TOO_LONG],
                duration_seconds=duration,
            )

        return self._transcribe_audio_data(audio_bytes, duration)

    def stop_recording(self) -> None:
        """
        Stop the current recording.

        Call this when the push-to-talk key is released.
        """
        self._stop_recording.set()

    def _record_audio(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
    ) -> Tuple[Optional[bytes], float, Optional[VoiceInputResult]]:
        """
        Record audio from microphone.

        Returns:
            Tuple of (audio_bytes, duration, error_result)
        """
        try:
            import pyaudio
        except ImportError:
            try:
                import sounddevice as sd
                import numpy as np
                return self._record_with_sounddevice(on_start, on_stop)
            except ImportError:
                return None, 0, VoiceInputResult(
                    success=False,
                    error_code=VoiceInputErrorCode.NO_MICROPHONE.value,
                    error_message="Audio recording library not available. Install pyaudio or sounddevice.",
                )

        # Use PyAudio for recording
        self._stop_recording.clear()
        self._recording = True

        try:
            pa = pyaudio.PyAudio()

            # Find input device
            device_count = pa.get_device_count()
            if device_count == 0:
                pa.terminate()
                return None, 0, VoiceInputResult(
                    success=False,
                    error_code=VoiceInputErrorCode.NO_MICROPHONE.value,
                    error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.NO_MICROPHONE],
                )

            # Open stream
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._config.sample_rate,
                input=True,
                frames_per_buffer=1024,
            )

            if on_start:
                on_start()

            frames: List[bytes] = []
            start_time = time.time()
            timeout = self._config.timeout_seconds

            # Record until stopped or timeout
            while not self._stop_recording.is_set():
                elapsed = time.time() - start_time

                if elapsed >= timeout:
                    break

                if elapsed >= self._config.max_duration_seconds:
                    break

                try:
                    data = stream.read(1024, exception_on_overflow=False)
                    frames.append(data)
                except Exception:
                    break

            stream.stop_stream()
            stream.close()
            pa.terminate()

            if on_stop:
                on_stop()

            self._recording = False

            duration = time.time() - start_time
            audio_data = b"".join(frames)

            if duration < 0.5:
                return None, duration, VoiceInputResult(
                    success=False,
                    error_code=VoiceInputErrorCode.AUDIO_TOO_SHORT.value,
                    error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.AUDIO_TOO_SHORT],
                    duration_seconds=duration,
                )

            return audio_data, duration, None

        except Exception as e:
            self._recording = False
            return None, 0, VoiceInputResult(
                success=False,
                error_code=VoiceInputErrorCode.RECORDING_FAILED.value,
                error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.RECORDING_FAILED],
            )

    def _record_with_sounddevice(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
    ) -> Tuple[Optional[bytes], float, Optional[VoiceInputResult]]:
        """
        Record audio using sounddevice library.

        Fallback when pyaudio is not available.
        """
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            return None, 0, VoiceInputResult(
                success=False,
                error_code=VoiceInputErrorCode.NO_MICROPHONE.value,
                error_message="Audio library not available.",
            )

        self._stop_recording.clear()
        self._recording = True

        try:
            if on_start:
                on_start()

            frames: List[np.ndarray] = []
            start_time = time.time()

            def callback(indata, frames_count, time_info, status):
                if not self._stop_recording.is_set():
                    frames.append(indata.copy())

            with sd.InputStream(
                samplerate=self._config.sample_rate,
                channels=1,
                dtype=np.int16,
                callback=callback,
            ):
                while not self._stop_recording.is_set():
                    elapsed = time.time() - start_time
                    if elapsed >= self._config.timeout_seconds:
                        break
                    if elapsed >= self._config.max_duration_seconds:
                        break
                    time.sleep(0.1)

            if on_stop:
                on_stop()

            self._recording = False

            duration = time.time() - start_time

            if len(frames) == 0 or duration < 0.5:
                return None, duration, VoiceInputResult(
                    success=False,
                    error_code=VoiceInputErrorCode.AUDIO_TOO_SHORT.value,
                    error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.AUDIO_TOO_SHORT],
                    duration_seconds=duration,
                )

            audio_array = np.concatenate(frames, axis=0)
            audio_bytes = audio_array.tobytes()

            return audio_bytes, duration, None

        except Exception:
            self._recording = False
            return None, 0, VoiceInputResult(
                success=False,
                error_code=VoiceInputErrorCode.RECORDING_FAILED.value,
                error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.RECORDING_FAILED],
            )

    def _transcribe_audio_data(
        self,
        audio_data: bytes,
        duration: float,
    ) -> VoiceInputResult:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            duration: Audio duration in seconds

        Returns:
            VoiceInputResult with text or error
        """
        if self._whisper_path == "python-whisper":
            return self._transcribe_with_python_whisper(audio_data, duration)
        else:
            return self._transcribe_with_whisper_cli(audio_data, duration)

    def _transcribe_with_whisper_cli(
        self,
        audio_data: bytes,
        duration: float,
    ) -> VoiceInputResult:
        """
        Transcribe using whisper CLI (whisper.cpp).

        Creates a temporary WAV file, runs whisper, deletes the file.
        """
        temp_wav = None
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as f:
                temp_wav = f.name
                # Write WAV header and data
                with wave.open(f, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(self._config.sample_rate)
                    wav.writeframes(audio_data)

            # Build whisper command
            cmd = [
                self._whisper_path,
                "-f", temp_wav,
                "-otxt",
                "--no-timestamps",
            ]

            if self._config.language != "auto":
                cmd.extend(["-l", self._config.language])

            # Run whisper
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self._config.timeout_seconds,
                text=True,
            )

            # Check for output file (whisper.cpp creates .txt file)
            txt_file = temp_wav + ".txt"
            text = ""

            if os.path.exists(txt_file):
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                os.unlink(txt_file)
            elif result.stdout:
                text = result.stdout.strip()

            if not text:
                return VoiceInputResult(
                    success=False,
                    error_code=VoiceInputErrorCode.NO_SPEECH.value,
                    error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.NO_SPEECH],
                    duration_seconds=duration,
                )

            return VoiceInputResult(
                success=True,
                text=text,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired:
            return VoiceInputResult(
                success=False,
                error_code=VoiceInputErrorCode.TIMEOUT.value,
                error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.TIMEOUT],
                duration_seconds=duration,
            )
        except Exception:
            return VoiceInputResult(
                success=False,
                error_code=VoiceInputErrorCode.TRANSCRIPTION_FAILED.value,
                error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.TRANSCRIPTION_FAILED],
                duration_seconds=duration,
            )
        finally:
            # Clean up temp file
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.unlink(temp_wav)
                except Exception:
                    pass

    def _transcribe_with_python_whisper(
        self,
        audio_data: bytes,
        duration: float,
    ) -> VoiceInputResult:
        """
        Transcribe using Python whisper library.
        """
        try:
            import whisper
            import numpy as np

            # Load model (cached after first load)
            model = whisper.load_model(self._config.model)

            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Transcribe
            language = None if self._config.language == "auto" else self._config.language
            result = model.transcribe(
                audio_float,
                language=language,
                fp16=False,  # CPU mode
            )

            text = result.get("text", "").strip()

            if not text:
                return VoiceInputResult(
                    success=False,
                    error_code=VoiceInputErrorCode.NO_SPEECH.value,
                    error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.NO_SPEECH],
                    duration_seconds=duration,
                )

            return VoiceInputResult(
                success=True,
                text=text,
                duration_seconds=duration,
            )

        except Exception:
            return VoiceInputResult(
                success=False,
                error_code=VoiceInputErrorCode.TRANSCRIPTION_FAILED.value,
                error_message=USER_ERROR_MESSAGES[VoiceInputErrorCode.TRANSCRIPTION_FAILED],
                duration_seconds=duration,
            )
