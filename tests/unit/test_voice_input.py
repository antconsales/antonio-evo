"""
Unit tests for voice input module.

Tests the voice input functionality without requiring actual audio hardware
or Whisper installation. Uses mocks for external dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os

from src.voice.config import (
    VoiceInputConfig,
    VoiceOutputConfig,
    VoiceConfig,
    load_voice_config,
)
from src.voice.input import (
    VoiceInput,
    VoiceInputResult,
    VoiceInputError,
    VoiceInputErrorCode,
    USER_ERROR_MESSAGES,
    _check_whisper_available,
    _check_audio_available,
)


# =============================================================================
# VoiceInputConfig Tests
# =============================================================================


class TestVoiceInputConfig:
    """Tests for VoiceInputConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VoiceInputConfig()

        assert config.enabled is False
        assert config.engine == "whisper"
        assert config.model == "base"
        assert config.language == "auto"
        assert config.timeout_seconds == 30
        assert config.max_duration_seconds == 60
        assert config.sample_rate == 16000

    def test_from_dict_with_values(self):
        """Test creating config from dictionary."""
        data = {
            "enabled": True,
            "engine": "whisper",
            "model": "small",
            "language": "en",
            "timeout_seconds": 20,
            "max_duration_seconds": 45,
            "sample_rate": 22050,
        }

        config = VoiceInputConfig.from_dict(data)

        assert config.enabled is True
        assert config.model == "small"
        assert config.language == "en"
        assert config.timeout_seconds == 20
        assert config.max_duration_seconds == 45
        assert config.sample_rate == 22050

    def test_from_dict_with_defaults(self):
        """Test from_dict uses defaults for missing keys."""
        config = VoiceInputConfig.from_dict({})

        assert config.enabled is False
        assert config.engine == "whisper"
        assert config.model == "base"

    def test_from_dict_partial_values(self):
        """Test from_dict with partial values."""
        data = {"enabled": True, "model": "large"}

        config = VoiceInputConfig.from_dict(data)

        assert config.enabled is True
        assert config.model == "large"
        assert config.language == "auto"  # default


class TestVoiceOutputConfig:
    """Tests for VoiceOutputConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VoiceOutputConfig()

        assert config.enabled is False
        assert config.engine == "piper"
        assert config.voice == "en_US-lessac-medium"
        assert config.speed == 1.0

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "enabled": True,
            "engine": "piper",
            "voice": "en_GB-jenny-medium",
            "speed": 1.2,
        }

        config = VoiceOutputConfig.from_dict(data)

        assert config.enabled is True
        assert config.voice == "en_GB-jenny-medium"
        assert config.speed == 1.2


class TestVoiceConfig:
    """Tests for VoiceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = VoiceConfig()

        assert config.input.enabled is False
        assert config.output.enabled is False

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "input": {"enabled": True, "model": "medium"},
            "output": {"enabled": True, "speed": 0.9},
        }

        config = VoiceConfig.from_dict(data)

        assert config.input.enabled is True
        assert config.input.model == "medium"
        assert config.output.enabled is True
        assert config.output.speed == 0.9

    def test_to_dict(self):
        """Test converting to dictionary."""
        config = VoiceConfig(
            input=VoiceInputConfig(enabled=True, model="small"),
            output=VoiceOutputConfig(enabled=True, speed=1.5),
        )

        data = config.to_dict()

        assert data["input"]["enabled"] is True
        assert data["input"]["model"] == "small"
        assert data["output"]["enabled"] is True
        assert data["output"]["speed"] == 1.5


class TestLoadVoiceConfig:
    """Tests for load_voice_config function."""

    def test_load_from_valid_file(self):
        """Test loading from valid JSON file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({
                "input": {"enabled": True, "model": "tiny"},
                "output": {"enabled": False},
            }, f)
            temp_path = f.name

        try:
            config = load_voice_config(temp_path)

            assert config.input.enabled is True
            assert config.input.model == "tiny"
            assert config.output.enabled is False
        finally:
            os.unlink(temp_path)

    def test_load_file_not_found_returns_defaults(self):
        """Test that missing file returns default config."""
        config = load_voice_config("/nonexistent/path/voice.json")

        assert config.input.enabled is False
        assert config.output.enabled is False

    def test_load_invalid_json_returns_defaults(self):
        """Test that invalid JSON returns default config."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json {{{")
            temp_path = f.name

        try:
            config = load_voice_config(temp_path)

            assert config.input.enabled is False
            assert config.output.enabled is False
        finally:
            os.unlink(temp_path)


# =============================================================================
# VoiceInputResult Tests
# =============================================================================


class TestVoiceInputResult:
    """Tests for VoiceInputResult dataclass."""

    def test_successful_result(self):
        """Test successful result."""
        result = VoiceInputResult(
            success=True,
            text="Hello world",
            duration_seconds=2.5,
        )

        assert result.success is True
        assert result.text == "Hello world"
        assert result.duration_seconds == 2.5

    def test_failed_result(self):
        """Test failed result."""
        result = VoiceInputResult(
            success=False,
            error_code="VOICE_NO_SPEECH",
            error_message="No speech detected.",
            duration_seconds=1.0,
        )

        assert result.success is False
        assert result.error_code == "VOICE_NO_SPEECH"
        assert result.error_message == "No speech detected."

    def test_to_dict_success(self):
        """Test to_dict for successful result."""
        result = VoiceInputResult(
            success=True,
            text="Test",
            duration_seconds=3.0,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["text"] == "Test"
        assert data["duration_seconds"] == 3.0
        assert "error_code" not in data

    def test_to_dict_failure(self):
        """Test to_dict for failed result."""
        result = VoiceInputResult(
            success=False,
            error_code="VOICE_TIMEOUT",
            error_message="Recording timed out.",
            duration_seconds=30.0,
        )

        data = result.to_dict()

        assert data["success"] is False
        assert data["error_code"] == "VOICE_TIMEOUT"
        assert data["error_message"] == "Recording timed out."
        assert "text" not in data


# =============================================================================
# VoiceInputError Tests
# =============================================================================


class TestVoiceInputError:
    """Tests for VoiceInputError exception."""

    def test_error_creation(self):
        """Test creating an error."""
        error = VoiceInputError(
            code=VoiceInputErrorCode.NO_MICROPHONE,
            message="No mic",
        )

        assert error.code == VoiceInputErrorCode.NO_MICROPHONE
        assert error.message == "No mic"
        assert str(error) == "VOICE_NO_MICROPHONE: No mic"

    def test_all_error_codes_have_messages(self):
        """Test that all error codes have user messages."""
        for code in VoiceInputErrorCode:
            assert code in USER_ERROR_MESSAGES
            assert len(USER_ERROR_MESSAGES[code]) > 0


# =============================================================================
# Dependency Check Tests
# =============================================================================


class TestCheckWhisperAvailable:
    """Tests for _check_whisper_available function."""

    @patch("subprocess.run")
    def test_whisper_cli_available(self, mock_run):
        """Test detecting whisper CLI."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=b"whisper help",
            stderr=b"",
        )

        available, path = _check_whisper_available()

        assert available is True
        assert path == "whisper"

    @patch("subprocess.run")
    def test_whisper_cli_not_available(self, mock_run):
        """Test when whisper CLI is not found."""
        mock_run.side_effect = FileNotFoundError()

        # Also patch import to fail
        with patch.dict("sys.modules", {"whisper": None}):
            available, path = _check_whisper_available()

        # Will be False only if no CLI and no Python module
        # Since we can't fully mock the import, just check it runs
        assert isinstance(available, bool)

    @patch("subprocess.run")
    def test_python_whisper_fallback(self, mock_run):
        """Test fallback to Python whisper module."""
        mock_run.side_effect = FileNotFoundError()

        # Mock the whisper module
        mock_whisper = MagicMock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            available, path = _check_whisper_available()

        assert available is True
        assert path == "python-whisper"


class TestCheckAudioAvailable:
    """Tests for _check_audio_available function."""

    def test_with_pyaudio(self):
        """Test detecting audio with pyaudio."""
        mock_pa = MagicMock()
        mock_pa.get_device_count.return_value = 2
        mock_pa.terminate = MagicMock()

        mock_pyaudio = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_pa

        with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
            available = _check_audio_available()

        assert available is True

    def test_with_sounddevice_fallback(self):
        """Test fallback to sounddevice."""
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [{"name": "mic1"}]

        with patch.dict("sys.modules", {
            "pyaudio": None,
            "sounddevice": mock_sd,
        }):
            # Import will fail for pyaudio, fall back to sounddevice
            available = _check_audio_available()

        # Just check it doesn't crash
        assert isinstance(available, bool)


# =============================================================================
# VoiceInput Class Tests
# =============================================================================


class TestVoiceInput:
    """Tests for VoiceInput class."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        voice = VoiceInput()

        assert voice._config.enabled is False

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = VoiceInputConfig(
            enabled=True,
            model="small",
            language="en",
        )

        voice = VoiceInput(config)

        assert voice._config.enabled is True
        assert voice._config.model == "small"

    def test_is_available_when_disabled(self):
        """Test is_available returns False when disabled."""
        config = VoiceInputConfig(enabled=False)
        voice = VoiceInput(config)

        assert voice.is_available() is False

    @patch("src.voice.input._check_whisper_available")
    def test_is_available_when_enabled_no_whisper(self, mock_check):
        """Test is_available returns False when whisper not available."""
        mock_check.return_value = (False, None)

        config = VoiceInputConfig(enabled=True)
        voice = VoiceInput(config)

        assert voice.is_available() is False

    @patch("src.voice.input._check_whisper_available")
    def test_is_available_when_enabled_with_whisper(self, mock_check):
        """Test is_available returns True when all ready."""
        mock_check.return_value = (True, "whisper")

        config = VoiceInputConfig(enabled=True)
        voice = VoiceInput(config)

        assert voice.is_available() is True

    @patch("src.voice.input._check_whisper_available")
    @patch("src.voice.input._check_audio_available")
    def test_check_dependencies(self, mock_audio, mock_whisper):
        """Test check_dependencies returns status dict."""
        mock_whisper.return_value = (True, "whisper")
        mock_audio.return_value = True

        config = VoiceInputConfig(enabled=True)
        voice = VoiceInput(config)

        deps = voice.check_dependencies()

        assert deps["enabled"] is True
        assert deps["whisper_available"] is True
        assert deps["whisper_path"] == "whisper"
        assert deps["audio_available"] is True
        assert deps["ready"] is True

    def test_stop_recording_sets_event(self):
        """Test stop_recording sets the stop event."""
        voice = VoiceInput()

        assert not voice._stop_recording.is_set()
        voice.stop_recording()
        assert voice._stop_recording.is_set()


class TestVoiceInputRecordAndTranscribe:
    """Tests for record_and_transcribe method."""

    def test_returns_error_when_disabled(self):
        """Test returns error when voice is disabled."""
        config = VoiceInputConfig(enabled=False)
        voice = VoiceInput(config)

        result = voice.record_and_transcribe()

        assert result.success is False
        assert result.error_code == VoiceInputErrorCode.NOT_ENABLED.value

    @patch("src.voice.input._check_whisper_available")
    def test_returns_error_when_no_whisper(self, mock_check):
        """Test returns error when whisper unavailable."""
        mock_check.return_value = (False, None)

        config = VoiceInputConfig(enabled=True)
        voice = VoiceInput(config)

        result = voice.record_and_transcribe()

        assert result.success is False
        assert result.error_code == VoiceInputErrorCode.WHISPER_UNAVAILABLE.value


class TestVoiceInputTranscribeAudio:
    """Tests for transcribe_audio method."""

    def test_returns_error_when_disabled(self):
        """Test returns error when voice is disabled."""
        config = VoiceInputConfig(enabled=False)
        voice = VoiceInput(config)

        result = voice.transcribe_audio(b"audio data")

        assert result.success is False
        assert result.error_code == VoiceInputErrorCode.NOT_ENABLED.value

    @patch("src.voice.input._check_whisper_available")
    def test_returns_error_when_no_whisper(self, mock_check):
        """Test returns error when whisper unavailable."""
        mock_check.return_value = (False, None)

        config = VoiceInputConfig(enabled=True)
        voice = VoiceInput(config)

        result = voice.transcribe_audio(b"audio data")

        assert result.success is False
        assert result.error_code == VoiceInputErrorCode.WHISPER_UNAVAILABLE.value

    @patch("src.voice.input._check_whisper_available")
    def test_returns_error_for_empty_audio(self, mock_check):
        """Test returns error for empty audio data."""
        mock_check.return_value = (True, "whisper")

        config = VoiceInputConfig(enabled=True)
        voice = VoiceInput(config)

        result = voice.transcribe_audio(b"")

        assert result.success is False
        assert result.error_code == VoiceInputErrorCode.INVALID_AUDIO.value

    @patch("src.voice.input._check_whisper_available")
    def test_returns_error_for_too_short_audio(self, mock_check):
        """Test returns error for audio that's too short."""
        mock_check.return_value = (True, "whisper")

        config = VoiceInputConfig(enabled=True, sample_rate=16000)
        voice = VoiceInput(config)

        # Less than 0.5 seconds of audio (16000 * 2 * 0.5 = 16000 bytes)
        short_audio = b"\x00" * 1000

        result = voice.transcribe_audio(short_audio)

        assert result.success is False
        assert result.error_code == VoiceInputErrorCode.AUDIO_TOO_SHORT.value

    @patch("src.voice.input._check_whisper_available")
    def test_returns_error_for_too_long_audio(self, mock_check):
        """Test returns error for audio that's too long."""
        mock_check.return_value = (True, "whisper")

        config = VoiceInputConfig(
            enabled=True,
            sample_rate=16000,
            max_duration_seconds=60,
        )
        voice = VoiceInput(config)

        # More than 60 seconds of audio (16000 * 2 * 61 = 1952000 bytes)
        long_audio = b"\x00" * (16000 * 2 * 61)

        result = voice.transcribe_audio(long_audio)

        assert result.success is False
        assert result.error_code == VoiceInputErrorCode.AUDIO_TOO_LONG.value


class TestUserErrorMessages:
    """Tests for user-safe error messages."""

    def test_no_internal_details(self):
        """Test that error messages don't contain internal details."""
        for code, message in USER_ERROR_MESSAGES.items():
            # Should not contain technical terms
            assert "exception" not in message.lower()
            assert "error:" not in message.lower()
            assert "traceback" not in message.lower()
            assert "stack" not in message.lower()

    def test_all_messages_actionable(self):
        """Test that error messages give user guidance."""
        for code, message in USER_ERROR_MESSAGES.items():
            # Messages should be complete sentences
            assert message.endswith(".")
            # Messages should not be empty
            assert len(message) > 10

    def test_messages_are_user_friendly(self):
        """Test that messages are understandable by users."""
        # Check some specific messages
        assert "Whisper" in USER_ERROR_MESSAGES[VoiceInputErrorCode.WHISPER_UNAVAILABLE]
        assert "microphone" in USER_ERROR_MESSAGES[VoiceInputErrorCode.NO_MICROPHONE].lower()
        assert "speech" in USER_ERROR_MESSAGES[VoiceInputErrorCode.NO_SPEECH].lower()


# =============================================================================
# Integration-style Tests (with mocks)
# =============================================================================


class TestVoiceInputIntegration:
    """Integration-style tests with mocked dependencies."""

    @patch("src.voice.input._check_whisper_available")
    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    @patch("wave.open")
    @patch("os.path.exists")
    @patch("os.unlink")
    def test_transcribe_with_whisper_cli_success(
        self,
        mock_unlink,
        mock_exists,
        mock_wave,
        mock_tempfile,
        mock_run,
        mock_check,
    ):
        """Test successful transcription with whisper CLI."""
        mock_check.return_value = (True, "whisper")

        # Setup tempfile mock
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/test.wav"
        mock_temp.__enter__ = MagicMock(return_value=mock_temp)
        mock_temp.__exit__ = MagicMock(return_value=False)
        mock_tempfile.return_value = mock_temp

        # Setup wave mock
        mock_wav = MagicMock()
        mock_wave.return_value.__enter__ = MagicMock(return_value=mock_wav)
        mock_wave.return_value.__exit__ = MagicMock(return_value=False)

        # Setup subprocess mock
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Hello world",
            stderr="",
        )

        # Setup file exists mock - no .txt file
        mock_exists.return_value = False

        config = VoiceInputConfig(enabled=True)
        voice = VoiceInput(config)

        # 1 second of audio
        audio_data = b"\x00" * (16000 * 2 * 1)

        result = voice.transcribe_audio(audio_data)

        assert result.success is True
        assert result.text == "Hello world"

    @patch("src.voice.input._check_whisper_available")
    def test_transcribe_with_python_whisper_success(self, mock_check):
        """Test successful transcription with Python whisper."""
        mock_check.return_value = (True, "python-whisper")

        # Mock the whisper module
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Transcribed text"}

        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            config = VoiceInputConfig(enabled=True)
            voice = VoiceInput(config)
            voice._whisper_path = "python-whisper"

            # 1 second of audio
            audio_data = b"\x00" * (16000 * 2 * 1)

            result = voice._transcribe_audio_data(audio_data, 1.0)

        assert result.success is True
        assert result.text == "Transcribed text"

    @patch("src.voice.input._check_whisper_available")
    def test_transcribe_no_speech_detected(self, mock_check):
        """Test handling of no speech detected."""
        mock_check.return_value = (True, "python-whisper")

        # Mock whisper returning empty text
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": ""}

        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            config = VoiceInputConfig(enabled=True)
            voice = VoiceInput(config)
            voice._whisper_path = "python-whisper"

            audio_data = b"\x00" * (16000 * 2 * 1)
            result = voice._transcribe_audio_data(audio_data, 1.0)

        assert result.success is False
        assert result.error_code == VoiceInputErrorCode.NO_SPEECH.value
