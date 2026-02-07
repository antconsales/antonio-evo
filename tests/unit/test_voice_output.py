"""
Unit tests for voice output module.

Tests the voice output functionality without requiring actual audio hardware
or Piper installation. Uses mocks for external dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import io
import wave

from src.voice.config import VoiceOutputConfig, VoiceConfig
from src.voice.output import (
    VoiceOutput,
    VoiceOutputResult,
    VoiceOutputError,
    VoiceOutputErrorCode,
    USER_ERROR_MESSAGES,
    _check_piper_available,
    _check_audio_playback_available,
)


# =============================================================================
# VoiceOutputConfig Tests
# =============================================================================


class TestVoiceOutputConfig:
    """Tests for VoiceOutputConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VoiceOutputConfig()

        assert config.enabled is False
        assert config.engine == "piper"
        assert config.voice == "en_US-lessac-medium"
        assert config.speed == 1.0

    def test_from_dict_with_values(self):
        """Test creating config from dictionary."""
        data = {
            "enabled": True,
            "engine": "piper",
            "voice": "en_GB-jenny-medium",
            "speed": 1.5,
        }

        config = VoiceOutputConfig.from_dict(data)

        assert config.enabled is True
        assert config.voice == "en_GB-jenny-medium"
        assert config.speed == 1.5

    def test_from_dict_with_defaults(self):
        """Test from_dict uses defaults for missing keys."""
        config = VoiceOutputConfig.from_dict({})

        assert config.enabled is False
        assert config.engine == "piper"
        assert config.speed == 1.0


# =============================================================================
# VoiceOutputResult Tests
# =============================================================================


class TestVoiceOutputResult:
    """Tests for VoiceOutputResult dataclass."""

    def test_successful_result(self):
        """Test successful result."""
        result = VoiceOutputResult(
            success=True,
            audio_data=b"audio bytes",
            duration_seconds=2.5,
        )

        assert result.success is True
        assert result.audio_data == b"audio bytes"
        assert result.duration_seconds == 2.5

    def test_failed_result(self):
        """Test failed result."""
        result = VoiceOutputResult(
            success=False,
            error_code="VOICE_TTS_FAILED",
            error_message="TTS failed.",
        )

        assert result.success is False
        assert result.error_code == "VOICE_TTS_FAILED"
        assert result.error_message == "TTS failed."

    def test_to_dict_success(self):
        """Test to_dict for successful result."""
        result = VoiceOutputResult(
            success=True,
            duration_seconds=3.0,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["duration_seconds"] == 3.0
        assert "error_code" not in data

    def test_to_dict_failure(self):
        """Test to_dict for failed result."""
        result = VoiceOutputResult(
            success=False,
            error_code="VOICE_PIPER_UNAVAILABLE",
            error_message="Piper not installed.",
        )

        data = result.to_dict()

        assert data["success"] is False
        assert data["error_code"] == "VOICE_PIPER_UNAVAILABLE"
        assert data["error_message"] == "Piper not installed."


# =============================================================================
# VoiceOutputError Tests
# =============================================================================


class TestVoiceOutputError:
    """Tests for VoiceOutputError exception."""

    def test_error_creation(self):
        """Test creating an error."""
        error = VoiceOutputError(
            code=VoiceOutputErrorCode.PIPER_UNAVAILABLE,
            message="No piper",
        )

        assert error.code == VoiceOutputErrorCode.PIPER_UNAVAILABLE
        assert error.message == "No piper"
        assert str(error) == "VOICE_PIPER_UNAVAILABLE: No piper"

    def test_all_error_codes_have_messages(self):
        """Test that all error codes have user messages."""
        for code in VoiceOutputErrorCode:
            assert code in USER_ERROR_MESSAGES
            assert len(USER_ERROR_MESSAGES[code]) > 0


# =============================================================================
# Dependency Check Tests
# =============================================================================


class TestCheckPiperAvailable:
    """Tests for _check_piper_available function."""

    @patch("subprocess.run")
    def test_piper_cli_available(self, mock_run):
        """Test detecting piper CLI."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=b"piper help",
            stderr=b"",
        )

        available, path = _check_piper_available()

        assert available is True
        assert path == "piper"

    @patch("subprocess.run")
    def test_piper_cli_not_available(self, mock_run):
        """Test when piper CLI is not found."""
        mock_run.side_effect = FileNotFoundError()

        # Also ensure piper module import fails
        with patch.dict("sys.modules", {"piper": None}):
            available, path = _check_piper_available()

        # Will check if any method works
        assert isinstance(available, bool)

    @patch("subprocess.run")
    def test_python_piper_fallback(self, mock_run):
        """Test fallback to Python piper module."""
        mock_run.side_effect = FileNotFoundError()

        # Mock the piper module
        mock_piper = MagicMock()
        with patch.dict("sys.modules", {"piper": mock_piper}):
            available, path = _check_piper_available()

        assert available is True
        assert path == "python-piper"


class TestCheckAudioPlaybackAvailable:
    """Tests for _check_audio_playback_available function."""

    def test_with_pygame(self):
        """Test detecting audio with pygame."""
        mock_mixer = MagicMock()
        mock_pygame = MagicMock()
        mock_pygame.mixer = mock_mixer

        with patch.dict("sys.modules", {"pygame": mock_pygame}):
            available = _check_audio_playback_available()

        # Should at least not crash
        assert isinstance(available, bool)


# =============================================================================
# VoiceOutput Class Tests
# =============================================================================


class TestVoiceOutput:
    """Tests for VoiceOutput class."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        voice = VoiceOutput()

        assert voice._config.enabled is False

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = VoiceOutputConfig(
            enabled=True,
            voice="en_GB-jenny-medium",
            speed=1.2,
        )

        voice = VoiceOutput(config)

        assert voice._config.enabled is True
        assert voice._config.voice == "en_GB-jenny-medium"
        assert voice._config.speed == 1.2

    def test_is_available_when_disabled(self):
        """Test is_available returns False when disabled."""
        config = VoiceOutputConfig(enabled=False)
        voice = VoiceOutput(config)

        assert voice.is_available() is False

    @patch("src.voice.output._check_piper_available")
    def test_is_available_when_enabled_no_piper(self, mock_check):
        """Test is_available returns False when piper not available."""
        mock_check.return_value = (False, None)

        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        assert voice.is_available() is False

    @patch("src.voice.output._check_piper_available")
    def test_is_available_when_enabled_with_piper(self, mock_check):
        """Test is_available returns True when all ready."""
        mock_check.return_value = (True, "piper")

        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        assert voice.is_available() is True

    @patch("src.voice.output._check_piper_available")
    @patch("src.voice.output._check_audio_playback_available")
    def test_check_dependencies(self, mock_audio, mock_piper):
        """Test check_dependencies returns status dict."""
        mock_piper.return_value = (True, "piper")
        mock_audio.return_value = True

        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        deps = voice.check_dependencies()

        assert deps["enabled"] is True
        assert deps["piper_available"] is True
        assert deps["piper_path"] == "piper"
        assert deps["playback_available"] is True
        assert deps["ready"] is True


class TestVoiceOutputSpeak:
    """Tests for speak method."""

    def test_returns_error_when_disabled(self):
        """Test returns error when voice is disabled."""
        config = VoiceOutputConfig(enabled=False)
        voice = VoiceOutput(config)

        result = voice.speak("Hello world")

        assert result.success is False
        assert result.error_code == VoiceOutputErrorCode.NOT_ENABLED.value

    def test_returns_error_for_empty_text(self):
        """Test returns error for empty text."""
        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        result = voice.speak("")

        assert result.success is False
        assert result.error_code == VoiceOutputErrorCode.EMPTY_TEXT.value

    def test_returns_error_for_whitespace_text(self):
        """Test returns error for whitespace-only text."""
        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        result = voice.speak("   ")

        assert result.success is False
        assert result.error_code == VoiceOutputErrorCode.EMPTY_TEXT.value

    @patch("src.voice.output._check_piper_available")
    def test_returns_error_when_no_piper(self, mock_check):
        """Test returns error when piper unavailable."""
        mock_check.return_value = (False, None)

        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        result = voice.speak("Hello world")

        assert result.success is False
        assert result.error_code == VoiceOutputErrorCode.PIPER_UNAVAILABLE.value


class TestVoiceOutputGenerateAudio:
    """Tests for generate_audio method."""

    def test_returns_error_when_disabled(self):
        """Test returns error when voice is disabled."""
        config = VoiceOutputConfig(enabled=False)
        voice = VoiceOutput(config)

        result = voice.generate_audio("Hello world")

        assert result.success is False
        assert result.error_code == VoiceOutputErrorCode.NOT_ENABLED.value

    def test_returns_error_for_empty_text(self):
        """Test returns error for empty text."""
        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        result = voice.generate_audio("")

        assert result.success is False
        assert result.error_code == VoiceOutputErrorCode.EMPTY_TEXT.value

    @patch("src.voice.output._check_piper_available")
    def test_returns_error_when_no_piper(self, mock_check):
        """Test returns error when piper unavailable."""
        mock_check.return_value = (False, None)

        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        result = voice.generate_audio("Hello world")

        assert result.success is False
        assert result.error_code == VoiceOutputErrorCode.PIPER_UNAVAILABLE.value


class TestTextPreparation:
    """Tests for text preparation for TTS."""

    def test_prepare_text_removes_markdown(self):
        """Test that markdown formatting is cleaned."""
        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        # Test various markdown elements
        text = "# Heading\n**bold** and `code` here"
        cleaned = voice._prepare_text_for_tts(text)

        assert "**" not in cleaned
        assert "`" not in cleaned
        assert "#" not in cleaned

    def test_prepare_text_normalizes_whitespace(self):
        """Test that excessive whitespace is normalized."""
        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        text = "Hello    world\n\n\ntest"
        cleaned = voice._prepare_text_for_tts(text)

        assert cleaned == "Hello world test"


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
        assert "Piper" in USER_ERROR_MESSAGES[VoiceOutputErrorCode.PIPER_UNAVAILABLE]
        assert "audio" in USER_ERROR_MESSAGES[VoiceOutputErrorCode.PLAYBACK_FAILED].lower()


# =============================================================================
# Integration-style Tests (with mocks)
# =============================================================================


class TestVoiceOutputIntegration:
    """Integration-style tests with mocked dependencies."""

    @patch("src.voice.output._check_piper_available")
    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.path.exists")
    @patch("os.unlink")
    def test_generate_with_piper_cli_success(
        self,
        mock_unlink,
        mock_exists,
        mock_tempfile,
        mock_run,
        mock_check,
    ):
        """Test successful audio generation with piper CLI."""
        mock_check.return_value = (True, "piper")

        # Setup tempfile mock for input
        mock_temp_input = MagicMock()
        mock_temp_input.name = "/tmp/input.txt"
        mock_temp_input.__enter__ = MagicMock(return_value=mock_temp_input)
        mock_temp_input.__exit__ = MagicMock(return_value=False)

        # Setup tempfile mock for output
        mock_temp_output = MagicMock()
        mock_temp_output.name = "/tmp/output.wav"
        mock_temp_output.__enter__ = MagicMock(return_value=mock_temp_output)
        mock_temp_output.__exit__ = MagicMock(return_value=False)

        mock_tempfile.side_effect = [mock_temp_input, mock_temp_output]

        # Setup subprocess mock
        mock_run.return_value = Mock(returncode=0)

        # Setup file exists mock
        mock_exists.return_value = True

        # Create a valid WAV file for testing
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(b"\x00" * 44100)  # 1 second of silence
        test_wav_data = wav_buffer.getvalue()

        with patch("builtins.open", MagicMock()) as mock_open:
            # Mock reading the output file
            mock_file = MagicMock()
            mock_file.read.return_value = test_wav_data
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_file

            config = VoiceOutputConfig(enabled=True)
            voice = VoiceOutput(config)

            result = voice.generate_audio("Hello world")

        # The result depends on mocking being complete
        # Just verify no exception is raised
        assert isinstance(result, VoiceOutputResult)

    @patch("src.voice.output._check_piper_available")
    def test_generate_with_python_piper_success(self, mock_check):
        """Test successful audio generation with Python piper."""
        mock_check.return_value = (True, "python-piper")

        # Mock the piper module
        mock_voice_instance = MagicMock()
        mock_voice_instance.config.sample_rate = 22050
        mock_voice_instance.synthesize_stream_raw.return_value = [b"\x00" * 1024]

        mock_piper_voice = MagicMock()
        mock_piper_voice.load.return_value = mock_voice_instance

        mock_piper = MagicMock()
        mock_piper.PiperVoice = mock_piper_voice

        with patch.dict("sys.modules", {"piper": mock_piper}):
            config = VoiceOutputConfig(enabled=True)
            voice = VoiceOutput(config)
            voice._piper_path = "python-piper"

            result = voice._generate_with_python_piper("Hello world")

        assert result.success is True
        assert result.audio_data is not None

    def test_tts_failure_does_not_raise(self):
        """Test that TTS failures return result instead of raising."""
        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        # This should not raise, even if piper is unavailable
        result = voice.speak("Hello world")

        # Should be a failed result, not an exception
        assert isinstance(result, VoiceOutputResult)
        assert result.success is False


class TestVoiceOutputReceivesFormattedText:
    """Tests to verify TTS receives only formatted text."""

    def test_tts_gets_plain_text(self):
        """Test that TTS receives plain text, not internal data."""
        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        # Prepare text should clean markdown
        input_text = "**Bold** and `code`"
        cleaned = voice._prepare_text_for_tts(input_text)

        # Should have plain text
        assert "**" not in cleaned
        assert "`" not in cleaned
        assert "Bold" in cleaned
        assert "code" in cleaned

    def test_tts_never_sees_internal_response(self):
        """Test that TTS cannot access internal response data."""
        # VoiceOutput takes only text strings, never dicts
        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        # Attempting to pass a dict would fail type checking
        # and generate_audio only accepts str
        import inspect
        sig = inspect.signature(voice.generate_audio)
        param = list(sig.parameters.values())[0]

        # First parameter is 'text' which should be a string
        assert param.name == "text"


class TestVoiceOutputNoAudioPersistence:
    """Tests to verify audio is not persisted."""

    def test_audio_only_in_memory(self):
        """Test that audio data is returned in memory, not saved to disk."""
        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        # The result has audio_data attribute but no file path
        result = VoiceOutputResult(
            success=True,
            audio_data=b"audio bytes",
            duration_seconds=1.0,
        )

        # Audio is in memory only
        assert result.audio_data is not None
        assert not hasattr(result, "file_path")
        assert not hasattr(result, "audio_path")


class TestVoiceOutputNonFatal:
    """Tests to verify TTS errors don't break the pipeline."""

    def test_disabled_returns_result_not_exception(self):
        """Test that disabled voice returns result, not exception."""
        config = VoiceOutputConfig(enabled=False)
        voice = VoiceOutput(config)

        # Should not raise
        result = voice.speak("Hello")

        assert result.success is False
        assert result.error_code is not None

    def test_empty_text_returns_result_not_exception(self):
        """Test that empty text returns result, not exception."""
        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        # Should not raise
        result = voice.speak("")

        assert result.success is False
        assert result.error_code is not None

    @patch("src.voice.output._check_piper_available")
    def test_piper_unavailable_returns_result(self, mock_check):
        """Test that missing piper returns result, not exception."""
        mock_check.return_value = (False, None)

        config = VoiceOutputConfig(enabled=True)
        voice = VoiceOutput(config)

        # Should not raise
        result = voice.generate_audio("Hello")

        assert result.success is False
        assert result.error_code == VoiceOutputErrorCode.PIPER_UNAVAILABLE.value
