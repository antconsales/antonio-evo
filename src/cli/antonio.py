#!/usr/bin/env python3
"""
Antonio CLI - Command-line interface for the offline assistant.

A simple, reliable CLI that routes input through the full orchestrator
pipeline and formats output for human consumption.

Usage:
    antonio "your question here"
    antonio --json "your question"
    antonio --raw "your question"
    antonio --voice              # Voice input via push-to-talk
    antonio --speak "question"   # Speak the response aloud
    antonio --voice --speak      # Voice input and output

Exit codes:
    0 - Success
    1 - User error (invalid input, validation failure)
    2 - System error (service unavailable, timeout)

DESIGN PRINCIPLES:
- Deterministic: Same input always produces same output
- Silent errors: No stack traces, no internal details (unless --raw)
- User-friendly: Clear error messages via ResponseFormatter
- No logging: Output only to stdout/stderr
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, Optional, Tuple

# Exit codes
EXIT_SUCCESS = 0
EXIT_USER_ERROR = 1
EXIT_SYSTEM_ERROR = 2

# Version
VERSION = "0.1.0"


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for the CLI.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="antonio",
        description="Antonio - Local AI Assistant",
        epilog="Example: antonio \"What is the capital of France?\"",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="The question or text to process",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Force JSON output format",
    )

    parser.add_argument(
        "--raw",
        action="store_true",
        help="Bypass formatting, show internal response (for debugging)",
    )

    parser.add_argument(
        "--voice",
        action="store_true",
        help="Use voice input instead of text (push-to-talk recording)",
    )

    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak the response aloud (text-to-speech)",
    )

    parser.add_argument(
        "--no-speak",
        action="store_true",
        dest="no_speak",
        help="Disable voice output (override config)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"antonio {VERSION}",
    )

    return parser


def initialize_orchestrator() -> Tuple[Optional[Any], Optional[str]]:
    """
    Initialize the orchestrator components.

    Returns:
        Tuple of (orchestrator, error_message)
        If successful, error_message is None.
        If failed, orchestrator is None and error_message describes the issue.

    Note:
        This function never raises exceptions.
    """
    try:
        # Import here to catch import errors gracefully
        from ..input.normalizer import Normalizer
        from ..policy.classifier import Classifier
        from ..policy.policy_engine import PolicyEngine
        from ..router.router import Router
        from ..output.response_builder import ResponseBuilder
        from ..output.response_formatter import ResponseFormatter

        # Create a lightweight orchestrator-like container
        class CLIOrchestrator:
            """Lightweight orchestrator for CLI use."""

            def __init__(self):
                self.normalizer = Normalizer()
                self.classifier = Classifier()
                self.policy = PolicyEngine("config/policy.json")
                self.router = Router("config/handlers.json")
                self.response_builder = ResponseBuilder()
                self.formatter = ResponseFormatter()

            def process(self, text: str) -> Dict[str, Any]:
                """
                Process a text input through the pipeline.

                Args:
                    text: User input text

                Returns:
                    Internal response dictionary
                """
                start_time = time.time()

                # Step 1: Normalize
                norm_result = self.normalizer.normalize(text)

                if not norm_result.success:
                    # Return validation error response
                    return {
                        "success": False,
                        "error": "Invalid input",
                        "error_code": norm_result.error_code or "VALIDATION_ERROR",
                        "_meta": {
                            "elapsed_ms": int((time.time() - start_time) * 1000),
                        },
                    }

                request = norm_result.request

                # Step 2: Classify
                classification = self.classifier.classify(request)

                # Step 3: Policy decision
                decision = self.policy.decide(request, classification)

                # Step 4: Route to handler
                result = self.router.route(request, decision)

                # Step 5: Build response
                elapsed_ms = int((time.time() - start_time) * 1000)
                response = self.response_builder.build(
                    result=result,
                    decision=decision,
                    classification=classification,
                    elapsed_ms=elapsed_ms,
                )

                return response

        return CLIOrchestrator(), None

    except ImportError as e:
        return None, f"Failed to load required modules: {e}"
    except Exception as e:
        return None, f"Failed to initialize: {e}"


def determine_exit_code(response: Dict[str, Any]) -> int:
    """
    Determine exit code from response.

    Args:
        response: Response dictionary

    Returns:
        Exit code (0 for success, 1 for user error, 2 for system error)
    """
    if response.get("success", False):
        return EXIT_SUCCESS

    error_code = response.get("error_code", "")

    # User errors (exit code 1)
    user_error_codes = {
        "VALIDATION_ERROR",
        "MISSING_TEXT",
        "LLM_MISSING_TEXT",
        "REJECTED",
    }

    if error_code in user_error_codes:
        return EXIT_USER_ERROR

    # Validation-like errors
    if "VALIDATION" in error_code.upper() or "INVALID" in error_code.upper():
        return EXIT_USER_ERROR

    # System errors (exit code 2)
    return EXIT_SYSTEM_ERROR


def get_voice_input(raw_output: bool) -> Tuple[Optional[str], Optional[int]]:
    """
    Get input from voice using push-to-talk.

    Args:
        raw_output: If True, show detailed error info

    Returns:
        Tuple of (transcribed_text, exit_code)
        If successful, exit_code is None.
        If failed, transcribed_text is None and exit_code is set.

    Note:
        This function never raises exceptions.
    """
    try:
        from ..voice import VoiceInput, VoiceInputResult, load_voice_config
    except ImportError as e:
        if raw_output:
            print(json.dumps({
                "success": False,
                "error": f"Voice module not available: {e}",
                "error_code": "VOICE_IMPORT_ERROR",
            }, indent=2), file=sys.stderr)
        else:
            print("Error: Voice features are not available.", file=sys.stderr)
        return None, EXIT_SYSTEM_ERROR

    # Load voice configuration
    config = load_voice_config()

    if not config.input.enabled:
        if raw_output:
            print(json.dumps({
                "success": False,
                "error": "Voice input is disabled in configuration",
                "error_code": "VOICE_DISABLED",
            }, indent=2), file=sys.stderr)
        else:
            print("Error: Voice input is disabled.", file=sys.stderr)
            print("Enable it in config/voice.json", file=sys.stderr)
        return None, EXIT_USER_ERROR

    # Create voice input handler
    try:
        voice = VoiceInput(config.input)
    except Exception as e:
        if raw_output:
            print(json.dumps({
                "success": False,
                "error": str(e),
                "error_code": "VOICE_INIT_ERROR",
            }, indent=2), file=sys.stderr)
        else:
            print("Error: Failed to initialize voice input.", file=sys.stderr)
        return None, EXIT_SYSTEM_ERROR

    # Check dependencies
    if not voice.is_available():
        if raw_output:
            print(json.dumps({
                "success": False,
                "error": "Voice dependencies not available",
                "error_code": "VOICE_DEPS_MISSING",
            }, indent=2), file=sys.stderr)
        else:
            print("Error: Voice input requires Whisper to be installed.", file=sys.stderr)
            print("Install with: pip install openai-whisper", file=sys.stderr)
        return None, EXIT_SYSTEM_ERROR

    # Prompt user and start recording
    print("Press Enter to start recording (Ctrl+C to cancel)...", file=sys.stderr)
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return None, EXIT_USER_ERROR

    print("Recording... Press Enter to stop.", file=sys.stderr)

    # Start recording in background and wait for Enter
    import threading

    recording_done = threading.Event()
    result_holder: list = []

    def record_thread():
        """Thread that runs record_and_transcribe."""
        try:
            result = voice.record_and_transcribe(
                on_recording_start=lambda: None,
                on_recording_stop=lambda: None,
            )
            result_holder.append(result)
        except Exception as e:
            result_holder.append(VoiceInputResult(
                success=False,
                error_code="VOICE_ERROR",
                error_message=str(e),
            ))
        finally:
            recording_done.set()

    # Start recording thread
    thread = threading.Thread(target=record_thread, daemon=True)
    thread.start()

    # Wait for Enter key to stop recording
    try:
        input()
        voice.stop_recording()
    except KeyboardInterrupt:
        voice.stop_recording()
        print("\nCancelled.", file=sys.stderr)
        return None, EXIT_USER_ERROR

    # Wait for recording thread to finish (transcription happens inside)
    print("Transcribing...", file=sys.stderr)
    recording_done.wait(timeout=60.0)  # Allow time for transcription

    if not result_holder:
        if raw_output:
            print(json.dumps({
                "success": False,
                "error": "Recording timed out",
                "error_code": "VOICE_TIMEOUT",
            }, indent=2), file=sys.stderr)
        else:
            print("Error: Recording timed out.", file=sys.stderr)
        return None, EXIT_SYSTEM_ERROR

    result = result_holder[0]

    if not result.success:
        if raw_output:
            print(json.dumps({
                "success": False,
                "error": result.error_message,
                "error_code": result.error_code or "VOICE_ERROR",
            }, indent=2), file=sys.stderr)
        else:
            print(f"Error: {result.error_message or 'Voice input failed'}", file=sys.stderr)

        # Determine exit code based on error type
        if result.error_code in ("VOICE_NO_SPEECH", "VOICE_AUDIO_TOO_SHORT"):
            return None, EXIT_USER_ERROR
        return None, EXIT_SYSTEM_ERROR

    if not result.text or not result.text.strip():
        if raw_output:
            print(json.dumps({
                "success": False,
                "error": "No speech detected in audio",
                "error_code": "VOICE_NO_SPEECH",
            }, indent=2), file=sys.stderr)
        else:
            print("Error: No speech was detected.", file=sys.stderr)
        return None, EXIT_USER_ERROR

    # Display transcribed text
    transcribed = result.text.strip()
    print(f"\nYou said: {transcribed}\n", file=sys.stderr)

    return transcribed, None


def speak_text(text: str, raw_output: bool) -> None:
    """
    Speak text aloud using text-to-speech.

    This is non-fatal: TTS failures are logged as warnings.
    Text output is always printed first.

    Args:
        text: The formatted text to speak
        raw_output: If True, show detailed error info
    """
    try:
        from ..voice import VoiceOutput, load_voice_config
    except ImportError as e:
        if raw_output:
            print(f"Warning: Voice output not available: {e}", file=sys.stderr)
        else:
            print("Warning: Voice output is not available.", file=sys.stderr)
        return

    # Load voice configuration
    config = load_voice_config()

    if not config.output.enabled:
        if raw_output:
            print("Warning: Voice output is disabled in config/voice.json", file=sys.stderr)
        else:
            print("Warning: Voice output is disabled.", file=sys.stderr)
        return

    # Create voice output handler
    try:
        voice = VoiceOutput(config.output)
    except Exception as e:
        if raw_output:
            print(f"Warning: Failed to initialize voice output: {e}", file=sys.stderr)
        else:
            print("Warning: Could not initialize voice output.", file=sys.stderr)
        return

    # Check if TTS is available
    if not voice.is_available():
        if raw_output:
            print("Warning: Piper TTS is not installed.", file=sys.stderr)
        else:
            print("Warning: Voice output requires Piper to be installed.", file=sys.stderr)
        return

    # Speak the text
    print("Speaking...", file=sys.stderr)
    result = voice.speak(text)

    if not result.success:
        if raw_output:
            print(f"Warning: TTS failed: {result.error_message}", file=sys.stderr)
        else:
            print(f"Warning: {result.error_message or 'Could not speak response.'}", file=sys.stderr)


def format_output(
    response: Dict[str, Any],
    json_output: bool,
    raw_output: bool,
    formatter: Any,
) -> str:
    """
    Format response for output.

    Args:
        response: Internal response dictionary
        json_output: If True, output as JSON
        raw_output: If True, output raw internal response
        formatter: ResponseFormatter instance

    Returns:
        Formatted output string
    """
    if raw_output:
        # Raw mode: show internal response as pretty JSON
        return json.dumps(response, indent=2, ensure_ascii=False)

    if json_output:
        # JSON mode: use formatter but output as JSON
        formatted = formatter.format(response)
        return formatted.to_json()

    # Text mode: use formatter text output
    formatted = formatter.format(response)
    return formatted.to_text()


def run_cli(args: Optional[list] = None) -> int:
    """
    Run the CLI with given arguments.

    Args:
        args: Command-line arguments (uses sys.argv if None)

    Returns:
        Exit code
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    # Handle voice input mode
    if parsed.voice:
        question, exit_code = get_voice_input(parsed.raw)
        if exit_code is not None:
            return exit_code
    else:
        # Check for missing input
        if not parsed.question:
            print("Error: No question provided.", file=sys.stderr)
            print("Usage: antonio \"your question here\"", file=sys.stderr)
            return EXIT_USER_ERROR

        # Check for empty input
        question = parsed.question.strip()
        if not question:
            print("Error: Question cannot be empty.", file=sys.stderr)
            return EXIT_USER_ERROR

    # Initialize orchestrator
    orchestrator, init_error = initialize_orchestrator()

    if orchestrator is None:
        if parsed.raw:
            # Raw mode: show the error
            print(json.dumps({
                "success": False,
                "error": init_error,
                "error_code": "INIT_ERROR",
            }, indent=2), file=sys.stderr)
        else:
            print("Error: The assistant is currently unavailable.", file=sys.stderr)
        return EXIT_SYSTEM_ERROR

    # Process the question
    try:
        response = orchestrator.process(question)
    except Exception as e:
        # Should not happen, but handle gracefully
        if parsed.raw:
            print(json.dumps({
                "success": False,
                "error": str(e),
                "error_code": "PROCESS_ERROR",
            }, indent=2), file=sys.stderr)
        else:
            print("Error: Something went wrong. Please try again.", file=sys.stderr)
        return EXIT_SYSTEM_ERROR

    # Format and output
    output = format_output(
        response=response,
        json_output=parsed.json_output,
        raw_output=parsed.raw,
        formatter=orchestrator.formatter,
    )

    # Determine output stream (stdout for success, stderr for errors)
    if response.get("success", False):
        print(output)
    else:
        # For text mode errors, print to stderr
        if not parsed.json_output and not parsed.raw:
            print(output, file=sys.stderr)
        else:
            # JSON and raw modes always print to stdout for easier parsing
            print(output)

    # Voice output (optional, non-fatal)
    if parsed.speak and not parsed.no_speak:
        # Get the text to speak from formatted output
        # For JSON mode, extract the message field
        if parsed.json_output or parsed.raw:
            # For JSON/raw mode, try to extract the message
            try:
                data = json.loads(output)
                speak_content = data.get("message", data.get("data", ""))
            except (json.JSONDecodeError, KeyError):
                speak_content = output
        else:
            speak_content = output

        if speak_content and speak_content.strip():
            speak_text(speak_content, parsed.raw)

    return determine_exit_code(response)


def main() -> None:
    """
    Main entry point for the CLI.

    This function is called when running `antonio` command.
    """
    try:
        exit_code = run_cli()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception:
        # Last resort error handling - should never happen
        print("Error: An unexpected error occurred.", file=sys.stderr)
        sys.exit(EXIT_SYSTEM_ERROR)


if __name__ == "__main__":
    main()
