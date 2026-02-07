#!/usr/bin/env python3
"""
Antonio HTTP API Server - Local HTTP interface for the assistant.

A minimal HTTP server using Python's stdlib that exposes the assistant
via a localhost-only API for programmatic access.

Endpoints:
    POST /ask    - Process a question
    POST /listen - Transcribe audio to text (optional processing)
    GET  /health - Check system health

Security:
    - Binds ONLY to localhost (127.0.0.1) by default
    - No authentication (trusted local environment)
    - No internal errors exposed (unless raw=true)

DESIGN PRINCIPLES:
- Minimal dependencies (stdlib only)
- Deterministic behavior
- Same orchestrator as CLI
- Localhost-only binding
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

# Default configuration
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8420

# HTTP Status codes
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_METHOD_NOT_ALLOWED = 405
HTTP_INTERNAL_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503


class APIError(Exception):
    """API error with HTTP status code."""

    def __init__(self, message: str, status_code: int = HTTP_BAD_REQUEST):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def create_orchestrator() -> Tuple[Optional[Any], Optional[str]]:
    """
    Create the orchestrator for processing requests.

    Returns:
        Tuple of (orchestrator, error_message)
    """
    try:
        from ..input.normalizer import Normalizer
        from ..policy.classifier import Classifier
        from ..policy.policy_engine import PolicyEngine
        from ..router.router import Router
        from ..output.response_builder import ResponseBuilder
        from ..output.response_formatter import ResponseFormatter, OutputMode

        class APIOrchestrator:
            """Orchestrator for API use."""

            def __init__(self):
                self.normalizer = Normalizer()
                self.classifier = Classifier()
                self.policy = PolicyEngine("config/policy.json")
                self.router = Router("config/handlers.json")
                self.response_builder = ResponseBuilder()
                self.formatter = ResponseFormatter()
                self._llm_available = None
                self._llm_check_time = 0

            def process(self, text: str) -> Dict[str, Any]:
                """Process a text input through the pipeline."""
                start_time = time.time()

                # Step 1: Normalize
                norm_result = self.normalizer.normalize(text)

                if not norm_result.success:
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

            def check_llm_available(self, force: bool = False) -> bool:
                """
                Check if LLM is available.

                Caches result for 30 seconds to avoid repeated checks.
                """
                now = time.time()
                if not force and self._llm_available is not None:
                    if now - self._llm_check_time < 30:
                        return self._llm_available

                try:
                    from ..handlers.llm_local import LLMLocalHandler
                    handler = LLMLocalHandler({})
                    self._llm_available = handler.is_available()
                except Exception:
                    self._llm_available = False

                self._llm_check_time = now
                return self._llm_available

        return APIOrchestrator(), None

    except ImportError as e:
        return None, f"Failed to import required modules: {e}"
    except Exception as e:
        return None, f"Failed to initialize: {e}"


class AntonioRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for Antonio API.

    Handles:
        POST /ask    - Process questions
        GET  /health - Health check
    """

    # Disable default logging
    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass

    def _send_json_response(
        self,
        data: Dict[str, Any],
        status_code: int = HTTP_OK,
    ) -> None:
        """Send a JSON response."""
        response_body = json.dumps(data, ensure_ascii=False, indent=2)
        response_bytes = response_body.encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response_bytes)

    def _send_text_response(
        self,
        text: str,
        status_code: int = HTTP_OK,
    ) -> None:
        """Send a plain text response."""
        response_bytes = text.encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_bytes)

    def _send_error_response(
        self,
        message: str,
        status_code: int = HTTP_BAD_REQUEST,
    ) -> None:
        """Send an error response."""
        self._send_json_response(
            {"success": False, "error": message},
            status_code=status_code,
        )

    def _read_request_body(self) -> bytes:
        """Read the request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            return self.rfile.read(content_length)
        return b""

    def _parse_json_body(self) -> Dict[str, Any]:
        """Parse JSON request body."""
        body = self._read_request_body()

        if not body:
            raise APIError("Request body is empty", HTTP_BAD_REQUEST)

        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON: {e}", HTTP_BAD_REQUEST)

        if not isinstance(data, dict):
            raise APIError("Request body must be a JSON object", HTTP_BAD_REQUEST)

        return data

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/health":
            self._handle_health()
        else:
            self._send_error_response("Not found", HTTP_NOT_FOUND)

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/ask":
            self._handle_ask()
        elif path == "/listen":
            self._handle_listen()
        else:
            self._send_error_response("Not found", HTTP_NOT_FOUND)

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(HTTP_OK)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _handle_health(self) -> None:
        """Handle GET /health endpoint."""
        try:
            orchestrator = self.server.orchestrator
            llm_available = False

            if orchestrator is not None:
                try:
                    llm_available = orchestrator.check_llm_available()
                except Exception:
                    llm_available = False

            self._send_json_response({
                "status": "ok",
                "llm_available": llm_available,
                "timestamp": time.time(),
            })

        except Exception:
            self._send_json_response({
                "status": "ok",
                "llm_available": False,
                "timestamp": time.time(),
            })

    def _handle_ask(self) -> None:
        """
        Handle POST /ask endpoint.

        Request body:
            {
                "question": "Your question here",
                "output_mode": "text",  # or "json"
                "raw": false,
                "speak": false,         # optional: generate TTS audio
                "return_audio": false   # optional: return base64 audio in response
            }

        When speak=true and return_audio=false:
            Audio is played locally on the server.

        When speak=true and return_audio=true:
            Response includes "audio" field with base64-encoded WAV.
        """
        import base64

        try:
            # Parse request body
            try:
                body = self._parse_json_body()
            except APIError as e:
                self._send_error_response(e.message, e.status_code)
                return

            # Validate required fields
            question = body.get("question")
            if not question:
                self._send_error_response("Missing required field: question")
                return

            if not isinstance(question, str):
                self._send_error_response("Field 'question' must be a string")
                return

            question = question.strip()
            if not question:
                self._send_error_response("Field 'question' cannot be empty")
                return

            # Parse options
            output_mode = body.get("output_mode", "text")
            if output_mode not in ("text", "json"):
                self._send_error_response("Field 'output_mode' must be 'text' or 'json'")
                return

            raw = body.get("raw", False)
            if not isinstance(raw, bool):
                self._send_error_response("Field 'raw' must be a boolean")
                return

            # Voice output options
            speak = body.get("speak", False)
            if not isinstance(speak, bool):
                self._send_error_response("Field 'speak' must be a boolean")
                return

            return_audio = body.get("return_audio", False)
            if not isinstance(return_audio, bool):
                self._send_error_response("Field 'return_audio' must be a boolean")
                return

            # Check orchestrator
            orchestrator = self.server.orchestrator
            if orchestrator is None:
                self._send_error_response(
                    "Service unavailable",
                    HTTP_SERVICE_UNAVAILABLE,
                )
                return

            # Process the question
            try:
                response = orchestrator.process(question)
            except Exception:
                self._send_error_response(
                    "An error occurred while processing your request",
                    HTTP_INTERNAL_ERROR,
                )
                return

            # Format output
            formatted_text = ""
            if raw:
                # Raw mode: return internal response
                result_data = response
                # Get text for TTS from response message
                formatted_text = response.get("message", "")
            elif output_mode == "json":
                # JSON mode: use formatter
                formatted = orchestrator.formatter.format(response)
                result_data = formatted.to_dict()
                formatted_text = result_data.get("message", "")
            else:
                # Text mode: use formatter
                formatted = orchestrator.formatter.format(response)
                formatted_text = formatted.to_text()
                result_data = None  # Will send text response

            # Handle voice output (non-fatal)
            audio_data = None
            voice_warning = None

            if speak and formatted_text:
                audio_data, voice_warning = self._generate_speech(formatted_text, return_audio)

            # Send response
            if result_data is not None:
                # JSON or raw mode
                if return_audio and audio_data:
                    result_data["audio"] = base64.b64encode(audio_data).decode("ascii")
                    result_data["audio_format"] = "wav"
                if voice_warning:
                    result_data["voice_warning"] = voice_warning
                self._send_json_response(result_data)
            else:
                # Text mode - convert to JSON if audio is included
                if return_audio and audio_data:
                    self._send_json_response({
                        "success": response.get("success", False),
                        "message": formatted_text,
                        "audio": base64.b64encode(audio_data).decode("ascii"),
                        "audio_format": "wav",
                        "voice_warning": voice_warning,
                    })
                else:
                    status = HTTP_OK if response.get("success", False) else HTTP_BAD_REQUEST
                    self._send_text_response(formatted_text, status)

        except Exception:
            # Catch-all for unexpected errors
            self._send_error_response(
                "An unexpected error occurred",
                HTTP_INTERNAL_ERROR,
            )

    def _generate_speech(
        self,
        text: str,
        return_audio: bool,
    ) -> tuple:
        """
        Generate speech from text (non-fatal).

        Args:
            text: The formatted text to speak
            return_audio: If True, return audio data; if False, play locally

        Returns:
            Tuple of (audio_data, warning_message)
            audio_data is bytes if return_audio=True and success, else None
            warning_message is set on non-fatal errors
        """
        try:
            from ..voice import VoiceOutput, load_voice_config
        except ImportError:
            return None, "Voice output module not available"

        config = load_voice_config()

        if not config.output.enabled:
            return None, "Voice output is disabled in configuration"

        try:
            voice = VoiceOutput(config.output)
        except Exception:
            return None, "Failed to initialize voice output"

        if not voice.is_available():
            return None, "Piper TTS is not installed"

        if return_audio:
            # Generate audio and return it
            result = voice.generate_audio(text)
            if result.success:
                return result.audio_data, None
            else:
                return None, result.error_message or "TTS generation failed"
        else:
            # Play audio locally
            result = voice.speak(text)
            if result.success:
                return None, None
            else:
                return None, result.error_message or "TTS playback failed"

    def _handle_listen(self) -> None:
        """
        Handle POST /listen endpoint.

        Accepts audio data (base64 encoded) and transcribes it.
        Optionally processes the transcribed text through the pipeline.

        Request body:
            {
                "audio": "<base64-encoded-audio-data>",
                "format": "wav",  # audio format (wav, mp3, etc.)
                "process": false, # if true, also process through pipeline
                "output_mode": "text",  # text or json (when process=true)
                "raw": false  # if true, return raw internal response
            }

        Response (when process=false):
            {
                "success": true,
                "text": "transcribed text"
            }

        Response (when process=true):
            {
                "success": true,
                "text": "transcribed text",
                "response": "assistant response"
            }
        """
        import base64
        import tempfile
        import os

        try:
            # Parse request body
            try:
                body = self._parse_json_body()
            except APIError as e:
                self._send_error_response(e.message, e.status_code)
                return

            # Validate required fields
            audio_b64 = body.get("audio")
            if not audio_b64:
                self._send_error_response("Missing required field: audio")
                return

            if not isinstance(audio_b64, str):
                self._send_error_response("Field 'audio' must be a base64-encoded string")
                return

            # Decode audio data
            try:
                audio_data = base64.b64decode(audio_b64)
            except Exception:
                self._send_error_response("Invalid base64 encoding in 'audio' field")
                return

            if len(audio_data) == 0:
                self._send_error_response("Audio data is empty")
                return

            # Parse optional fields
            audio_format = body.get("format", "wav")
            if not isinstance(audio_format, str):
                self._send_error_response("Field 'format' must be a string")
                return

            process_text = body.get("process", False)
            if not isinstance(process_text, bool):
                self._send_error_response("Field 'process' must be a boolean")
                return

            output_mode = body.get("output_mode", "text")
            if output_mode not in ("text", "json"):
                self._send_error_response("Field 'output_mode' must be 'text' or 'json'")
                return

            raw = body.get("raw", False)
            if not isinstance(raw, bool):
                self._send_error_response("Field 'raw' must be a boolean")
                return

            # Check if voice module is available
            try:
                from ..voice import VoiceInput, VoiceInputError, load_voice_config
            except ImportError:
                self._send_error_response(
                    "Voice features are not available",
                    HTTP_SERVICE_UNAVAILABLE,
                )
                return

            # Load voice configuration
            config = load_voice_config()

            if not config.input.enabled:
                self._send_error_response(
                    "Voice input is disabled in configuration",
                    HTTP_SERVICE_UNAVAILABLE,
                )
                return

            # Create voice input handler
            try:
                voice = VoiceInput(config.input)
            except VoiceInputError as e:
                self._send_error_response(
                    f"Voice initialization failed: {e.message}",
                    HTTP_SERVICE_UNAVAILABLE,
                )
                return

            # Check if Whisper is available
            if not voice.is_available():
                self._send_error_response(
                    "Voice transcription requires Whisper to be installed",
                    HTTP_SERVICE_UNAVAILABLE,
                )
                return

            # Transcribe the audio data
            try:
                result = voice.transcribe_audio(audio_data)
            except VoiceInputError as e:
                if raw:
                    self._send_json_response({
                        "success": False,
                        "error": e.message,
                        "error_code": e.code.value,
                    })
                else:
                    self._send_error_response(
                        f"Transcription failed: {e.message}",
                        HTTP_INTERNAL_ERROR,
                    )
                return
            except Exception as e:
                self._send_error_response(
                    "An error occurred during transcription",
                    HTTP_INTERNAL_ERROR,
                )
                return

            if not result.success:
                if raw:
                    self._send_json_response({
                        "success": False,
                        "error": result.error_message,
                        "error_code": result.error_code.value if result.error_code else "TRANSCRIPTION_ERROR",
                    })
                else:
                    self._send_error_response(
                        result.error_message or "Transcription failed",
                        HTTP_INTERNAL_ERROR,
                    )
                return

            transcribed_text = result.text.strip() if result.text else ""

            if not transcribed_text:
                self._send_json_response({
                    "success": False,
                    "error": "No speech detected in audio",
                })
                return

            # If process=true, run through the pipeline
            if process_text:
                orchestrator = self.server.orchestrator
                if orchestrator is None:
                    self._send_error_response(
                        "Service unavailable",
                        HTTP_SERVICE_UNAVAILABLE,
                    )
                    return

                try:
                    response = orchestrator.process(transcribed_text)
                except Exception:
                    self._send_error_response(
                        "An error occurred while processing your request",
                        HTTP_INTERNAL_ERROR,
                    )
                    return

                # Format output
                if raw:
                    # Raw mode: return internal response with text
                    response["input_text"] = transcribed_text
                    self._send_json_response(response)
                elif output_mode == "json":
                    formatted = orchestrator.formatter.format(response)
                    result_dict = formatted.to_dict()
                    result_dict["input_text"] = transcribed_text
                    self._send_json_response(result_dict)
                else:
                    # Text mode with both input and output
                    formatted = orchestrator.formatter.format(response)
                    self._send_json_response({
                        "success": True,
                        "text": transcribed_text,
                        "response": formatted.to_text(),
                    })
            else:
                # Just return the transcribed text
                self._send_json_response({
                    "success": True,
                    "text": transcribed_text,
                })

        except Exception:
            # Catch-all for unexpected errors
            self._send_error_response(
                "An unexpected error occurred",
                HTTP_INTERNAL_ERROR,
            )


class AntonioAPIServer(HTTPServer):
    """
    Antonio API HTTP Server.

    Custom HTTPServer that holds a reference to the orchestrator.
    """

    def __init__(
        self,
        server_address: Tuple[str, int],
        handler_class: type,
        orchestrator: Optional[Any] = None,
    ):
        """
        Initialize the server.

        Args:
            server_address: (host, port) tuple
            handler_class: Request handler class
            orchestrator: APIOrchestrator instance
        """
        super().__init__(server_address, handler_class)
        self.orchestrator = orchestrator


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> Tuple[Optional[AntonioAPIServer], Optional[str]]:
    """
    Create and configure the API server.

    Args:
        host: Host to bind to (default: 127.0.0.1)
        port: Port to bind to (default: 8420)

    Returns:
        Tuple of (server, error_message)

    Note:
        For security, only localhost binding is allowed.
    """
    # Security check: only allow localhost
    allowed_hosts = ("127.0.0.1", "localhost", "::1")
    if host not in allowed_hosts:
        return None, f"Security: Only localhost binding allowed. Got: {host}"

    # Create orchestrator
    orchestrator, init_error = create_orchestrator()
    if orchestrator is None:
        return None, init_error

    # Create server
    try:
        server = AntonioAPIServer(
            (host, port),
            AntonioRequestHandler,
            orchestrator=orchestrator,
        )
        return server, None

    except OSError as e:
        return None, f"Failed to bind to {host}:{port}: {e}"
    except Exception as e:
        return None, f"Failed to create server: {e}"


def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    quiet: bool = False,
) -> int:
    """
    Run the API server.

    Args:
        host: Host to bind to
        port: Port to bind to
        quiet: If True, suppress startup messages

    Returns:
        Exit code (0 for clean shutdown, 1 for error)
    """
    server, error = create_server(host, port)

    if server is None:
        if not quiet:
            print(f"Error: {error}")
        return 1

    if not quiet:
        print(f"Antonio API Server running at http://{host}:{port}")
        print("Endpoints:")
        print(f"  POST http://{host}:{port}/ask")
        print(f"  POST http://{host}:{port}/listen")
        print(f"  GET  http://{host}:{port}/health")
        print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if not quiet:
            print("\nShutting down...")
    finally:
        server.server_close()

    return 0


def main() -> None:
    """Main entry point for running the server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Antonio HTTP API Server",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host to bind to (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind to (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress startup messages",
    )

    args = parser.parse_args()

    exit_code = run_server(
        host=args.host,
        port=args.port,
        quiet=args.quiet,
    )

    import sys
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
