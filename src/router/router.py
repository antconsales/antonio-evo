"""
Router - Simple deterministic dispatch with sandbox isolation.

NO MAGIC. Just a switch statement.
This is intentionally boring.

All handler execution is wrapped in ProcessSandbox to enforce
resource limits and prevent runaway processes.

Supports both subprocess and HTTP-based handlers for ASR/TTS,
configurable via .env or handlers.json.
"""

import json
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

from ..models.request import Request
from ..models.response import Response, ResponseMeta
from ..models.policy import PolicyDecision, Handler

from ..handlers.base import BaseHandler
from ..handlers.mistral import MistralHandler
from ..handlers.whisper import WhisperHandler
from ..handlers.tts import TTSHandler
from ..handlers.clip import CLIPHandler
from ..handlers.external import ExternalHandler
from ..handlers.rejection import RejectionHandler

# HTTP handlers (optional)
from ..handlers.whisper_http import WhisperHTTPHandler
from ..handlers.tts_http import TTSHTTPHandler

# Image generation handler
from ..handlers.zimage import ZImageHandler

from ..sandbox import ProcessSandbox, SandboxConfig, SandboxResult, SandboxViolation

if TYPE_CHECKING:
    from ..config.env_loader import ServiceConfig

logger = logging.getLogger(__name__)


# Default sandbox configuration applied when not specified in config
DEFAULT_SANDBOX_CONFIG = {
    "cpu_seconds": 120,
    "memory_mb": 1024,
    "timeout_seconds": 180,
}

# Handler-specific defaults (override the global defaults)
HANDLER_SANDBOX_DEFAULTS = {
    "mistral": {"cpu_seconds": 300, "memory_mb": 2048, "timeout_seconds": 300},
    "whisper": {"cpu_seconds": 120, "memory_mb": 1024, "timeout_seconds": 120},
    "tts": {"cpu_seconds": 60, "memory_mb": 512, "timeout_seconds": 60},
    "clip": {"cpu_seconds": 60, "memory_mb": 1024, "timeout_seconds": 60},
    "zimage": {"cpu_seconds": 600, "memory_mb": 16384, "timeout_seconds": 600},  # 10 min for CPU
    "external": {"cpu_seconds": 30, "memory_mb": 256, "timeout_seconds": 120},
    "reject": {"cpu_seconds": 5, "memory_mb": 64, "timeout_seconds": 10},
}

# Map Handler enum to config key names
HANDLER_CONFIG_KEYS = {
    Handler.TEXT_LOCAL: "mistral",
    Handler.TEXT_SOCIAL: "mistral",  # Same sandbox limits as mistral
    Handler.TEXT_LOGIC: "mistral",   # Same sandbox limits as mistral
    Handler.AUDIO_IN: "whisper",
    Handler.AUDIO_OUT: "tts",
    Handler.IMAGE_CAPTION: "clip",
    Handler.IMAGE_GEN: "zimage",     # Z-Image Turbo for image generation
    Handler.EXTERNAL_LLM: "external",
    Handler.REJECT: "reject",
}

# Error codes for sandbox violations
SANDBOX_ERROR_CODES = {
    SandboxViolation.TIMEOUT: "SANDBOX_TIMEOUT",
    SandboxViolation.CPU_EXCEEDED: "SANDBOX_CPU_EXCEEDED",
    SandboxViolation.MEMORY_EXCEEDED: "SANDBOX_MEMORY_EXCEEDED",
    SandboxViolation.EXCEPTION: "SANDBOX_EXCEPTION",
}


def _execute_handler(handler: BaseHandler, request: Request) -> Response:
    """
    Execute handler process method.

    This function exists to be passed to ProcessSandbox.execute().
    It must be a module-level function (not a method) to be picklable.
    """
    return handler.process(request)


class Router:
    """
    Deterministic router with sandbox isolation.

    This is just a switch statement that dispatches to handlers.
    The handler CANNOT call other handlers.

    All handler execution is wrapped in ProcessSandbox to enforce:
    - CPU time limits
    - Memory limits
    - Execution timeout

    Sandbox configuration is read from handlers.json under the "sandbox" key,
    with fallback to built-in defaults.

    Supports HTTP-based handlers for ASR/TTS when USE_HTTP_HANDLERS=true.
    """

    def __init__(
        self,
        config_path: str = "config/handlers.json",
        service_config: Optional["ServiceConfig"] = None
    ):
        self.config = self._load_config(config_path)
        self.sandbox_config = self._load_sandbox_config()
        self.service_config = service_config

        # Get handlers config (may be nested under "handlers" key)
        handlers_config = self.config.get("handlers", self.config)

        # Check if HTTP handlers should be used
        use_http = False
        if service_config:
            use_http = service_config.use_http_handlers
        else:
            use_http = handlers_config.get("whisper", {}).get("use_http", False)

        # Base mistral config (shared settings)
        mistral_base = handlers_config.get("mistral", {})

        # Override with service config if available
        if service_config:
            mistral_base["base_url"] = service_config.llm_server
            mistral_base["model"] = service_config.ollama_model
            mistral_base["timeout"] = service_config.llm_timeout

        # SOCIAL persona config
        social_config = {
            **mistral_base,
            "system_prompt_path": "prompts/mistral_social.txt",
            "temperature": 0.7,
            "max_tokens": 384,   # ~80s on CPU at ~4.7 tok/s
            "persona_name": "SOCIAL",
        }

        # LOGIC persona config
        logic_config = {
            **mistral_base,
            "system_prompt_path": "prompts/mistral_logic.txt",
            "temperature": 0.3,
            "max_tokens": 512,   # ~110s on CPU at ~4.7 tok/s
            "persona_name": "LOGIC",
        }

        # Whisper config (HTTP or subprocess)
        whisper_config = handlers_config.get("whisper", {})
        if service_config:
            whisper_config["server_url"] = service_config.asr_server
            whisper_config["timeout"] = service_config.asr_timeout
            whisper_config["language"] = service_config.whisper_language
            whisper_config["model"] = service_config.whisper_model

        # TTS config (HTTP or subprocess)
        tts_config = handlers_config.get("tts", {})
        if service_config:
            tts_config["server_url"] = service_config.tts_server
            tts_config["timeout"] = service_config.tts_timeout
            tts_config["voice"] = service_config.piper_voice
            tts_config["speed"] = service_config.piper_speed

        # Select handler implementations
        if use_http:
            logger.info("Using HTTP handlers for ASR/TTS")
            whisper_handler = WhisperHTTPHandler(whisper_config)
            tts_handler = TTSHTTPHandler(tts_config)
        else:
            logger.info("Using subprocess handlers for ASR/TTS")
            whisper_handler = WhisperHandler(whisper_config)
            tts_handler = TTSHandler(tts_config)

        # Z-Image config
        zimage_config = handlers_config.get("zimage", {})
        zimage_config.setdefault("device", "cpu")
        zimage_config.setdefault("output_dir", "output/images")

        # Initialize all handlers
        self.handlers: Dict[Handler, BaseHandler] = {
            Handler.TEXT_LOCAL: MistralHandler(mistral_base),
            Handler.TEXT_SOCIAL: MistralHandler(social_config),
            Handler.TEXT_LOGIC: MistralHandler(logic_config),
            Handler.AUDIO_IN: whisper_handler,
            Handler.AUDIO_OUT: tts_handler,
            Handler.IMAGE_CAPTION: CLIPHandler(handlers_config.get("clip", {})),
            Handler.IMAGE_GEN: ZImageHandler(zimage_config),
            Handler.EXTERNAL_LLM: ExternalHandler(handlers_config.get("external", {})),
            Handler.REJECT: RejectionHandler(),
        }

        # Document Parser service (dots.ocr - shared across text handlers)
        self.document_parser = self._init_document_parser(handlers_config)
        if self.document_parser:
            for handler_enum in (Handler.TEXT_LOCAL, Handler.TEXT_SOCIAL, Handler.TEXT_LOGIC):
                handler = self.handlers.get(handler_enum)
                if handler and hasattr(handler, 'set_document_parser'):
                    handler.set_document_parser(self.document_parser)

        # Vision Service (SmolVLM2 via Ollama - shared across text handlers)
        self.vision_service = self._init_vision_service(handlers_config)
        if self.vision_service:
            for handler_enum in (Handler.TEXT_LOCAL, Handler.TEXT_SOCIAL, Handler.TEXT_LOGIC):
                handler = self.handlers.get(handler_enum)
                if handler and hasattr(handler, 'set_vision_service'):
                    handler.set_vision_service(self.vision_service)

        # Initialize sandboxes for each handler
        self.sandboxes: Dict[Handler, ProcessSandbox] = {}
        for handler_enum in self.handlers.keys():
            config_key = HANDLER_CONFIG_KEYS.get(handler_enum, "default")
            sandbox_cfg = self._get_sandbox_config_for_handler(config_key)
            self.sandboxes[handler_enum] = ProcessSandbox(sandbox_cfg)

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load handler configuration."""
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _load_sandbox_config(self) -> Dict[str, Any]:
        """Load sandbox configuration from config file."""
        return self.config.get("sandbox", {})

    def _get_sandbox_config_for_handler(self, handler_key: str) -> SandboxConfig:
        """
        Get sandbox configuration for a specific handler.

        Priority:
        1. Handler-specific config from config file (sandbox.handler_key)
        2. Handler-specific defaults (HANDLER_SANDBOX_DEFAULTS)
        3. Global config from config file (sandbox.default)
        4. Global defaults (DEFAULT_SANDBOX_CONFIG)
        """
        # Start with global defaults
        config = DEFAULT_SANDBOX_CONFIG.copy()

        # Apply global config from file if present
        if "default" in self.sandbox_config:
            config.update(self.sandbox_config["default"])

        # Apply handler-specific defaults
        if handler_key in HANDLER_SANDBOX_DEFAULTS:
            config.update(HANDLER_SANDBOX_DEFAULTS[handler_key])

        # Apply handler-specific config from file if present
        if handler_key in self.sandbox_config:
            config.update(self.sandbox_config[handler_key])

        return SandboxConfig(
            cpu_seconds=config.get("cpu_seconds"),
            memory_mb=config.get("memory_mb"),
            timeout_seconds=config.get("timeout_seconds"),
        )

    # Handlers that are pure logic (no LLM, no I/O) and don't need sandboxing.
    # Sandboxing these on Windows can corrupt multiprocessing state on timeout.
    _SAFE_HANDLERS = {Handler.REJECT}

    def _should_bypass_sandbox(self, decision: PolicyDecision) -> bool:
        """
        Check if handler should bypass sandbox.

        Bypass for:
        - Safe handlers (reject - pure logic)
        - Text handlers with tools active (v5.0) - the ReAct loop makes
          HTTP calls to Ollama that fail inside Windows subprocess sandbox
        """
        if decision.handler in self._SAFE_HANDLERS:
            return True

        # v5.0: Text handlers with tools need main process for HTTP calls
        if decision.handler in (Handler.TEXT_LOCAL, Handler.TEXT_SOCIAL, Handler.TEXT_LOGIC):
            handler = self.handlers.get(decision.handler)
            if handler and getattr(handler, '_tool_registry', None) is not None:
                return True

        return False

    def route(self, request: Request, decision: PolicyDecision) -> Response:
        """
        Route request to appropriate handler via sandbox.

        This is the ONLY place where handlers are called.
        Safe handlers (reject) and tool-enabled text handlers run directly;
        all others go through ProcessSandbox.
        """
        handler = self.handlers.get(decision.handler)

        if handler is None:
            return Response.error_response(
                error=f"No handler for {decision.handler.value}",
                code="HANDLER_NOT_FOUND",
                meta=ResponseMeta(
                    request_id=request.request_id,
                    handler="none",
                    handler_reason=decision.reason
                )
            )

        # Bypass sandbox for safe handlers and tool-enabled text handlers
        if self._should_bypass_sandbox(decision):
            import time as _time
            _start = _time.time()
            try:
                result = handler.process(request)
                _elapsed = int((_time.time() - _start) * 1000)
                if not isinstance(result, Response):
                    result = Response.success_response(
                        output=result,
                        meta=ResponseMeta(
                            request_id=request.request_id,
                            handler=decision.handler.value,
                            handler_reason=decision.reason,
                            elapsed_ms=_elapsed,
                        )
                    )
                else:
                    result.meta.request_id = request.request_id
                    result.meta.handler = decision.handler.value
                    result.meta.handler_reason = decision.reason
                    result.meta.elapsed_ms = _elapsed
                return result
            except Exception as e:
                return Response.error_response(
                    error=str(e),
                    code="HANDLER_ERROR",
                    meta=ResponseMeta(
                        request_id=request.request_id,
                        handler=decision.handler.value,
                        handler_reason=decision.reason,
                    )
                )

        # Get sandbox for this handler
        sandbox = self.sandboxes.get(decision.handler)
        if sandbox is None:
            # Fallback to default sandbox if somehow missing
            sandbox = ProcessSandbox(SandboxConfig())

        # Execute handler through sandbox
        sandbox_result = sandbox.execute(_execute_handler, handler, request)

        # Handle sandbox result
        if not sandbox_result.success:
            return self._handle_sandbox_violation(
                request=request,
                decision=decision,
                sandbox_result=sandbox_result,
            )

        # Sandbox succeeded - process the handler output
        result = sandbox_result.output

        # Ensure result is a Response
        if not isinstance(result, Response):
            # Wrap raw output in Response
            result = Response.success_response(
                output=result,
                meta=ResponseMeta(
                    request_id=request.request_id,
                    handler=decision.handler.value,
                    handler_reason=decision.reason,
                    elapsed_ms=sandbox_result.elapsed_ms,
                )
            )
        else:
            # Update meta on existing Response
            result.meta.request_id = request.request_id
            result.meta.handler = decision.handler.value
            result.meta.handler_reason = decision.reason
            result.meta.elapsed_ms = sandbox_result.elapsed_ms

        # Track external API usage
        result.meta.used_external = decision.allow_external
        result.meta.external_justification = decision.external_justification

        return result

    def _handle_sandbox_violation(
        self,
        request: Request,
        decision: PolicyDecision,
        sandbox_result: SandboxResult,
    ) -> Response:
        """
        Create error response for sandbox violation.

        Maps sandbox violation types to structured error codes.
        Includes violation type and elapsed time in response.
        """
        violation = sandbox_result.violation
        error_code = SANDBOX_ERROR_CODES.get(violation, "SANDBOX_ERROR")

        # Build error message with violation details
        error_message = f"Sandbox violation: {sandbox_result.error}"

        # Store sandbox details in policy_decision dict for audit trail
        policy_with_sandbox = decision.to_dict()
        policy_with_sandbox["sandbox_violation"] = violation.value
        policy_with_sandbox["sandbox_elapsed_ms"] = sandbox_result.elapsed_ms
        policy_with_sandbox["sandbox_exit_code"] = sandbox_result.exit_code

        meta = ResponseMeta(
            request_id=request.request_id,
            handler=decision.handler.value,
            handler_reason=decision.reason,
            elapsed_ms=sandbox_result.elapsed_ms,
            policy_decision=policy_with_sandbox,
        )

        return Response.error_response(
            error=error_message,
            code=error_code,
            meta=meta,
        )

    def _init_document_parser(self, handlers_config: Dict) -> Optional[Any]:
        """Initialize document parser (dots.ocr) if configured and available."""
        try:
            from ..services.document_parser import DocumentParser

            parser_config = handlers_config.get("document_parser", {})
            if not parser_config.get("enabled", True):
                logger.info("Document parser disabled in config")
                return None

            parser = DocumentParser(parser_config)
            if parser.is_available():
                logger.info("Document parser (dots.ocr) initialized")
                return parser
            else:
                logger.info("Document parser dependencies not available, OCR disabled")
                return None

        except ImportError:
            logger.info("Document parser module not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize document parser: {e}")
            return None

    def _init_vision_service(self, handlers_config: Dict) -> Optional[Any]:
        """Initialize SmolVLM2 vision service if model is available."""
        try:
            from ..services.vision import VisionService

            vision_config = handlers_config.get("vision", {})
            if not vision_config.get("enabled", True):
                logger.info("Vision service disabled in config")
                return None

            # Use Ollama server URL from service config
            if self.service_config:
                vision_config.setdefault("base_url", self.service_config.llm_server)

            service = VisionService(vision_config)
            if service.is_available():
                logger.info(f"Vision service (SmolVLM2) initialized, model: {service.model}")
                return service
            else:
                logger.info("Vision model not available in Ollama, vision disabled")
                return None

        except ImportError:
            logger.info("Vision service module not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize vision service: {e}")
            return None

    def get_available_handlers(self) -> list:
        """Return list of available handlers."""
        return [h.value for h in self.handlers.keys()]

    def get_sandbox_config(self, handler: Handler) -> Optional[SandboxConfig]:
        """
        Get sandbox configuration for a handler.

        Useful for debugging and testing.
        """
        sandbox = self.sandboxes.get(handler)
        if sandbox:
            return sandbox.config
        return None
