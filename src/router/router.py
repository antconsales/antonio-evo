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
            "max_tokens": 1024,
            "persona_name": "SOCIAL",
        }

        # LOGIC persona config
        logic_config = {
            **mistral_base,
            "system_prompt_path": "prompts/mistral_logic.txt",
            "temperature": 0.3,
            "max_tokens": 2048,
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

    def route(self, request: Request, decision: PolicyDecision) -> Response:
        """
        Route request to appropriate handler via sandbox.

        This is the ONLY place where handlers are called.
        All execution goes through ProcessSandbox.
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
