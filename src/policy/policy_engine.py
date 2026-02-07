"""
Policy Engine - THE BRAIN OF THE SYSTEM

All routing decisions happen HERE, not in the LLM.
The LLM can suggest, but this code DECIDES.

Now with:
- Persona selection (SOCIAL/LOGIC)
- Memory-informed routing
"""

import json
import time
from typing import Optional, Dict, Any

from ..models.request import Request, Modality
from ..models.policy import PolicyDecision, Handler, RejectReason, Classification, Persona
from ..persona.selector import PersonaSelector


class PolicyEngine:
    """
    Deterministic policy engine.

    RULE: Code decides, LLM does not.
    """

    def __init__(self, config_path: str = "config/policy.json"):
        self.config = self._load_config(config_path)

        # Load persona config
        personas_config = self._load_config("config/personas.json")
        self.persona_selector = PersonaSelector(personas_config.get("routing_rules", {}))

        # Rate limiting state
        self._request_count = 0
        self._external_count = 0
        self._minute_start = time.time()
        self._hour_start = time.time()

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load policy configuration."""
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "rate_limits": {
                "requests_per_minute": 30,
                "external_calls_per_hour": 10
            },
            "blocked_patterns": [],
            "max_input_length": 4000,
            "allow_external_fallback": True,
            "external_triggers": {
                "min_complexity": "complex",
                "min_tokens": 4000,
                "modalities": ["video"]
            }
        }

    def decide(self, request: Request, classification: Classification) -> PolicyDecision:
        """
        Main decision function.

        Order of checks:
        1. Rate limits
        2. Blocked classification
        3. Input validation
        4. Modality routing (deterministic)
        5. Complexity-based routing
        """

        # === CHECK 1: Rate limits ===
        if self._is_rate_limited():
            return PolicyDecision(
                handler=Handler.REJECT,
                reason="Rate limit exceeded",
                reject_reason=RejectReason.RATE_LIMITED
            )

        # === CHECK 2: Blocked classification ===
        if classification.intent == "blocked":
            return PolicyDecision(
                handler=Handler.REJECT,
                reason=classification.reasoning,
                reject_reason=RejectReason.BLOCKED_CONTENT
            )

        # === CHECK 3: Input validation ===
        if len(request.text) > self.config["max_input_length"]:
            if self.config["allow_external_fallback"] and self._can_use_external():
                return PolicyDecision(
                    handler=Handler.EXTERNAL_LLM,
                    reason=f"Input too long ({len(request.text)} chars)",
                    allow_external=True,
                    external_justification="Context length exceeds local model capacity"
                )
            else:
                return PolicyDecision(
                    handler=Handler.REJECT,
                    reason="Input too long, external fallback disabled",
                    reject_reason=RejectReason.TOO_COMPLEX_NO_FALLBACK
                )

        # === CHECK 4: Modality routing (DETERMINISTIC) ===
        modality = request.modality

        if modality == Modality.AUDIO_INPUT:
            return PolicyDecision(
                handler=Handler.AUDIO_IN,
                reason="Audio input detected -> Whisper"
            )

        if modality == Modality.AUDIO_OUTPUT:
            return PolicyDecision(
                handler=Handler.AUDIO_OUT,
                reason="Audio output requested -> TTS"
            )

        if modality == Modality.IMAGE_CAPTION:
            return PolicyDecision(
                handler=Handler.IMAGE_CAPTION,
                reason="Image understanding -> CLIP"
            )

        if modality == Modality.IMAGE_GENERATION:
            # Image gen is LIMITED on CPU
            if request.quality == "high":
                if self.config["allow_external_fallback"] and self._can_use_external():
                    return PolicyDecision(
                        handler=Handler.EXTERNAL_LLM,
                        reason="High quality image requested",
                        allow_external=True,
                        external_justification="Local SD is slow and low quality"
                    )
            return PolicyDecision(
                handler=Handler.IMAGE_GEN,
                reason="Simple image generation -> SD local (slow)",
                metadata={"warning": "This will be slow on CPU"}
            )

        if modality == Modality.VIDEO:
            # NEVER local
            if self.config["allow_external_fallback"] and self._can_use_external():
                return PolicyDecision(
                    handler=Handler.EXTERNAL_LLM,
                    reason="Video is NEVER local",
                    allow_external=True,
                    external_justification="Video generation not supported locally"
                )
            return PolicyDecision(
                handler=Handler.REJECT,
                reason="Video not supported, external fallback disabled",
                reject_reason=RejectReason.UNSUPPORTED
            )

        # === CHECK 5: Complexity-based routing ===
        complexity = classification.complexity
        requires_external = classification.requires_external

        if requires_external or complexity == "complex":
            if self.config["allow_external_fallback"] and self._can_use_external():
                return PolicyDecision(
                    handler=Handler.EXTERNAL_LLM,
                    reason=f"Complexity={complexity}, classification suggests external",
                    allow_external=True,
                    external_justification=classification.reasoning
                )
            # Fall through to local if external not available

        # === CHECK 6: Persona selection ===
        persona = self.persona_selector.select(request, classification)
        persona_reason = self.persona_selector.get_selection_reason(
            request, classification, persona
        )

        # Map persona to handler
        if persona == Persona.LOGIC:
            handler = Handler.TEXT_LOGIC
            handler_name = "Mistral LOGIC"
        else:
            handler = Handler.TEXT_SOCIAL
            handler_name = "Mistral SOCIAL"

        # === DEFAULT: Local text processing with persona ===
        return PolicyDecision(
            handler=handler,
            reason=f"Text request -> {handler_name} ({persona_reason})",
            persona=persona,
        )

    def _is_rate_limited(self) -> bool:
        """Check if rate limited."""
        now = time.time()

        # Reset minute counter
        if now - self._minute_start > 60:
            self._request_count = 0
            self._minute_start = now

        # Reset hour counter
        if now - self._hour_start > 3600:
            self._external_count = 0
            self._hour_start = now

        self._request_count += 1
        return self._request_count > self.config["rate_limits"]["requests_per_minute"]

    def _can_use_external(self) -> bool:
        """Check if external API quota available."""
        return self._external_count < self.config["rate_limits"]["external_calls_per_hour"]

    def record_external_call(self):
        """Record that an external call was made."""
        self._external_count += 1
