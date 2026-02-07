"""
Hardcoded rules - NO ML HERE.

These rules are checked BEFORE the LLM is consulted.
They are fast, deterministic, and auditable.
"""

import re
from typing import Optional, Tuple
from ..models.request import Request, Modality
from ..models.policy import Classification


class Rules:
    """
    Deterministic rules for classification.

    ORDER OF PRIORITY:
    1. Blocked patterns (security)
    2. Modality detection (obvious from input type)
    3. Keyword-based intent detection
    4. If uncertain -> defer to LLM classifier
    """

    # Patterns that should NEVER be processed
    BLOCKED_PATTERNS = [
        # Add security-sensitive patterns here
        # Example: r"rm\s+-rf\s+/",
    ]

    # Keywords for quick intent detection
    QUESTION_KEYWORDS = [
        "what", "who", "where", "when", "why", "how",
        "cosa", "chi", "dove", "quando", "perche", "come",
        "is it", "can you", "could you", "would you",
        "explain", "describe", "spiega", "descrivi"
    ]

    COMMAND_KEYWORDS = [
        "create", "make", "build", "generate", "write",
        "crea", "fai", "costruisci", "genera", "scrivi",
        "delete", "remove", "update", "change",
        "elimina", "rimuovi", "aggiorna", "cambia"
    ]

    CODE_KEYWORDS = [
        "function", "class", "def ", "import ", "const ", "let ", "var ",
        "python", "javascript", "typescript", "java", "rust", "go",
        "code", "codice", "programma", "script"
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self._blocked_compiled = [re.compile(p, re.IGNORECASE) for p in self.BLOCKED_PATTERNS]

    def is_blocked(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text matches any blocked pattern.

        Returns: (is_blocked, pattern_matched)
        """
        for i, pattern in enumerate(self._blocked_compiled):
            if pattern.search(text):
                return True, self.BLOCKED_PATTERNS[i]
        return False, None

    def detect_modality(self, request: Request) -> Optional[Modality]:
        """
        Detect modality from request data.

        This is DETERMINISTIC - based on what data is present.
        """
        # Audio input takes priority
        if request.audio_path or request.audio_bytes:
            return Modality.AUDIO_INPUT

        # Image input
        if request.image_path or request.image_bytes:
            return Modality.IMAGE_CAPTION

        # Check text for TTS requests
        text_lower = request.text.lower()
        if any(kw in text_lower for kw in ["speak", "say", "read aloud", "leggi", "parla", "dimmi"]):
            return Modality.AUDIO_OUTPUT

        # Check for image generation requests
        if any(kw in text_lower for kw in ["generate image", "create image", "draw", "genera immagine", "crea immagine", "disegna"]):
            return Modality.IMAGE_GENERATION

        # Check for video (always external)
        if any(kw in text_lower for kw in ["video", "animate", "animation"]):
            return Modality.VIDEO

        # Default to text
        return Modality.TEXT

    def quick_classify(self, request: Request) -> Optional[Classification]:
        """
        Try to classify without LLM.

        Returns Classification if confident, None if LLM should be consulted.
        """
        text = request.text.lower().strip()

        if not text:
            return None

        # Check for questions
        is_question = (
            text.endswith("?") or
            any(text.startswith(kw) for kw in self.QUESTION_KEYWORDS)
        )

        if is_question:
            # Detect domain
            domain = self._detect_domain(text)
            return Classification(
                intent="question",
                domain=domain,
                complexity="simple" if len(text) < 100 else "moderate",
                requires_external=False,
                confidence=0.8,
                reasoning="Detected question pattern"
            )

        # Check for commands
        is_command = any(kw in text for kw in self.COMMAND_KEYWORDS)

        if is_command:
            domain = self._detect_domain(text)
            complexity = self._estimate_complexity(text)
            return Classification(
                intent="command",
                domain=domain,
                complexity=complexity,
                requires_external=complexity == "complex",
                confidence=0.7,
                reasoning="Detected command keywords"
            )

        # Can't determine with confidence -> return None
        return None

    def _detect_domain(self, text: str) -> str:
        """Detect domain from text content."""
        if any(kw in text for kw in self.CODE_KEYWORDS):
            return "code"

        if any(kw in text for kw in ["math", "calculate", "equation", "formula", "calcola"]):
            return "math"

        if any(kw in text for kw in ["image", "picture", "photo", "immagine", "foto"]):
            return "image"

        if any(kw in text for kw in ["audio", "sound", "music", "voice", "suono", "musica", "voce"]):
            return "audio"

        return "general"

    def _estimate_complexity(self, text: str) -> str:
        """Estimate task complexity from text length and keywords."""
        # Simple heuristics
        word_count = len(text.split())

        if word_count < 20:
            return "simple"
        elif word_count < 100:
            return "moderate"
        else:
            return "complex"
