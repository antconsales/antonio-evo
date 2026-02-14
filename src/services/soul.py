"""
Soul Engine — Antonio's deep identity system.

Loads the soul definition (prompts/soul.md) and dynamically builds
system prompts that incorporate:
- Core identity and values
- Persona-specific behavior (SOCIAL vs LOGIC)
- Emotional adaptation from EmotionalMemory
- Personality traits from PersonalityEvolutionEngine

The Soul is not a static prompt — it evolves with the user.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SoulEngine:
    """
    Antonio's soul — the foundation of every response.

    Merges static identity (soul.md) with dynamic signals
    (personality traits, emotional context, persona mode).
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.soul_path = Path(config.get("soul_path", "prompts/soul.md"))
        self.enable_personality = config.get("enable_personality_adaptation", True)
        self.enable_emotional = config.get("enable_emotional_adaptation", True)
        self.name = config.get("name", "Antonio")
        self.version = config.get("version", "6.0")

        # Optional injected services
        self._personality_engine = None
        self._emotional_memory = None

        # Load soul definition
        self._soul_text = self._load_soul()
        logger.info(f"Soul Engine loaded ({self.soul_path}, {len(self._soul_text)} chars)")

    def set_personality_engine(self, engine) -> None:
        """Inject PersonalityEvolutionEngine for trait-based adaptation."""
        self._personality_engine = engine

    def set_emotional_memory(self, memory) -> None:
        """Inject EmotionalMemory for tone adaptation."""
        self._emotional_memory = memory

    def _load_soul(self) -> str:
        """Load soul definition from markdown file."""
        try:
            if self.soul_path.exists():
                return self.soul_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to load soul from {self.soul_path}: {e}")
        return f"Sei {self.name}, un assistente AI locale amichevole e competente."

    def get_system_prompt(
        self,
        persona: str = "SOCIAL",
        emotional_context=None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Build a complete system prompt from soul + dynamic signals.

        Args:
            persona: "SOCIAL" or "LOGIC"
            emotional_context: EmotionalContext from EmotionalMemory
            session_id: Current session for emotional lookup
        """
        parts = [self._soul_text]

        # Persona-specific behavior
        if persona == "LOGIC":
            parts.append(
                "\n\n## Modalità Corrente: LOGIC\n"
                "In questa conversazione, usa uno stile analitico e preciso. "
                "Struttura le risposte con chiarezza. Usa dati e fatti. "
                "Mantieni un tono professionale ma accessibile."
            )
        else:
            parts.append(
                "\n\n## Modalità Corrente: SOCIAL\n"
                "In questa conversazione, usa uno stile amichevole e naturale. "
                "Rispondi come un amico esperto. Sii diretto e conciso. "
                "Non analizzare troppo — rispondi alla domanda."
            )

        # Personality trait adaptation
        if self.enable_personality and self._personality_engine:
            try:
                traits = self._personality_engine.get_current_traits()
                if traits:
                    trait_instructions = self._traits_to_instructions(traits)
                    if trait_instructions:
                        parts.append(f"\n\n## Adattamento Personalità\n{trait_instructions}")
            except Exception as e:
                logger.debug(f"Personality adaptation skipped: {e}")

        # Emotional adaptation
        if self.enable_emotional and emotional_context:
            try:
                tone_instruction = self._emotional_to_instruction(emotional_context)
                if tone_instruction:
                    parts.append(f"\n\n## Adattamento Emotivo\n{tone_instruction}")
            except Exception as e:
                logger.debug(f"Emotional adaptation skipped: {e}")

        return "\n".join(parts)

    def _traits_to_instructions(self, traits) -> str:
        """Convert personality traits to natural language instructions."""
        instructions = []

        # traits is expected to be a dict or object with trait values 0.0-1.0
        trait_dict = traits if isinstance(traits, dict) else getattr(traits, '__dict__', {})

        formality = trait_dict.get("formality", 0.5)
        verbosity = trait_dict.get("verbosity", 0.5)
        humor = trait_dict.get("humor", 0.5)
        empathy = trait_dict.get("empathy", 0.5)

        if formality > 0.7:
            instructions.append("Usa un tono più formale del solito.")
        elif formality < 0.3:
            instructions.append("Sii molto informale, come un amico stretto.")

        if verbosity > 0.7:
            instructions.append("L'utente preferisce risposte dettagliate.")
        elif verbosity < 0.3:
            instructions.append("L'utente preferisce risposte brevi e concise.")

        if humor > 0.7:
            instructions.append("Puoi usare umorismo e battute quando appropriato.")

        if empathy > 0.7:
            instructions.append("Mostra particolare attenzione ai sentimenti dell'utente.")

        return " ".join(instructions)

    def _emotional_to_instruction(self, emotional_context) -> str:
        """Convert emotional context to tone instruction."""
        tone = getattr(emotional_context, "tone_recommendation", None)
        state = getattr(emotional_context, "current_state", None)

        if not tone and not state:
            return ""

        tone_value = tone.value if hasattr(tone, "value") else str(tone) if tone else ""
        state_value = state.value if hasattr(state, "value") else str(state) if state else ""

        tone_map = {
            "supportive": "L'utente sembra aver bisogno di supporto. Sii incoraggiante e paziente.",
            "enthusiastic": "L'utente è entusiasta! Condividi il suo entusiasmo.",
            "calm": "Mantieni un tono calmo e rassicurante.",
            "direct": "L'utente vuole risposte dirette. Vai al punto.",
            "empathetic": "Mostra empatia e comprensione per la situazione dell'utente.",
        }

        if tone_value in tone_map:
            return tone_map[tone_value]

        if state_value:
            return f"L'utente sembra in uno stato emotivo: {state_value}. Adatta il tono di conseguenza."

        return ""

    def get_identity(self) -> Dict[str, Any]:
        """Return identity info for API responses."""
        return {
            "name": self.name,
            "version": self.version,
            "soul_loaded": bool(self._soul_text),
            "personality_adaptation": self.enable_personality and self._personality_engine is not None,
            "emotional_adaptation": self.enable_emotional and self._emotional_memory is not None,
        }
