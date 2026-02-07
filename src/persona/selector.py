"""
Persona Selector for Antonio Evo.

Determines which persona (SOCIAL or LOGIC) should handle a request.
Decision is DETERMINISTIC - based on rules, not LLM suggestions.

Order of evaluation (first match wins):
1. Explicit user request in text
2. Memory-learned preference for this domain
3. Classification-based rules
4. Keyword detection
5. Domain defaults
"""

from typing import Optional, Dict, Any

from ..models.policy import Persona, Classification
from ..models.request import Request


class PersonaSelector:
    """
    Selects the appropriate persona for a request.

    This is a pure function - no side effects, no state changes.
    """

    # Keywords that suggest SOCIAL persona
    SOCIAL_KEYWORDS = [
        # Friendly/casual
        "ciao", "hello", "hi", "hey", "salve",
        "grazie", "thanks", "thank you",
        "please", "per favore", "perfavore",
        # Help-seeking
        "help me", "aiutami", "can you", "puoi",
        "could you", "potresti",
        # Emotional
        "feeling", "mi sento", "sono triste", "sono felice",
        # Simple explanations
        "explain simply", "spiega semplicemente",
        "eli5", "for a beginner", "per un principiante",
        "in simple terms", "in parole semplici",
    ]

    # Keywords that suggest LOGIC persona
    LOGIC_KEYWORDS = [
        # Technical precision
        "technical", "tecnico", "precisely", "precisamente",
        "exact", "esatto", "accurate", "accurato",
        # Analysis
        "analyze", "analizza", "analyse",
        "compare", "confronta", "comparison",
        "evaluate", "valuta",
        # Code/Math
        "debug", "optimize", "ottimizza",
        "calculate", "calcola", "compute",
        "algorithm", "algoritmo",
        "complexity", "complessità",
        "performance", "benchmark",
        # Structured output
        "step by step", "passo passo",
        "list the", "elenca",
        "pros and cons", "pro e contro",
    ]

    # Domain to persona mapping
    DOMAIN_DEFAULTS = {
        "code": Persona.LOGIC,
        "math": Persona.LOGIC,
        "analysis": Persona.LOGIC,
        "data": Persona.LOGIC,
        "science": Persona.LOGIC,
        "general": Persona.SOCIAL,
        "text": Persona.SOCIAL,
        "conversation": Persona.SOCIAL,
        "creative": Persona.SOCIAL,
        "image": Persona.SOCIAL,
        "audio": Persona.SOCIAL,
    }

    # Intent to persona mapping
    INTENT_DEFAULTS = {
        "question": Persona.SOCIAL,  # Questions are usually conversational
        "command": Persona.LOGIC,     # Commands need precision
        "generation": Persona.SOCIAL, # Creative generation
        "analysis": Persona.LOGIC,    # Analysis needs structure
        "conversation": Persona.SOCIAL,
        "unknown": Persona.SOCIAL,    # Default to friendly
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize selector.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}

        # Allow config to override defaults
        self.default_persona = Persona(
            self.config.get("default_persona", "social")
        )

    def select(
        self,
        request: Request,
        classification: Classification,
    ) -> Persona:
        """
        Select the appropriate persona for a request.

        Args:
            request: The normalized request
            classification: The classification result

        Returns:
            Selected Persona (SOCIAL or LOGIC)
        """
        text_lower = request.text.lower() if request.text else ""

        # === RULE 1: Explicit user request ===
        explicit = self._check_explicit_request(text_lower)
        if explicit:
            return explicit

        # === RULE 2: Memory-learned preference ===
        if request.memory_context:
            memory_pref = self._check_memory_preference(
                request.memory_context,
                classification.domain,
            )
            if memory_pref:
                return memory_pref

        # === RULE 3: Classification suggestion ===
        if classification.suggested_persona != Persona.AUTO:
            return classification.suggested_persona

        # === RULE 4: Keyword detection ===
        keyword_result = self._check_keywords(text_lower)
        if keyword_result:
            return keyword_result

        # === RULE 5: Domain + Intent based ===
        domain_persona = self.DOMAIN_DEFAULTS.get(classification.domain)
        intent_persona = self.INTENT_DEFAULTS.get(classification.intent)

        # If domain and intent agree, use that
        if domain_persona and domain_persona == intent_persona:
            return domain_persona

        # Domain takes precedence for technical domains
        if classification.domain in ["code", "math", "analysis", "data"]:
            return Persona.LOGIC

        # Intent takes precedence for conversational intents
        if classification.intent in ["conversation", "question"]:
            return Persona.SOCIAL

        # Use domain default if available
        if domain_persona:
            return domain_persona

        # Use intent default if available
        if intent_persona:
            return intent_persona

        # === FALLBACK: Default persona ===
        return self.default_persona

    def _check_explicit_request(self, text: str) -> Optional[Persona]:
        """
        Check if user explicitly requested a persona.

        Args:
            text: Lowercase request text

        Returns:
            Persona if explicitly requested, None otherwise
        """
        # Check for explicit SOCIAL requests
        social_explicit = [
            "be friendly", "sii amichevole",
            "explain like", "spiegami come",
            "talk to me", "parlami",
            "casual", "informale",
        ]
        for phrase in social_explicit:
            if phrase in text:
                return Persona.SOCIAL

        # Check for explicit LOGIC requests
        logic_explicit = [
            "be precise", "sii preciso",
            "technical details", "dettagli tecnici",
            "formal", "formale",
            "structured", "strutturato",
            "as an expert", "come esperto",
        ]
        for phrase in logic_explicit:
            if phrase in text:
                return Persona.LOGIC

        return None

    def _check_memory_preference(
        self,
        memory_context: Any,
        domain: str,
    ) -> Optional[Persona]:
        """
        Check if user has a learned preference for this domain.

        Args:
            memory_context: The MemoryContext object
            domain: Classification domain

        Returns:
            Persona if preference exists with high confidence, None otherwise
        """
        # Check for domain-specific preference
        pref_key = f"persona_{domain}"
        pref_value = memory_context.get_preference(pref_key)

        if pref_value:
            try:
                return Persona(pref_value)
            except ValueError:
                pass

        # Check for general preference
        general_pref = memory_context.get_preference("default_persona")
        if general_pref:
            try:
                return Persona(general_pref)
            except ValueError:
                pass

        # Check what persona was used for similar past queries
        if memory_context.relevant_neurons:
            # Get the persona from the best matching neuron
            best_neuron = memory_context.relevant_neurons[0]
            if best_neuron.relevance_score > 0.8:  # High relevance
                return best_neuron.neuron.persona

        return None

    def _check_keywords(self, text: str) -> Optional[Persona]:
        """
        Check for persona-indicating keywords.

        Args:
            text: Lowercase request text

        Returns:
            Persona based on keyword count, None if inconclusive
        """
        social_count = sum(1 for kw in self.SOCIAL_KEYWORDS if kw in text)
        logic_count = sum(1 for kw in self.LOGIC_KEYWORDS if kw in text)

        # Need clear winner (at least 2 difference)
        if social_count > logic_count + 1:
            return Persona.SOCIAL
        if logic_count > social_count + 1:
            return Persona.LOGIC

        return None

    def get_selection_reason(
        self,
        request: Request,
        classification: Classification,
        selected: Persona,
    ) -> str:
        """
        Get human-readable reason for persona selection.

        Useful for debugging and transparency.

        Args:
            request: The request
            classification: The classification
            selected: The selected persona

        Returns:
            Reason string
        """
        text_lower = request.text.lower() if request.text else ""

        # Check what rule matched
        if self._check_explicit_request(text_lower):
            return f"Explicit user request for {selected.value} style"

        if request.memory_context:
            if self._check_memory_preference(request.memory_context, classification.domain):
                return f"Learned preference for {selected.value} in {classification.domain} domain"

        if classification.suggested_persona != Persona.AUTO:
            return f"Classifier suggested {selected.value}"

        if self._check_keywords(text_lower):
            return f"Keyword analysis indicated {selected.value}"

        if classification.domain in self.DOMAIN_DEFAULTS:
            return f"Domain '{classification.domain}' defaults to {selected.value}"

        if classification.intent in self.INTENT_DEFAULTS:
            return f"Intent '{classification.intent}' defaults to {selected.value}"

        return f"Default persona: {selected.value}"
