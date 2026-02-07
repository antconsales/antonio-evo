"""
Classifier - Uses rules first, LLM second.

The LLM is ONLY used when rules can't determine intent.
"""

import json
import requests
from typing import Optional, Dict, Any

from .rules import Rules
from ..models.request import Request
from ..models.policy import Classification


class Classifier:
    """
    Intent classifier.

    STRATEGY:
    1. Try rules-based classification (fast, deterministic)
    2. If uncertain, use LLM for classification ONLY
    3. LLM cannot execute - it only classifies
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "mistral"):
        self.rules = Rules()
        self.ollama_url = ollama_url
        self.model = model
        self._system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the Mistral system prompt."""
        try:
            with open("prompts/mistral_system.txt") as f:
                return f.read()
        except FileNotFoundError:
            return "You are a classifier. Respond only with JSON."

    def classify(self, request: Request) -> Classification:
        """
        Classify the request.

        Returns Classification with intent, domain, complexity, etc.
        """

        # === STEP 1: Check blocked patterns ===
        is_blocked, pattern = self.rules.is_blocked(request.text)
        if is_blocked:
            return Classification(
                intent="blocked",
                domain="security",
                complexity="simple",
                requires_external=False,
                confidence=1.0,
                reasoning=f"Blocked by pattern: {pattern}"
            )

        # === STEP 2: Try rules-based classification ===
        rule_classification = self.rules.quick_classify(request)
        if rule_classification and rule_classification.confidence >= 0.7:
            return rule_classification

        # === STEP 3: Use LLM for classification ===
        llm_classification = self._llm_classify(request)
        if llm_classification:
            return llm_classification

        # === FALLBACK: Default classification ===
        return Classification(
            intent="unknown",
            domain="general",
            complexity="moderate",
            requires_external=False,
            confidence=0.3,
            reasoning="Could not classify with confidence"
        )

    def _llm_classify(self, request: Request) -> Optional[Classification]:
        """
        Use LLM for classification.

        The LLM ONLY classifies - it cannot execute anything.
        """

        prompt = f"""TASK: CLASSIFY

INPUT: {request.text[:500]}

Classify this input. Respond with JSON:
{{
  "intent": "question|command|generation|analysis|unknown",
  "domain": "text|code|math|image|audio|general",
  "complexity": "simple|moderate|complex",
  "requires_external": false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": self._system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Very low for determinism
                        "num_predict": 256
                    }
                },
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            output = result.get("response", "")

            # Parse JSON from response
            try:
                # Try to extract JSON from response
                json_start = output.find("{")
                json_end = output.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = output[json_start:json_end]
                    data = json.loads(json_str)
                    return Classification.from_dict(data)
            except json.JSONDecodeError:
                pass

        except requests.RequestException:
            pass

        return None
