"""
Persona module for Antonio Evo.

Handles dual-model persona selection:
- SOCIAL: Conversational, empathetic, friendly
- LOGIC: Analytical, precise, structured

Philosophy: Code decides the persona, not the LLM.
"""

from .selector import PersonaSelector

__all__ = [
    "PersonaSelector",
]
