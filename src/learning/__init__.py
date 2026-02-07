"""
Learning module for Antonio Evo.

Handles auto-learning from interactions:
- Neuron creation (storing successful interactions)
- Confidence scoring and updates
- Preference learning

Philosophy: Learn from experience, but code controls what gets learned.
"""

from .neuron_creator import NeuronCreator

__all__ = [
    "NeuronCreator",
]
