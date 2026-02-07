"""
EvoMemory System - Evolutionary Memory for Antonio Evo.

This module provides persistent, learning memory capabilities:
- Neuron storage (input/output pairs with confidence)
- BM25-based retrieval (RAG-Lite without vector DB)
- Session tracking
- Preference learning
- Emotional memory and sentiment tracking (v2.1)

Philosophy: Memory informs, Code decides.
"""

from .neuron import Neuron, NeuronCreate, MemoryContext
from .storage import MemoryStorage
from .retriever import MemoryRetriever
from .emotional import (
    EmotionalMemory,
    EmotionalContext,
    EmotionalSignal,
    EmotionalAnalyzer,
    UserEmotionalState,
    ToneRecommendation,
)

__all__ = [
    "Neuron",
    "NeuronCreate",
    "MemoryContext",
    "MemoryStorage",
    "MemoryRetriever",
    # v2.1 Emotional Memory
    "EmotionalMemory",
    "EmotionalContext",
    "EmotionalSignal",
    "EmotionalAnalyzer",
    "UserEmotionalState",
    "ToneRecommendation",
]
