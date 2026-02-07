"""
Reasoning Module for Antonio Evo Proto-AGI (v4.0).

This module implements bounded, adaptive reasoning capabilities:
- Cognitive budget management
- Internal simulation (thinking without acting)
- Concept graphs and world representation
- Proto-learning without retraining

Per Proto-AGI Spec:
- You reason under a cognitive budget
- You may simulate outcomes internally
- Simulations are non-executing and non-authoritative
- Learning refines representations, never modifies rules
"""

from .cognitive_budget import (
    AbstractionLevel,
    BudgetExceededError,
    BudgetConstraint,
    ReasoningDepthLimit,
    ContextSizeLimit,
    CognitiveBudget,
    CognitiveBudgetManager,
    get_budget_manager,
)

from .simulation import (
    SimulationType,
    SimulationStatus,
    SimulationScenario,
    SimulationOutcome,
    SimulationResult,
    InternalSimulator,
    get_simulator,
    format_simulated_reasoning,
    SIMULATION_DISCLOSURE,
)

from .concept_graph import (
    RelationType,
    ConceptSource,
    ConceptNode,
    ConceptRelation,
    HeuristicRule,
    ConceptGraph,
    get_concept_graph,
)

from .proto_learning import (
    LearningType,
    FeedbackType,
    LearningEvent,
    ConceptAbstraction,
    FailureRecord,
    LearningBoundary,
    ProtoLearner,
    get_proto_learner,
)

__all__ = [
    # Cognitive Budget
    "AbstractionLevel",
    "BudgetExceededError",
    "BudgetConstraint",
    "ReasoningDepthLimit",
    "ContextSizeLimit",
    "CognitiveBudget",
    "CognitiveBudgetManager",
    "get_budget_manager",
    # Simulation
    "SimulationType",
    "SimulationStatus",
    "SimulationScenario",
    "SimulationOutcome",
    "SimulationResult",
    "InternalSimulator",
    "get_simulator",
    "format_simulated_reasoning",
    "SIMULATION_DISCLOSURE",
    # Concept Graph
    "RelationType",
    "ConceptSource",
    "ConceptNode",
    "ConceptRelation",
    "HeuristicRule",
    "ConceptGraph",
    "get_concept_graph",
    # Proto-Learning
    "LearningType",
    "FeedbackType",
    "LearningEvent",
    "ConceptAbstraction",
    "FailureRecord",
    "LearningBoundary",
    "ProtoLearner",
    "get_proto_learner",
]
