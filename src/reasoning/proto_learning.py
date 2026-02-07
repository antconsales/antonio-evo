"""
Proto-Learning System for Antonio Evo Proto-AGI (v4.0).

Per Proto-AGI System Prompt:
You may participate in learning WITHOUT RETRAINING via:
- Concept abstraction
- Confidence adjustment
- Failure-driven updates

Learning MEANS:
- Refining internal representations
- Adjusting confidence
- Improving generalization

Learning does NOT mean:
- Changing your own rules
- Modifying policies
- Altering safety boundaries
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LearningType(Enum):
    """Types of proto-learning."""
    CONCEPT_ABSTRACTION = "concept_abstraction"  # Abstract patterns from examples
    CONFIDENCE_ADJUSTMENT = "confidence_adjustment"  # Update confidence based on feedback
    FAILURE_DRIVEN = "failure_driven"  # Learn from errors
    GENERALIZATION = "generalization"  # Extend to new cases
    REFINEMENT = "refinement"  # Improve existing knowledge


class FeedbackType(Enum):
    """Types of feedback for learning."""
    POSITIVE = "positive"  # Reinforcement
    NEGATIVE = "negative"  # Correction
    NEUTRAL = "neutral"  # Observation only
    EXPLICIT = "explicit"  # User-provided feedback
    IMPLICIT = "implicit"  # Inferred from context


@dataclass
class LearningEvent:
    """
    A learning event.

    Per spec: Learning is observational, not authoritative.
    """
    id: str
    learning_type: LearningType
    feedback_type: FeedbackType
    source: str  # Where the learning came from
    content: Dict[str, Any]  # What was learned
    context: Dict[str, Any]  # Learning context
    confidence_delta: float = 0.0  # Change in confidence
    created_at: float = field(default_factory=time.time)
    applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "learning_type": self.learning_type.value,
            "feedback_type": self.feedback_type.value,
            "source": self.source,
            "content": self.content,
            "context": self.context,
            "confidence_delta": self.confidence_delta,
            "created_at": self.created_at,
            "applied": self.applied,
        }


@dataclass
class ConceptAbstraction:
    """
    An abstracted concept from learning.

    Per spec: Concepts are domain-limited and probabilistic.
    """
    id: str
    name: str
    description: str
    source_examples: List[str]  # IDs of examples that led to this abstraction
    properties: Dict[str, Any]  # Abstracted properties
    domain: str  # Domain this applies to
    confidence: float = 0.5  # Confidence in abstraction
    mutable: bool = True  # Can be updated by further learning
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source_examples": self.source_examples,
            "properties": self.properties,
            "domain": self.domain,
            "confidence": self.confidence,
            "mutable": self.mutable,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    def update_confidence(self, delta: float):
        """
        Update confidence based on feedback.

        Per spec: Confidence adjustment is valid learning.
        """
        if not self.mutable:
            logger.warning(f"Cannot update immutable concept: {self.id}")
            return

        old_confidence = self.confidence
        self.confidence = max(0.0, min(1.0, self.confidence + delta))
        self.last_updated = time.time()
        logger.debug(f"Concept {self.id} confidence: {old_confidence:.2f} -> {self.confidence:.2f}")


@dataclass
class FailureRecord:
    """
    Record of a failure for failure-driven learning.

    Per spec: Failures drive learning without retraining.
    """
    id: str
    failure_type: str
    description: str
    context: Dict[str, Any]
    expected_outcome: str
    actual_outcome: str
    analysis: Optional[str] = None
    correction_applied: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "failure_type": self.failure_type,
            "description": self.description,
            "context": self.context,
            "expected_outcome": self.expected_outcome,
            "actual_outcome": self.actual_outcome,
            "analysis": self.analysis,
            "correction_applied": self.correction_applied,
            "created_at": self.created_at,
        }


@dataclass
class LearningBoundary:
    """
    Boundary that learning cannot cross.

    Per spec: Learning cannot modify policies or safety boundaries.
    """
    name: str
    description: str
    enforced: bool = True

    IMMUTABLE_BOUNDARIES = [
        ("core_axioms", "CODE DECIDES, MODELS DO NOT - Core system axiom"),
        ("safety_policies", "Safety policies cannot be modified by learning"),
        ("permission_model", "Permission and consent requirements are immutable"),
        ("identity", "System identity and constraints are fixed"),
        ("audit_requirements", "Audit and logging requirements are enforced"),
    ]


class ProtoLearner:
    """
    Proto-Learning engine for Antonio Evo.

    Per Proto-AGI Spec:
    - Learning happens without retraining
    - Learning refines representations
    - Learning adjusts confidence
    - Learning NEVER modifies rules or policies
    """

    def __init__(self):
        """Initialize the proto-learner."""
        self._learning_events: List[LearningEvent] = []
        self._abstractions: Dict[str, ConceptAbstraction] = {}
        self._failures: List[FailureRecord] = []
        self._boundaries = [
            LearningBoundary(name=name, description=desc)
            for name, desc in LearningBoundary.IMMUTABLE_BOUNDARIES
        ]

    def check_boundary(self, learning_target: str) -> Tuple[bool, str]:
        """
        Check if learning target crosses a boundary.

        Per spec: Certain things cannot be learned/modified.
        """
        target_lower = learning_target.lower()
        for boundary in self._boundaries:
            if boundary.name in target_lower:
                return False, f"Learning boundary: {boundary.description}"
        return True, ""

    def record_feedback(
        self,
        feedback_type: FeedbackType,
        content: Dict[str, Any],
        source: str = "interaction",
        context: Optional[Dict[str, Any]] = None,
    ) -> LearningEvent:
        """
        Record feedback for learning.

        Per spec: Learning from feedback is valid.
        """
        event = LearningEvent(
            id=str(uuid.uuid4())[:12],
            learning_type=LearningType.CONFIDENCE_ADJUSTMENT,
            feedback_type=feedback_type,
            source=source,
            content=content,
            context=context or {},
        )

        # Calculate confidence delta based on feedback
        if feedback_type == FeedbackType.POSITIVE:
            event.confidence_delta = 0.1
        elif feedback_type == FeedbackType.NEGATIVE:
            event.confidence_delta = -0.1
        elif feedback_type == FeedbackType.EXPLICIT:
            # Explicit feedback has more weight
            event.confidence_delta = content.get("confidence_delta", 0.15)

        self._learning_events.append(event)
        logger.info(f"Recorded learning event: {event.id} ({feedback_type.value})")
        return event

    def abstract_concept(
        self,
        examples: List[Dict[str, Any]],
        domain: str,
        name: str,
        description: str,
    ) -> Optional[ConceptAbstraction]:
        """
        Abstract a concept from examples.

        Per spec: Concept abstraction is valid learning.
        """
        if len(examples) < 2:
            logger.warning("Need at least 2 examples for abstraction")
            return None

        # Find common properties
        common_properties = {}
        first_example = examples[0]
        for key in first_example:
            values = [ex.get(key) for ex in examples if key in ex]
            if len(values) == len(examples):
                # All examples have this property
                if len(set(str(v) for v in values)) == 1:
                    # Same value across all examples
                    common_properties[key] = values[0]
                else:
                    # Variable property - note the pattern
                    common_properties[f"{key}_varies"] = True

        abstraction = ConceptAbstraction(
            id=str(uuid.uuid4())[:12],
            name=name,
            description=description,
            source_examples=[str(i) for i in range(len(examples))],
            properties=common_properties,
            domain=domain,
            confidence=0.5 + (len(examples) * 0.05),  # More examples = higher confidence
        )

        self._abstractions[abstraction.id] = abstraction

        # Record learning event
        event = LearningEvent(
            id=str(uuid.uuid4())[:12],
            learning_type=LearningType.CONCEPT_ABSTRACTION,
            feedback_type=FeedbackType.IMPLICIT,
            source="abstraction",
            content={"abstraction_id": abstraction.id},
            context={"example_count": len(examples), "domain": domain},
        )
        self._learning_events.append(event)

        logger.info(f"Abstracted concept: {abstraction.id} ({name})")
        return abstraction

    def record_failure(
        self,
        failure_type: str,
        description: str,
        expected: str,
        actual: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> FailureRecord:
        """
        Record a failure for failure-driven learning.

        Per spec: Failure-driven updates are valid learning.
        """
        failure = FailureRecord(
            id=str(uuid.uuid4())[:12],
            failure_type=failure_type,
            description=description,
            context=context or {},
            expected_outcome=expected,
            actual_outcome=actual,
        )

        self._failures.append(failure)

        # Record learning event
        event = LearningEvent(
            id=str(uuid.uuid4())[:12],
            learning_type=LearningType.FAILURE_DRIVEN,
            feedback_type=FeedbackType.NEGATIVE,
            source="failure",
            content={"failure_id": failure.id},
            context=context or {},
            confidence_delta=-0.15,  # Failures reduce confidence
        )
        self._learning_events.append(event)

        logger.info(f"Recorded failure: {failure.id} ({failure_type})")
        return failure

    def analyze_failure(
        self,
        failure_id: str,
        analysis: str,
    ) -> bool:
        """Analyze a recorded failure."""
        failure = next((f for f in self._failures if f.id == failure_id), None)
        if not failure:
            return False

        failure.analysis = analysis
        logger.info(f"Analyzed failure: {failure_id}")
        return True

    def update_concept_confidence(
        self,
        concept_id: str,
        feedback: FeedbackType,
        magnitude: float = 0.1,
    ) -> bool:
        """
        Update confidence for a concept.

        Per spec: Confidence adjustment is valid learning.
        """
        concept = self._abstractions.get(concept_id)
        if not concept:
            return False

        delta = magnitude if feedback == FeedbackType.POSITIVE else -magnitude
        concept.update_confidence(delta)

        # Record learning event
        event = LearningEvent(
            id=str(uuid.uuid4())[:12],
            learning_type=LearningType.CONFIDENCE_ADJUSTMENT,
            feedback_type=feedback,
            source="explicit_update",
            content={"concept_id": concept_id},
            context={},
            confidence_delta=delta,
            applied=True,
        )
        self._learning_events.append(event)

        return True

    def generalize_concept(
        self,
        concept_id: str,
        new_domain: str,
    ) -> Optional[ConceptAbstraction]:
        """
        Generalize a concept to a new domain.

        Per spec: Improving generalization is valid learning.
        """
        original = self._abstractions.get(concept_id)
        if not original:
            return None

        # Check boundary
        allowed, reason = self.check_boundary(new_domain)
        if not allowed:
            logger.warning(f"Cannot generalize to domain {new_domain}: {reason}")
            return None

        # Create generalized version with lower confidence
        generalized = ConceptAbstraction(
            id=str(uuid.uuid4())[:12],
            name=f"{original.name} (generalized)",
            description=f"Generalization of {original.name} to {new_domain}",
            source_examples=original.source_examples,
            properties=original.properties.copy(),
            domain=new_domain,
            confidence=original.confidence * 0.6,  # Lower confidence for generalization
        )

        self._abstractions[generalized.id] = generalized

        # Record learning event
        event = LearningEvent(
            id=str(uuid.uuid4())[:12],
            learning_type=LearningType.GENERALIZATION,
            feedback_type=FeedbackType.IMPLICIT,
            source="generalization",
            content={
                "original_id": concept_id,
                "generalized_id": generalized.id,
            },
            context={"new_domain": new_domain},
        )
        self._learning_events.append(event)

        logger.info(f"Generalized concept {concept_id} to {generalized.id}")
        return generalized

    def refine_concept(
        self,
        concept_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """
        Refine a concept with new information.

        Per spec: Refining representations is valid learning.
        """
        concept = self._abstractions.get(concept_id)
        if not concept or not concept.mutable:
            return False

        # Apply updates to properties
        for key, value in updates.items():
            concept.properties[key] = value

        concept.last_updated = time.time()

        # Record learning event
        event = LearningEvent(
            id=str(uuid.uuid4())[:12],
            learning_type=LearningType.REFINEMENT,
            feedback_type=FeedbackType.EXPLICIT,
            source="refinement",
            content={"concept_id": concept_id, "updates": updates},
            context={},
            applied=True,
        )
        self._learning_events.append(event)

        logger.info(f"Refined concept: {concept_id}")
        return True

    def get_concept(self, concept_id: str) -> Optional[ConceptAbstraction]:
        """Get a concept by ID."""
        return self._abstractions.get(concept_id)

    def list_concepts(self, domain: Optional[str] = None) -> List[ConceptAbstraction]:
        """List concepts, optionally filtered by domain."""
        concepts = list(self._abstractions.values())
        if domain:
            concepts = [c for c in concepts if c.domain == domain]
        return concepts

    def get_learning_history(self, limit: int = 50) -> List[LearningEvent]:
        """Get recent learning events."""
        return self._learning_events[-limit:]

    def get_failure_history(self, limit: int = 20) -> List[FailureRecord]:
        """Get recent failures."""
        return self._failures[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_learning_events": len(self._learning_events),
            "total_abstractions": len(self._abstractions),
            "total_failures": len(self._failures),
            "boundaries_enforced": len(self._boundaries),
            "learning_types": {
                lt.value: sum(
                    1 for e in self._learning_events if e.learning_type == lt
                )
                for lt in LearningType
            },
        }


# Singleton instance
_proto_learner: Optional[ProtoLearner] = None


def get_proto_learner() -> ProtoLearner:
    """Get or create the proto-learner singleton."""
    global _proto_learner
    if _proto_learner is None:
        _proto_learner = ProtoLearner()
    return _proto_learner
