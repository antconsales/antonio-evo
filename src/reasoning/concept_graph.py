"""
Concept Graph for Antonio Evo Proto-AGI (v4.0).

Per Proto-AGI System Prompt:
You may reason using ABSTRACT INTERNAL MODELS, such as:
- Concept graphs
- Cause-effect relationships
- Heuristic rules

These models are:
- Incomplete
- Probabilistic
- Domain-limited

They exist to support simulation, NOT action.

Concept Properties:
- confidence: How certain the concept is (0.0-1.0)
- source: Where the concept originated
- domain: What domain it applies to
- relations: Links to other concepts
- mutable: Can be updated by learning
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of relations between concepts."""
    IS_A = "is_a"  # Taxonomy (dog IS_A animal)
    HAS_A = "has_a"  # Composition (car HAS_A engine)
    PART_OF = "part_of"  # Meronymy (wheel PART_OF car)
    CAUSES = "causes"  # Causation (fire CAUSES heat)
    CAUSED_BY = "caused_by"  # Inverse causation
    PRECEDES = "precedes"  # Temporal (spring PRECEDES summer)
    FOLLOWS = "follows"  # Inverse temporal
    RELATED_TO = "related_to"  # General relation
    SIMILAR_TO = "similar_to"  # Similarity
    OPPOSITE_OF = "opposite_of"  # Opposition
    REQUIRES = "requires"  # Dependency (cooking REQUIRES heat)
    ENABLES = "enables"  # Enablement (key ENABLES lock_open)


class ConceptSource(Enum):
    """Sources of concept knowledge."""
    TRAINING = "training"  # From model training
    USER_STATED = "user_stated"  # Explicitly stated by user
    INFERRED = "inferred"  # Inferred from context
    ABSTRACTED = "abstracted"  # Abstracted from examples
    EXTERNAL = "external"  # From external source (if allowed)


@dataclass
class ConceptNode:
    """
    A concept in the graph.

    Per spec: Concepts are incomplete, probabilistic, domain-limited.
    """
    id: str
    name: str
    description: str
    domain: str
    confidence: float = 0.5  # 0.0-1.0
    source: ConceptSource = ConceptSource.INFERRED
    properties: Dict[str, Any] = field(default_factory=dict)
    mutable: bool = True
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "confidence": self.confidence,
            "source": self.source.value,
            "properties": self.properties,
            "mutable": self.mutable,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
        }

    def access(self):
        """Record an access to this concept."""
        self.last_accessed = time.time()
        self.access_count += 1

    def update_confidence(self, delta: float):
        """Update confidence within bounds."""
        if self.mutable:
            self.confidence = max(0.0, min(1.0, self.confidence + delta))


@dataclass
class ConceptRelation:
    """
    A relation between concepts.

    Per spec: Cause-effect relationships are part of world model.
    """
    id: str
    source_id: str  # Source concept ID
    target_id: str  # Target concept ID
    relation_type: RelationType
    confidence: float = 0.5  # Confidence in this relation
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "confidence": self.confidence,
            "properties": self.properties,
            "created_at": self.created_at,
        }


@dataclass
class HeuristicRule:
    """
    A heuristic rule for reasoning.

    Per spec: Heuristic rules are part of internal models.
    """
    id: str
    name: str
    condition: str  # Natural language condition
    action: str  # Natural language consequence
    domain: str
    confidence: float = 0.5
    exceptions: List[str] = field(default_factory=list)
    source: ConceptSource = ConceptSource.INFERRED
    usage_count: int = 0
    success_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "condition": self.condition,
            "action": self.action,
            "domain": self.domain,
            "confidence": self.confidence,
            "exceptions": self.exceptions,
            "source": self.source.value,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / max(1, self.usage_count),
        }

    def record_usage(self, success: bool):
        """Record usage of this heuristic."""
        self.usage_count += 1
        if success:
            self.success_count += 1


class ConceptGraph:
    """
    Concept graph for world representation.

    Per Proto-AGI Spec:
    - Incomplete: Does not represent full reality
    - Probabilistic: All knowledge has confidence
    - Domain-limited: Organized by domain
    - For simulation: Supports reasoning, not action
    """

    def __init__(self):
        """Initialize the concept graph."""
        self._concepts: Dict[str, ConceptNode] = {}
        self._relations: Dict[str, ConceptRelation] = {}
        self._heuristics: Dict[str, HeuristicRule] = {}
        self._domain_index: Dict[str, Set[str]] = {}  # domain -> concept IDs

    def add_concept(
        self,
        name: str,
        description: str,
        domain: str,
        confidence: float = 0.5,
        source: ConceptSource = ConceptSource.INFERRED,
        properties: Optional[Dict[str, Any]] = None,
        mutable: bool = True,
    ) -> ConceptNode:
        """
        Add a concept to the graph.

        Per spec: Concepts are domain-limited.
        """
        concept_id = str(uuid.uuid4())[:12]
        concept = ConceptNode(
            id=concept_id,
            name=name,
            description=description,
            domain=domain,
            confidence=confidence,
            source=source,
            properties=properties or {},
            mutable=mutable,
        )

        self._concepts[concept_id] = concept

        # Update domain index
        if domain not in self._domain_index:
            self._domain_index[domain] = set()
        self._domain_index[domain].add(concept_id)

        logger.debug(f"Added concept: {name} ({domain})")
        return concept

    def get_concept(self, concept_id: str) -> Optional[ConceptNode]:
        """Get a concept by ID."""
        concept = self._concepts.get(concept_id)
        if concept:
            concept.access()
        return concept

    def find_concept_by_name(
        self,
        name: str,
        domain: Optional[str] = None,
    ) -> Optional[ConceptNode]:
        """Find a concept by name, optionally filtered by domain."""
        name_lower = name.lower()
        for concept in self._concepts.values():
            if concept.name.lower() == name_lower:
                if domain is None or concept.domain == domain:
                    concept.access()
                    return concept
        return None

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        confidence: float = 0.5,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConceptRelation]:
        """
        Add a relation between concepts.

        Per spec: Cause-effect relationships are valid.
        """
        if source_id not in self._concepts or target_id not in self._concepts:
            logger.warning("Cannot add relation: concept not found")
            return None

        relation_id = str(uuid.uuid4())[:12]
        relation = ConceptRelation(
            id=relation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            properties=properties or {},
        )

        self._relations[relation_id] = relation
        logger.debug(f"Added relation: {relation_type.value} ({source_id} -> {target_id})")
        return relation

    def get_relations(
        self,
        concept_id: str,
        direction: str = "both",  # "outgoing", "incoming", "both"
        relation_type: Optional[RelationType] = None,
    ) -> List[ConceptRelation]:
        """Get relations for a concept."""
        relations = []
        for relation in self._relations.values():
            if direction in ("outgoing", "both") and relation.source_id == concept_id:
                if relation_type is None or relation.relation_type == relation_type:
                    relations.append(relation)
            if direction in ("incoming", "both") and relation.target_id == concept_id:
                if relation_type is None or relation.relation_type == relation_type:
                    relations.append(relation)
        return relations

    def add_causal_chain(
        self,
        cause_id: str,
        effect_id: str,
        confidence: float = 0.5,
    ) -> Optional[ConceptRelation]:
        """
        Add a cause-effect relationship.

        Per spec: Cause-effect relationships are part of world model.
        """
        return self.add_relation(
            source_id=cause_id,
            target_id=effect_id,
            relation_type=RelationType.CAUSES,
            confidence=confidence,
        )

    def add_heuristic(
        self,
        name: str,
        condition: str,
        action: str,
        domain: str,
        confidence: float = 0.5,
        exceptions: Optional[List[str]] = None,
        source: ConceptSource = ConceptSource.INFERRED,
    ) -> HeuristicRule:
        """
        Add a heuristic rule.

        Per spec: Heuristic rules are part of internal models.
        """
        heuristic_id = str(uuid.uuid4())[:12]
        heuristic = HeuristicRule(
            id=heuristic_id,
            name=name,
            condition=condition,
            action=action,
            domain=domain,
            confidence=confidence,
            exceptions=exceptions or [],
            source=source,
        )

        self._heuristics[heuristic_id] = heuristic
        logger.debug(f"Added heuristic: {name} ({domain})")
        return heuristic

    def get_heuristics(
        self,
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[HeuristicRule]:
        """Get heuristics, optionally filtered."""
        heuristics = list(self._heuristics.values())
        if domain:
            heuristics = [h for h in heuristics if h.domain == domain]
        heuristics = [h for h in heuristics if h.confidence >= min_confidence]
        return heuristics

    def get_concepts_by_domain(self, domain: str) -> List[ConceptNode]:
        """Get all concepts in a domain."""
        concept_ids = self._domain_index.get(domain, set())
        return [self._concepts[cid] for cid in concept_ids if cid in self._concepts]

    def list_domains(self) -> List[str]:
        """List all domains in the graph."""
        return list(self._domain_index.keys())

    def trace_causation(
        self,
        concept_id: str,
        direction: str = "forward",  # "forward" (effects) or "backward" (causes)
        max_depth: int = 5,
    ) -> List[Tuple[ConceptNode, int]]:
        """
        Trace causal chain from a concept.

        Per spec: Cause-effect tracing for simulation.
        Returns list of (concept, depth) tuples.
        """
        result = []
        visited = set()
        relation_type = RelationType.CAUSES if direction == "forward" else RelationType.CAUSED_BY

        def trace(cid: str, depth: int):
            if cid in visited or depth > max_depth:
                return
            visited.add(cid)

            concept = self._concepts.get(cid)
            if concept:
                result.append((concept, depth))

            relations = self.get_relations(
                cid,
                direction="outgoing" if direction == "forward" else "incoming",
                relation_type=relation_type,
            )
            for rel in relations:
                next_id = rel.target_id if direction == "forward" else rel.source_id
                trace(next_id, depth + 1)

        trace(concept_id, 0)
        return result

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 10,
    ) -> Optional[List[Tuple[ConceptNode, ConceptRelation]]]:
        """Find a path between two concepts."""
        if start_id not in self._concepts or end_id not in self._concepts:
            return None

        visited = set()
        queue = [(start_id, [])]

        while queue:
            current_id, path = queue.pop(0)
            if current_id == end_id:
                return path

            if current_id in visited or len(path) > max_depth:
                continue
            visited.add(current_id)

            relations = self.get_relations(current_id, direction="outgoing")
            for relation in relations:
                concept = self._concepts.get(relation.target_id)
                if concept:
                    new_path = path + [(concept, relation)]
                    queue.append((relation.target_id, new_path))

        return None

    def get_related_concepts(
        self,
        concept_id: str,
        max_depth: int = 2,
        min_confidence: float = 0.3,
    ) -> List[Tuple[ConceptNode, float]]:
        """
        Get concepts related to a given concept.

        Returns list of (concept, relevance_score) tuples.
        """
        result = {}
        visited = set()

        def explore(cid: str, depth: int, accumulated_confidence: float):
            if cid in visited or depth > max_depth:
                return
            visited.add(cid)

            relations = self.get_relations(cid, direction="both")
            for relation in relations:
                other_id = (
                    relation.target_id
                    if relation.source_id == cid
                    else relation.source_id
                )
                if other_id == concept_id:
                    continue

                concept = self._concepts.get(other_id)
                if concept and concept.confidence >= min_confidence:
                    relevance = accumulated_confidence * relation.confidence
                    if other_id not in result or result[other_id] < relevance:
                        result[other_id] = relevance

                    explore(other_id, depth + 1, relevance)

        explore(concept_id, 0, 1.0)
        return [
            (self._concepts[cid], score)
            for cid, score in sorted(result.items(), key=lambda x: -x[1])
            if cid in self._concepts
        ]

    def prune_low_confidence(self, min_confidence: float = 0.2):
        """
        Remove low-confidence concepts and relations.

        Per spec: Concepts are probabilistic and may decay.
        """
        # Identify concepts to remove
        to_remove = [
            cid for cid, c in self._concepts.items()
            if c.confidence < min_confidence and c.mutable
        ]

        # Remove concepts
        for cid in to_remove:
            concept = self._concepts.pop(cid)
            # Update domain index
            if concept.domain in self._domain_index:
                self._domain_index[concept.domain].discard(cid)
            logger.debug(f"Pruned concept: {concept.name}")

        # Remove orphaned relations
        relation_ids_to_remove = []
        for rid, rel in self._relations.items():
            if rel.source_id not in self._concepts or rel.target_id not in self._concepts:
                relation_ids_to_remove.append(rid)
            elif rel.confidence < min_confidence:
                relation_ids_to_remove.append(rid)

        for rid in relation_ids_to_remove:
            self._relations.pop(rid)

        logger.info(f"Pruned {len(to_remove)} concepts and {len(relation_ids_to_remove)} relations")

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_concepts": len(self._concepts),
            "total_relations": len(self._relations),
            "total_heuristics": len(self._heuristics),
            "domains": list(self._domain_index.keys()),
            "domain_counts": {d: len(ids) for d, ids in self._domain_index.items()},
            "relation_types": {
                rt.value: sum(1 for r in self._relations.values() if r.relation_type == rt)
                for rt in RelationType
            },
        }

    def export_subgraph(
        self,
        concept_ids: List[str],
    ) -> Dict[str, Any]:
        """Export a subgraph containing specified concepts."""
        concepts = [
            self._concepts[cid].to_dict()
            for cid in concept_ids
            if cid in self._concepts
        ]
        relations = [
            r.to_dict()
            for r in self._relations.values()
            if r.source_id in concept_ids and r.target_id in concept_ids
        ]
        return {
            "concepts": concepts,
            "relations": relations,
        }


# Singleton instance
_concept_graph: Optional[ConceptGraph] = None


def get_concept_graph() -> ConceptGraph:
    """Get or create the concept graph singleton."""
    global _concept_graph
    if _concept_graph is None:
        _concept_graph = ConceptGraph()
    return _concept_graph
