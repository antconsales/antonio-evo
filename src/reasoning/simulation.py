"""
Internal Simulation Engine for Antonio Evo Proto-AGI (v4.0).

Per Proto-AGI System Prompt:
You may simulate outcomes internally.

Simulation Is:
- Non-executing
- Non-persistent
- Non-authoritative

You May Use Simulation To:
- Compare strategies
- Anticipate consequences
- Reject unsafe paths
- Explore hypotheticals

You Must Clearly Distinguish:
- Simulated reasoning
- Real-world actions

NO SIMULATION IMPLIES EXECUTION.

When presenting simulated reasoning, frame it as:
    [Simulated reasoning - not executed]
    If X were to happen, then Y would likely follow...
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SimulationType(Enum):
    """Types of internal simulation."""
    STRATEGY_COMPARISON = "strategy_comparison"  # Compare alternative approaches
    CONSEQUENCE_ANALYSIS = "consequence_analysis"  # Anticipate outcomes
    SAFETY_CHECK = "safety_check"  # Reject unsafe paths
    HYPOTHETICAL = "hypothetical"  # Explore "what if" scenarios
    COUNTERFACTUAL = "counterfactual"  # Reason about alternatives


class SimulationStatus(Enum):
    """Status of a simulation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    BUDGET_EXCEEDED = "budget_exceeded"


# Mandatory disclosure prefix for simulated reasoning
SIMULATION_DISCLOSURE = "[Simulated reasoning - not executed]"


@dataclass
class SimulationScenario:
    """
    A scenario to simulate.

    Per spec: Simulations are hypothetical explorations.
    """
    id: str
    description: str
    preconditions: List[str]  # What must be true for this scenario
    variables: Dict[str, Any]  # Variables in the scenario
    constraints: List[str]  # Constraints on the simulation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "preconditions": self.preconditions,
            "variables": self.variables,
            "constraints": self.constraints,
        }


@dataclass
class SimulationOutcome:
    """
    An outcome from a simulation.

    Per spec: Outcomes are non-authoritative predictions.
    """
    scenario_id: str
    outcome_id: str
    description: str
    probability: float  # Estimated probability (0.0-1.0)
    confidence: float  # Confidence in the estimate (0.0-1.0)
    consequences: List[str]  # Predicted consequences
    risks: List[str]  # Identified risks
    is_safe: bool  # Whether outcome is safe

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "outcome_id": self.outcome_id,
            "description": self.description,
            "probability": self.probability,
            "confidence": self.confidence,
            "consequences": self.consequences,
            "risks": self.risks,
            "is_safe": self.is_safe,
        }

    def format_disclosure(self) -> str:
        """
        Format outcome with mandatory disclosure.

        Per spec: Always clearly mark as simulated.
        """
        lines = [
            SIMULATION_DISCLOSURE,
            f"If {self.description}, then:",
        ]

        for consequence in self.consequences:
            lines.append(f"  - {consequence}")

        if self.risks:
            lines.append("Identified risks:")
            for risk in self.risks:
                lines.append(f"  - {risk}")

        lines.append(f"Estimated probability: {self.probability:.0%}")
        lines.append(f"Confidence level: {self.confidence:.0%}")

        return "\n".join(lines)


@dataclass
class SimulationResult:
    """
    Complete result of a simulation run.

    Per spec: Results are for reasoning support only.
    """
    simulation_id: str
    simulation_type: SimulationType
    status: SimulationStatus
    scenarios: List[SimulationScenario]
    outcomes: List[SimulationOutcome]
    best_outcome: Optional[str] = None  # ID of recommended outcome
    unsafe_paths: List[str] = field(default_factory=list)  # Rejected paths
    reasoning_steps: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    budget_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "simulation_type": self.simulation_type.value,
            "status": self.status.value,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "outcomes": [o.to_dict() for o in self.outcomes],
            "best_outcome": self.best_outcome,
            "unsafe_paths": self.unsafe_paths,
            "reasoning_steps": self.reasoning_steps,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "budget_used": self.budget_used,
        }

    def get_safe_outcomes(self) -> List[SimulationOutcome]:
        """Get only safe outcomes."""
        return [o for o in self.outcomes if o.is_safe]

    def format_summary(self) -> str:
        """
        Format a summary of the simulation.

        Per spec: Always disclose this is simulated reasoning.
        """
        lines = [
            "=" * 50,
            SIMULATION_DISCLOSURE,
            "=" * 50,
            f"Simulation Type: {self.simulation_type.value}",
            f"Status: {self.status.value}",
            f"Scenarios Explored: {len(self.scenarios)}",
            f"Outcomes Generated: {len(self.outcomes)}",
            f"Safe Outcomes: {len(self.get_safe_outcomes())}",
            f"Unsafe Paths Rejected: {len(self.unsafe_paths)}",
            "",
        ]

        if self.reasoning_steps:
            lines.append("Reasoning Steps:")
            for i, step in enumerate(self.reasoning_steps, 1):
                lines.append(f"  {i}. {step}")
            lines.append("")

        if self.best_outcome:
            best = next((o for o in self.outcomes if o.outcome_id == self.best_outcome), None)
            if best:
                lines.append("Recommended Path:")
                lines.append(f"  {best.description}")
                lines.append(f"  Confidence: {best.confidence:.0%}")

        lines.append("")
        lines.append("NOTE: This analysis is simulated, not executed.")
        lines.append("No real-world actions have been taken.")
        lines.append("=" * 50)

        return "\n".join(lines)


class InternalSimulator:
    """
    Internal simulation engine for Antonio Evo.

    Per Proto-AGI Spec:
    - Simulations are non-executing
    - Simulations are non-persistent
    - Simulations are non-authoritative
    - Simulations help with reasoning, not action
    """

    def __init__(self, simulation_budget: int = 3):
        """
        Initialize the simulator.

        Args:
            simulation_budget: Maximum simulations allowed
        """
        self.simulation_budget = simulation_budget
        self.simulations_used = 0
        self._active_simulation: Optional[SimulationResult] = None
        self._history: List[SimulationResult] = []

    def can_simulate(self) -> bool:
        """Check if simulation budget is available."""
        return self.simulations_used < self.simulation_budget

    def remaining_budget(self) -> int:
        """Get remaining simulation budget."""
        return max(0, self.simulation_budget - self.simulations_used)

    def create_simulation(
        self,
        simulation_type: SimulationType,
        scenarios: List[SimulationScenario],
    ) -> Optional[SimulationResult]:
        """
        Create a new simulation.

        Per spec: Simulations are for reasoning support only.
        Returns None if budget exceeded.
        """
        if not self.can_simulate():
            logger.warning("Simulation budget exceeded")
            return None

        simulation_id = str(uuid.uuid4())[:12]

        result = SimulationResult(
            simulation_id=simulation_id,
            simulation_type=simulation_type,
            status=SimulationStatus.PENDING,
            scenarios=scenarios,
            outcomes=[],
        )

        self._active_simulation = result
        self.simulations_used += 1
        result.budget_used = 1

        logger.info(f"Created simulation: {simulation_id} ({simulation_type.value})")
        return result

    def add_reasoning_step(self, step: str):
        """Add a reasoning step to the active simulation."""
        if self._active_simulation:
            self._active_simulation.reasoning_steps.append(step)

    def add_outcome(self, outcome: SimulationOutcome):
        """Add an outcome to the active simulation."""
        if self._active_simulation:
            self._active_simulation.outcomes.append(outcome)
            if not outcome.is_safe:
                self._active_simulation.unsafe_paths.append(outcome.outcome_id)

    def reject_unsafe_path(self, description: str):
        """
        Reject an unsafe path.

        Per spec: Simulations can reject unsafe paths.
        """
        if self._active_simulation:
            self._active_simulation.unsafe_paths.append(description)
            self.add_reasoning_step(f"Rejected unsafe path: {description}")

    def select_best_outcome(self, outcome_id: str):
        """Select the best outcome from the simulation."""
        if self._active_simulation:
            # Verify it's a safe outcome
            outcome = next(
                (o for o in self._active_simulation.outcomes if o.outcome_id == outcome_id),
                None
            )
            if outcome and outcome.is_safe:
                self._active_simulation.best_outcome = outcome_id
            elif outcome:
                logger.warning(f"Cannot select unsafe outcome: {outcome_id}")

    def complete_simulation(self) -> Optional[SimulationResult]:
        """
        Complete the active simulation.

        Per spec: Results are non-authoritative.
        """
        if not self._active_simulation:
            return None

        self._active_simulation.status = SimulationStatus.COMPLETED
        self._active_simulation.completed_at = time.time()

        result = self._active_simulation
        self._history.append(result)
        self._active_simulation = None

        logger.info(f"Completed simulation: {result.simulation_id}")
        return result

    def abort_simulation(self, reason: str = "") -> Optional[SimulationResult]:
        """Abort the active simulation."""
        if not self._active_simulation:
            return None

        self._active_simulation.status = SimulationStatus.ABORTED
        self._active_simulation.completed_at = time.time()
        if reason:
            self.add_reasoning_step(f"Aborted: {reason}")

        result = self._active_simulation
        self._history.append(result)
        self._active_simulation = None

        logger.info(f"Aborted simulation: {result.simulation_id}")
        return result

    def compare_strategies(
        self,
        strategies: List[Dict[str, Any]],
    ) -> Optional[SimulationResult]:
        """
        Compare multiple strategies.

        Per spec: Simulation can compare strategies.
        """
        scenarios = [
            SimulationScenario(
                id=f"strategy_{i}",
                description=s.get("description", f"Strategy {i}"),
                preconditions=s.get("preconditions", []),
                variables=s.get("variables", {}),
                constraints=s.get("constraints", []),
            )
            for i, s in enumerate(strategies)
        ]

        return self.create_simulation(
            SimulationType.STRATEGY_COMPARISON,
            scenarios,
        )

    def analyze_consequences(
        self,
        action: str,
        context: Dict[str, Any],
    ) -> Optional[SimulationResult]:
        """
        Analyze consequences of an action.

        Per spec: Simulation can anticipate consequences.
        """
        scenario = SimulationScenario(
            id="consequence_analysis",
            description=f"Analyzing: {action}",
            preconditions=context.get("preconditions", []),
            variables=context,
            constraints=context.get("constraints", []),
        )

        return self.create_simulation(
            SimulationType.CONSEQUENCE_ANALYSIS,
            [scenario],
        )

    def check_safety(
        self,
        proposed_action: str,
        safety_constraints: List[str],
    ) -> Optional[SimulationResult]:
        """
        Check if an action is safe.

        Per spec: Simulation can reject unsafe paths.
        """
        scenario = SimulationScenario(
            id="safety_check",
            description=f"Safety analysis: {proposed_action}",
            preconditions=[],
            variables={"action": proposed_action},
            constraints=safety_constraints,
        )

        return self.create_simulation(
            SimulationType.SAFETY_CHECK,
            [scenario],
        )

    def explore_hypothetical(
        self,
        hypothesis: str,
        assumptions: List[str],
    ) -> Optional[SimulationResult]:
        """
        Explore a hypothetical scenario.

        Per spec: Simulation can explore hypotheticals.
        """
        scenario = SimulationScenario(
            id="hypothetical",
            description=hypothesis,
            preconditions=assumptions,
            variables={"hypothesis": hypothesis},
            constraints=[],
        )

        return self.create_simulation(
            SimulationType.HYPOTHETICAL,
            [scenario],
        )

    def get_history(self, limit: int = 10) -> List[SimulationResult]:
        """Get simulation history."""
        return self._history[-limit:]

    def reset_budget(self):
        """Reset the simulation budget."""
        self.simulations_used = 0
        self._active_simulation = None

    def get_status(self) -> Dict[str, Any]:
        """Get simulator status."""
        return {
            "budget_total": self.simulation_budget,
            "budget_used": self.simulations_used,
            "budget_remaining": self.remaining_budget(),
            "has_active_simulation": self._active_simulation is not None,
            "history_count": len(self._history),
        }


# Singleton instance
_simulator: Optional[InternalSimulator] = None


def get_simulator(budget: int = 3) -> InternalSimulator:
    """Get or create the simulator singleton."""
    global _simulator
    if _simulator is None:
        _simulator = InternalSimulator(simulation_budget=budget)
    return _simulator


def format_simulated_reasoning(reasoning: str) -> str:
    """
    Format reasoning with mandatory simulation disclosure.

    Per spec: Always clearly mark simulated reasoning.
    """
    return f"{SIMULATION_DISCLOSURE}\n{reasoning}"
