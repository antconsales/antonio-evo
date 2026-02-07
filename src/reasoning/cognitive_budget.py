"""
Cognitive Budget Management for Antonio Evo Proto-AGI (v4.0).

Per Proto-AGI System Prompt:
- You reason under a cognitive budget
- If a request exceeds your budget, you must:
  1. State the limitation clearly
  2. Reduce scope
  3. Decompose the problem
  4. Or propose an alternative
- You must NEVER hallucinate capability

Budget Parameters:
- max_reasoning_depth: Maximum chain-of-thought steps
- max_context_tokens: Available context window
- abstraction_level: Allowed abstraction complexity
- simulation_budget: Max internal simulations
- external_allowed: Can request external processing
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AbstractionLevel(Enum):
    """Allowed abstraction complexity levels."""
    LOW = "low"          # Concrete, specific reasoning only
    MEDIUM = "medium"    # Standard abstraction allowed
    HIGH = "high"        # Complex abstractions permitted
    UNLIMITED = "unlimited"  # No abstraction limits


class BudgetExceededError(Exception):
    """Raised when cognitive budget is exceeded."""

    def __init__(
        self,
        constraint: str,
        requested: Any,
        available: Any,
        suggestion: str = "",
    ):
        self.constraint = constraint
        self.requested = requested
        self.available = available
        self.suggestion = suggestion
        message = (
            f"Cognitive budget exceeded: {constraint}. "
            f"Requested: {requested}, Available: {available}. "
            f"{suggestion}"
        )
        super().__init__(message)


@dataclass
class BudgetConstraint:
    """A single budget constraint."""
    name: str
    limit: Any
    current: Any = 0
    unit: str = ""
    exceeded: bool = False

    def check(self, requested: Any = 1) -> bool:
        """Check if constraint would be exceeded."""
        if isinstance(self.limit, (int, float)) and isinstance(requested, (int, float)):
            return (self.current + requested) <= self.limit
        return True

    def consume(self, amount: Any = 1) -> bool:
        """Consume budget. Returns False if exceeded."""
        if not self.check(amount):
            self.exceeded = True
            return False
        if isinstance(self.current, (int, float)) and isinstance(amount, (int, float)):
            self.current += amount
        return True

    def remaining(self) -> Any:
        """Get remaining budget."""
        if isinstance(self.limit, (int, float)) and isinstance(self.current, (int, float)):
            return max(0, self.limit - self.current)
        return self.limit

    def reset(self):
        """Reset constraint."""
        self.current = 0
        self.exceeded = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "limit": self.limit,
            "current": self.current,
            "remaining": self.remaining(),
            "unit": self.unit,
            "exceeded": self.exceeded,
        }


@dataclass
class ReasoningDepthLimit:
    """Limit on chain-of-thought reasoning depth."""
    max_steps: int = 5
    current_step: int = 0
    allow_decomposition: bool = True

    def can_continue(self) -> bool:
        """Check if more reasoning steps are allowed."""
        return self.current_step < self.max_steps

    def step(self) -> bool:
        """Attempt to take a reasoning step."""
        if self.can_continue():
            self.current_step += 1
            return True
        return False

    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.current_step)

    def reset(self):
        self.current_step = 0


@dataclass
class ContextSizeLimit:
    """Limit on context window usage."""
    max_tokens: int = 4096
    used_tokens: int = 0
    reserve_for_output: int = 512

    def available_for_input(self) -> int:
        """Get tokens available for input."""
        return max(0, self.max_tokens - self.reserve_for_output - self.used_tokens)

    def can_fit(self, tokens: int) -> bool:
        """Check if tokens can fit."""
        return tokens <= self.available_for_input()

    def consume(self, tokens: int) -> bool:
        """Consume tokens. Returns False if exceeded."""
        if not self.can_fit(tokens):
            return False
        self.used_tokens += tokens
        return True

    def reset(self):
        self.used_tokens = 0


@dataclass
class CognitiveBudget:
    """
    Complete cognitive budget for a reasoning session.

    Per Proto-AGI Spec:
    - You reason under a cognitive budget
    - The system provides maximum reasoning depth, context size, etc.
    - You must never hallucinate capability
    """

    # Core limits
    max_reasoning_depth: int = 5
    max_context_tokens: int = 4096
    abstraction_level: AbstractionLevel = AbstractionLevel.MEDIUM
    simulation_budget: int = 3
    external_allowed: bool = False

    # Runtime state
    reasoning_depth: ReasoningDepthLimit = field(default_factory=lambda: ReasoningDepthLimit())
    context_size: ContextSizeLimit = field(default_factory=lambda: ContextSizeLimit())
    simulations_used: int = 0

    # Tracking
    created_at: float = field(default_factory=time.time)
    last_check: float = field(default_factory=time.time)
    violations: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize derived limits."""
        self.reasoning_depth = ReasoningDepthLimit(max_steps=self.max_reasoning_depth)
        self.context_size = ContextSizeLimit(max_tokens=self.max_context_tokens)

    def can_reason_deeper(self) -> bool:
        """Check if more reasoning depth is available."""
        return self.reasoning_depth.can_continue()

    def take_reasoning_step(self) -> bool:
        """Take a reasoning step. Returns False if budget exceeded."""
        if not self.reasoning_depth.step():
            self.violations.append("reasoning_depth_exceeded")
            return False
        return True

    def can_simulate(self) -> bool:
        """Check if simulation budget is available."""
        return self.simulations_used < self.simulation_budget

    def use_simulation(self) -> bool:
        """Use a simulation slot. Returns False if budget exceeded."""
        if not self.can_simulate():
            self.violations.append("simulation_budget_exceeded")
            return False
        self.simulations_used += 1
        return True

    def can_use_external(self) -> bool:
        """Check if external processing is allowed."""
        return self.external_allowed

    def can_fit_context(self, tokens: int) -> bool:
        """Check if tokens can fit in context."""
        return self.context_size.can_fit(tokens)

    def consume_context(self, tokens: int) -> bool:
        """Consume context tokens. Returns False if exceeded."""
        if not self.context_size.consume(tokens):
            self.violations.append("context_size_exceeded")
            return False
        return True

    def check_abstraction(self, level: AbstractionLevel) -> bool:
        """Check if abstraction level is allowed."""
        levels = [AbstractionLevel.LOW, AbstractionLevel.MEDIUM, AbstractionLevel.HIGH, AbstractionLevel.UNLIMITED]
        allowed_idx = levels.index(self.abstraction_level)
        requested_idx = levels.index(level)
        return requested_idx <= allowed_idx

    def get_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        return {
            "reasoning_depth": {
                "max": self.max_reasoning_depth,
                "used": self.reasoning_depth.current_step,
                "remaining": self.reasoning_depth.remaining_steps(),
            },
            "context": {
                "max": self.max_context_tokens,
                "used": self.context_size.used_tokens,
                "available": self.context_size.available_for_input(),
            },
            "simulations": {
                "max": self.simulation_budget,
                "used": self.simulations_used,
                "remaining": self.simulation_budget - self.simulations_used,
            },
            "abstraction_level": self.abstraction_level.value,
            "external_allowed": self.external_allowed,
            "violations": self.violations,
            "is_healthy": len(self.violations) == 0,
        }

    def get_limitation_message(self) -> Optional[str]:
        """
        Get a message explaining current limitations.

        Per Proto-AGI Spec: State limitations clearly.
        """
        limitations = []

        if not self.can_reason_deeper():
            limitations.append(
                f"Reasoning depth limit reached ({self.max_reasoning_depth} steps). "
                "Consider decomposing the problem into smaller parts."
            )

        if self.context_size.available_for_input() < 500:
            limitations.append(
                f"Context space is limited ({self.context_size.available_for_input()} tokens remaining). "
                "Consider reducing input size or summarizing context."
            )

        if not self.can_simulate():
            limitations.append(
                f"Simulation budget exhausted ({self.simulation_budget} simulations used). "
                "Cannot explore additional hypothetical scenarios."
            )

        if not self.external_allowed:
            limitations.append(
                "External processing is not enabled in current runtime profile. "
                "Complex queries requiring external knowledge cannot be fully addressed."
            )

        if limitations:
            return "\n".join(limitations)
        return None

    def propose_alternative(self, request_type: str) -> str:
        """
        Propose an alternative approach when budget is exceeded.

        Per Proto-AGI Spec: Propose alternatives when limits are reached.
        """
        alternatives = {
            "reasoning_depth": (
                "The problem can be decomposed into smaller sub-problems. "
                "Would you like me to break this down into simpler steps?"
            ),
            "context_size": (
                "The input exceeds available context. Options:\n"
                "1. Summarize the key points\n"
                "2. Focus on a specific section\n"
                "3. Process in chunks"
            ),
            "simulation": (
                "Simulation budget is exhausted. I can provide:\n"
                "1. Direct analysis without simulation\n"
                "2. Previously simulated results\n"
                "3. A request to increase simulation budget"
            ),
            "external": (
                "External processing is not available. I can:\n"
                "1. Provide best-effort local analysis\n"
                "2. Clearly state what information is missing\n"
                "3. Suggest enabling external access for this request"
            ),
            "abstraction": (
                "The requested abstraction level exceeds current limits. I can:\n"
                "1. Provide a more concrete, specific analysis\n"
                "2. Break down into simpler concepts\n"
                "3. Request higher abstraction permissions"
            ),
        }
        return alternatives.get(request_type, "Consider simplifying the request or adjusting budget parameters.")

    def reset(self):
        """Reset budget for new session."""
        self.reasoning_depth.reset()
        self.context_size.reset()
        self.simulations_used = 0
        self.violations.clear()
        self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "4.0",
            "max_reasoning_depth": self.max_reasoning_depth,
            "max_context_tokens": self.max_context_tokens,
            "abstraction_level": self.abstraction_level.value,
            "simulation_budget": self.simulation_budget,
            "external_allowed": self.external_allowed,
            "status": self.get_status(),
            "created_at": self.created_at,
        }


class CognitiveBudgetManager:
    """
    Manages cognitive budgets across sessions.

    Per Proto-AGI Spec:
    - Budget is tied to runtime profile
    - Budget must be respected at all times
    """

    # Profile-based default budgets
    PROFILE_BUDGETS = {
        "EVO-LITE": CognitiveBudget(
            max_reasoning_depth=3,
            max_context_tokens=2048,
            abstraction_level=AbstractionLevel.LOW,
            simulation_budget=0,
            external_allowed=False,
        ),
        "EVO-STANDARD": CognitiveBudget(
            max_reasoning_depth=5,
            max_context_tokens=4096,
            abstraction_level=AbstractionLevel.MEDIUM,
            simulation_budget=3,
            external_allowed=False,
        ),
        "EVO-FULL": CognitiveBudget(
            max_reasoning_depth=10,
            max_context_tokens=8192,
            abstraction_level=AbstractionLevel.HIGH,
            simulation_budget=10,
            external_allowed=False,
        ),
        "EVO-HYBRID": CognitiveBudget(
            max_reasoning_depth=10,
            max_context_tokens=8192,
            abstraction_level=AbstractionLevel.UNLIMITED,
            simulation_budget=10,
            external_allowed=True,
        ),
    }

    def __init__(self):
        """Initialize the budget manager."""
        self._current_budget: Optional[CognitiveBudget] = None
        self._lock = threading.Lock()

    def get_budget_for_profile(self, profile: str) -> CognitiveBudget:
        """Get a fresh budget for the given runtime profile."""
        template = self.PROFILE_BUDGETS.get(profile, self.PROFILE_BUDGETS["EVO-STANDARD"])
        # Create a new instance with template values
        return CognitiveBudget(
            max_reasoning_depth=template.max_reasoning_depth,
            max_context_tokens=template.max_context_tokens,
            abstraction_level=template.abstraction_level,
            simulation_budget=template.simulation_budget,
            external_allowed=template.external_allowed,
        )

    def set_budget(self, budget: CognitiveBudget):
        """Set the current budget."""
        with self._lock:
            self._current_budget = budget

    def get_current_budget(self) -> Optional[CognitiveBudget]:
        """Get the current budget."""
        return self._current_budget

    def require_budget(self) -> CognitiveBudget:
        """Get current budget or raise if none set."""
        budget = self.get_current_budget()
        if budget is None:
            raise RuntimeError("No cognitive budget set. Initialize budget before reasoning.")
        return budget

    def reset_current(self):
        """Reset the current budget."""
        with self._lock:
            if self._current_budget:
                self._current_budget.reset()

    def get_stats(self) -> Dict[str, Any]:
        """Get budget manager statistics."""
        budget = self._current_budget
        return {
            "has_active_budget": budget is not None,
            "current_budget": budget.to_dict() if budget else None,
            "available_profiles": list(self.PROFILE_BUDGETS.keys()),
        }


# Singleton instance
_budget_manager: Optional[CognitiveBudgetManager] = None


def get_budget_manager() -> CognitiveBudgetManager:
    """Get or create the budget manager singleton."""
    global _budget_manager
    if _budget_manager is None:
        _budget_manager = CognitiveBudgetManager()
    return _budget_manager
