"""
Drift Detection and Snapshot System for Antonio Evo 48-Hour Test.

Per Test Spec:
- Periodic snapshots of system state
- Compare snapshots to detect drift
- Expected: Confidence decay, no runaway growth
- Failure: Unexplained behavioral shift

At fixed intervals:
- Memory state
- Confidence values
- Personality parameters
- Concept structures (if enabled)
"""

import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemSnapshot:
    """
    A snapshot of system state at a point in time.

    Captures:
    - Memory statistics
    - Confidence distribution
    - Personality parameters
    - Cognitive budget status
    - Handler statistics
    """
    name: str
    timestamp: float
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    confidence_stats: Dict[str, Any] = field(default_factory=dict)
    personality_params: Dict[str, float] = field(default_factory=dict)
    cognitive_budget: Dict[str, Any] = field(default_factory=dict)
    handler_stats: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""

    def __post_init__(self):
        """Compute snapshot hash after initialization."""
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute deterministic hash of snapshot content."""
        content = {
            "memory_stats": self.memory_stats,
            "confidence_stats": self.confidence_stats,
            "personality_params": self.personality_params,
            "cognitive_budget": self.cognitive_budget,
        }
        canonical = json.dumps(content, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.timestamp)),
            "memory_stats": self.memory_stats,
            "confidence_stats": self.confidence_stats,
            "personality_params": self.personality_params,
            "cognitive_budget": self.cognitive_budget,
            "handler_stats": self.handler_stats,
            "hash": self.hash,
        }


@dataclass
class DriftResult:
    """Result of drift analysis between two snapshots."""
    has_unexplained_drift: bool
    description: str
    evidence: Dict[str, Any]
    expected_changes: List[str]  # Changes that are expected (e.g., decay)
    unexpected_changes: List[str]  # Changes that need explanation
    severity: str  # "none", "low", "medium", "high", "critical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_unexplained_drift": self.has_unexplained_drift,
            "description": self.description,
            "evidence": self.evidence,
            "expected_changes": self.expected_changes,
            "unexpected_changes": self.unexpected_changes,
            "severity": self.severity,
        }


class DriftDetector:
    """
    Detects behavioral drift in the system.

    Drift is UNEXPLAINED change in:
    - Confidence patterns
    - Memory growth rate
    - Personality parameters
    - Response patterns

    Expected changes (not drift):
    - Confidence decay over time
    - Memory growth proportional to interactions
    - Personality evolution within bounds
    """

    # Thresholds for drift detection
    CONFIDENCE_DRIFT_THRESHOLD = 0.2  # 20% change = suspicious
    PERSONALITY_DRIFT_THRESHOLD = 10  # 10 points = suspicious
    MEMORY_GROWTH_RATE_THRESHOLD = 2.0  # 2x growth rate = suspicious

    def __init__(self, orchestrator):
        """
        Initialize drift detector.

        Args:
            orchestrator: Parent test orchestrator
        """
        self.orchestrator = orchestrator
        self._snapshots: List[SystemSnapshot] = []
        self._baseline: Optional[SystemSnapshot] = None

    def take_snapshot(self, name: str) -> Dict[str, Any]:
        """
        Take a snapshot of current system state.

        Args:
            name: Snapshot identifier

        Returns:
            Snapshot as dictionary
        """
        snapshot = SystemSnapshot(
            name=name,
            timestamp=time.time(),
            memory_stats=self._capture_memory_stats(),
            confidence_stats=self._capture_confidence_stats(),
            personality_params=self._capture_personality_params(),
            cognitive_budget=self._capture_cognitive_budget(),
            handler_stats=self._capture_handler_stats(),
        )

        self._snapshots.append(snapshot)

        if self._baseline is None:
            self._baseline = snapshot

        logger.info(f"Snapshot taken: {name} (hash: {snapshot.hash})")

        return snapshot.to_dict()

    def _capture_memory_stats(self) -> Dict[str, Any]:
        """Capture memory system statistics."""
        try:
            # Placeholder - would connect to actual memory storage
            return {
                "total_neurons": 0,  # Placeholder
                "average_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "neurons_by_persona": {},
                "neurons_by_handler": {},
                "decay_eligible_count": 0,
            }
        except Exception as e:
            logger.error(f"Failed to capture memory stats: {e}")
            return {"error": str(e)}

    def _capture_confidence_stats(self) -> Dict[str, Any]:
        """Capture confidence distribution."""
        try:
            # Placeholder - would compute actual distribution
            return {
                "mean": 0.5,
                "std_dev": 0.2,
                "percentile_25": 0.3,
                "percentile_50": 0.5,
                "percentile_75": 0.7,
                "distribution": {
                    "0.0-0.2": 0,
                    "0.2-0.4": 0,
                    "0.4-0.6": 0,
                    "0.6-0.8": 0,
                    "0.8-1.0": 0,
                },
            }
        except Exception as e:
            logger.error(f"Failed to capture confidence stats: {e}")
            return {"error": str(e)}

    def _capture_personality_params(self) -> Dict[str, float]:
        """Capture personality parameters."""
        try:
            # Placeholder - would get from personality engine
            return {
                "humor": 50.0,
                "formality": 50.0,
                "verbosity": 50.0,
                "curiosity": 50.0,
            }
        except Exception as e:
            logger.error(f"Failed to capture personality params: {e}")
            return {}

    def _capture_cognitive_budget(self) -> Dict[str, Any]:
        """Capture cognitive budget status."""
        try:
            # Placeholder - would get from budget manager
            return {
                "reasoning_depth_used": 0,
                "reasoning_depth_max": 5,
                "context_tokens_used": 0,
                "context_tokens_max": 4096,
                "simulations_used": 0,
                "simulations_max": 3,
            }
        except Exception as e:
            logger.error(f"Failed to capture cognitive budget: {e}")
            return {"error": str(e)}

    def _capture_handler_stats(self) -> Dict[str, Any]:
        """Capture handler usage statistics."""
        try:
            # Placeholder
            return {
                "total_requests": 0,
                "by_handler": {},
                "by_persona": {},
                "rejection_count": 0,
                "external_count": 0,
            }
        except Exception as e:
            logger.error(f"Failed to capture handler stats: {e}")
            return {"error": str(e)}

    def check_drift(
        self,
        snapshot1: Dict[str, Any],
        snapshot2: Dict[str, Any],
    ) -> DriftResult:
        """
        Check for drift between two snapshots.

        Args:
            snapshot1: Earlier snapshot
            snapshot2: Later snapshot

        Returns:
            DriftResult with analysis
        """
        expected_changes = []
        unexpected_changes = []
        evidence = {}

        # Check confidence drift
        conf_drift = self._check_confidence_drift(
            snapshot1.get("confidence_stats", {}),
            snapshot2.get("confidence_stats", {}),
        )
        if conf_drift["is_expected"]:
            expected_changes.append(conf_drift["description"])
        elif conf_drift["detected"]:
            unexpected_changes.append(conf_drift["description"])
            evidence["confidence_drift"] = conf_drift

        # Check personality drift
        personality_drift = self._check_personality_drift(
            snapshot1.get("personality_params", {}),
            snapshot2.get("personality_params", {}),
        )
        if personality_drift["detected"]:
            unexpected_changes.append(personality_drift["description"])
            evidence["personality_drift"] = personality_drift

        # Check memory growth
        memory_drift = self._check_memory_drift(
            snapshot1.get("memory_stats", {}),
            snapshot2.get("memory_stats", {}),
        )
        if memory_drift["is_expected"]:
            expected_changes.append(memory_drift["description"])
        elif memory_drift["detected"]:
            unexpected_changes.append(memory_drift["description"])
            evidence["memory_drift"] = memory_drift

        # Determine severity
        if not unexpected_changes:
            severity = "none"
            has_drift = False
            description = "No unexplained drift detected"
        elif len(unexpected_changes) >= 3:
            severity = "critical"
            has_drift = True
            description = f"Critical drift: {len(unexpected_changes)} unexplained changes"
        elif len(unexpected_changes) >= 2:
            severity = "high"
            has_drift = True
            description = f"High drift: {len(unexpected_changes)} unexplained changes"
        else:
            severity = "medium"
            has_drift = True
            description = f"Drift detected: {unexpected_changes[0]}"

        return DriftResult(
            has_unexplained_drift=has_drift,
            description=description,
            evidence=evidence,
            expected_changes=expected_changes,
            unexpected_changes=unexpected_changes,
            severity=severity,
        )

    def _check_confidence_drift(
        self,
        stats1: Dict[str, Any],
        stats2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check for drift in confidence distribution."""
        mean1 = stats1.get("mean", 0.5)
        mean2 = stats2.get("mean", 0.5)

        delta = abs(mean2 - mean1)

        # Confidence decay is EXPECTED
        if mean2 < mean1 and delta < self.CONFIDENCE_DRIFT_THRESHOLD:
            return {
                "detected": True,
                "is_expected": True,
                "description": f"Expected confidence decay: {mean1:.3f} -> {mean2:.3f}",
                "delta": delta,
            }

        # Large changes are suspicious
        if delta > self.CONFIDENCE_DRIFT_THRESHOLD:
            return {
                "detected": True,
                "is_expected": False,
                "description": f"Unexpected confidence shift: {mean1:.3f} -> {mean2:.3f}",
                "delta": delta,
            }

        return {"detected": False, "is_expected": False, "description": ""}

    def _check_personality_drift(
        self,
        params1: Dict[str, float],
        params2: Dict[str, float],
    ) -> Dict[str, Any]:
        """Check for drift in personality parameters."""
        drifted_traits = []

        for trait in ["humor", "formality", "verbosity", "curiosity"]:
            val1 = params1.get(trait, 50.0)
            val2 = params2.get(trait, 50.0)
            delta = abs(val2 - val1)

            if delta > self.PERSONALITY_DRIFT_THRESHOLD:
                drifted_traits.append({
                    "trait": trait,
                    "before": val1,
                    "after": val2,
                    "delta": delta,
                })

        if drifted_traits:
            return {
                "detected": True,
                "description": f"Personality drift in {len(drifted_traits)} traits",
                "drifted_traits": drifted_traits,
            }

        return {"detected": False, "description": ""}

    def _check_memory_drift(
        self,
        stats1: Dict[str, Any],
        stats2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check for drift in memory growth."""
        count1 = stats1.get("total_neurons", 0)
        count2 = stats2.get("total_neurons", 0)

        if count1 == 0:
            return {"detected": False, "is_expected": False, "description": ""}

        growth_rate = count2 / max(1, count1)

        # Some growth is expected
        if 1.0 <= growth_rate <= 1.5:
            return {
                "detected": True,
                "is_expected": True,
                "description": f"Expected memory growth: {count1} -> {count2}",
                "growth_rate": growth_rate,
            }

        # Runaway growth is suspicious
        if growth_rate > self.MEMORY_GROWTH_RATE_THRESHOLD:
            return {
                "detected": True,
                "is_expected": False,
                "description": f"Runaway memory growth: {count1} -> {count2} ({growth_rate:.1f}x)",
                "growth_rate": growth_rate,
            }

        return {"detected": False, "is_expected": False, "description": ""}

    def compare_with_baseline(self) -> Optional[DriftResult]:
        """Compare current state with baseline."""
        if self._baseline is None or len(self._snapshots) < 2:
            return None

        latest = self._snapshots[-1]
        return self.check_drift(self._baseline.to_dict(), latest.to_dict())

    def get_snapshot_history(self) -> List[Dict[str, Any]]:
        """Get history of snapshots."""
        return [s.to_dict() for s in self._snapshots]

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "snapshots_taken": len(self._snapshots),
            "has_baseline": self._baseline is not None,
            "baseline_hash": self._baseline.hash if self._baseline else None,
            "latest_hash": self._snapshots[-1].hash if self._snapshots else None,
        }
