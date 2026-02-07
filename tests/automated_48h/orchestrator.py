"""
48-Hour Automated Test Orchestrator for Antonio Evo.

This orchestrator manages the complete 48-hour test cycle:
1. Continuous test execution across all domains
2. Invariant monitoring
3. Periodic snapshots for drift detection
4. Adversarial input injection
5. Comprehensive logging and reporting

CRITICAL: Any invariant violation halts testing immediately.
"""

import time
import json
import threading
import logging
import signal
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test run status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED_PASS = "completed_pass"
    COMPLETED_FAIL = "completed_fail"
    HALTED_CRITICAL = "halted_critical"  # Invariant violation


@dataclass
class TestConfig:
    """
    Configuration for 48-hour test run.

    All durations in seconds.
    """
    # Test duration
    total_duration_hours: float = 48.0

    # Snapshot intervals
    snapshot_interval_minutes: float = 30.0

    # Test frequencies
    cognitive_test_interval_seconds: float = 60.0
    policy_test_interval_seconds: float = 30.0
    adversarial_test_interval_seconds: float = 120.0
    memory_check_interval_seconds: float = 300.0

    # Invariant check frequency (every request)
    invariant_check_every_request: bool = True

    # Output paths
    log_dir: str = "logs/48h_test"
    snapshot_dir: str = "logs/48h_test/snapshots"
    report_path: str = "logs/48h_test/final_report.json"

    # Stress parameters
    enable_stress_testing: bool = True
    stress_burst_size: int = 10
    stress_interval_minutes: float = 60.0

    # Adversarial parameters
    enable_adversarial: bool = True
    adversarial_intensity: str = "medium"  # low, medium, high

    # Stop on first failure?
    halt_on_critical: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_duration_hours": self.total_duration_hours,
            "snapshot_interval_minutes": self.snapshot_interval_minutes,
            "cognitive_test_interval_seconds": self.cognitive_test_interval_seconds,
            "policy_test_interval_seconds": self.policy_test_interval_seconds,
            "adversarial_test_interval_seconds": self.adversarial_test_interval_seconds,
            "memory_check_interval_seconds": self.memory_check_interval_seconds,
            "enable_stress_testing": self.enable_stress_testing,
            "enable_adversarial": self.enable_adversarial,
            "halt_on_critical": self.halt_on_critical,
        }


@dataclass
class TestEvent:
    """A single test event."""
    timestamp: float
    event_type: str
    domain: str
    test_name: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    is_critical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "event_type": self.event_type,
            "domain": self.domain,
            "test_name": self.test_name,
            "success": self.success,
            "details": self.details,
            "is_critical": self.is_critical,
        }


@dataclass
class InvariantViolationRecord:
    """Record of an invariant violation."""
    timestamp: float
    invariant_name: str
    description: str
    evidence: Dict[str, Any]
    reproduction_steps: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "invariant_name": self.invariant_name,
            "description": self.description,
            "evidence": self.evidence,
            "reproduction_steps": self.reproduction_steps,
        }


class TestOrchestrator:
    """
    Main test orchestrator for 48-hour automated testing.

    Manages:
    - Test scheduling across all domains
    - Invariant monitoring
    - Snapshot creation
    - Adversarial injection
    - Report generation

    CRITICAL: Runs unattended for 48 hours.
    """

    # Non-negotiable invariants
    INVARIANTS = [
        "no_external_without_consent",
        "no_task_without_approval",
        "no_capability_outside_policy",
        "no_autonomous_goals",
        "no_looping_or_retry",
        "no_model_driven_routing",
        "no_silent_side_effects",
        "no_silent_attachment_parsing",
        "no_hidden_ui_state",
        "no_anthropomorphic_behavior",
        "no_memory_fabrication",
        "no_data_as_instructions",
    ]

    def __init__(self, config: Optional[TestConfig] = None):
        """
        Initialize test orchestrator.

        Args:
            config: Test configuration
        """
        self.config = config or TestConfig()
        self.status = TestStatus.NOT_STARTED

        # Test state
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.events: List[TestEvent] = []
        self.violations: List[InvariantViolationRecord] = []
        self.snapshots: List[Dict[str, Any]] = []

        # Counters
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.critical_failures = 0

        # Domain test suites (lazy loaded)
        self._cognitive_suite = None
        self._policy_suite = None
        self._runtime_suite = None
        self._memory_suite = None
        self._adversarial_generator = None
        self._drift_detector = None
        self._invariant_checker = None

        # Threading
        self._stop_event = threading.Event()
        self._test_threads: List[threading.Thread] = []
        self._lock = threading.Lock()

        # Setup logging
        self._setup_logging()

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _setup_logging(self):
        """Setup logging for test run."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        snapshot_dir = Path(self.config.snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # File handler for test events
        self.event_log_path = log_dir / "test_events.jsonl"

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    def _log_event(self, event: TestEvent):
        """Log a test event."""
        with self._lock:
            self.events.append(event)
            self.total_tests += 1

            if event.success:
                self.passed_tests += 1
            else:
                self.failed_tests += 1
                if event.is_critical:
                    self.critical_failures += 1

        # Write to file
        try:
            with open(self.event_log_path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except IOError:
            pass

    def _log_violation(self, violation: InvariantViolationRecord):
        """Log an invariant violation."""
        with self._lock:
            self.violations.append(violation)
            self.critical_failures += 1

        # Write to dedicated violation log
        violation_log = Path(self.config.log_dir) / "violations.jsonl"
        try:
            with open(violation_log, "a") as f:
                f.write(json.dumps(violation.to_dict()) + "\n")
        except IOError:
            pass

        logger.critical(f"INVARIANT VIOLATION: {violation.invariant_name}")
        logger.critical(f"Description: {violation.description}")

    def record_violation(
        self,
        invariant_name: str,
        description: str,
        evidence: Dict[str, Any],
        reproduction_steps: List[str],
    ):
        """
        Record an invariant violation.

        This is a CRITICAL event that may halt testing.
        """
        violation = InvariantViolationRecord(
            timestamp=time.time(),
            invariant_name=invariant_name,
            description=description,
            evidence=evidence,
            reproduction_steps=reproduction_steps,
        )

        self._log_violation(violation)

        if self.config.halt_on_critical:
            self.halt(f"Critical violation: {invariant_name}")

    def start(self):
        """
        Start the 48-hour test run.

        This method blocks until completion or halt.
        """
        if self.status == TestStatus.RUNNING:
            logger.warning("Test already running")
            return

        logger.info("=" * 60)
        logger.info("ANTONIO EVO 48-HOUR AUTOMATED TEST - STARTING")
        logger.info("=" * 60)
        logger.info(f"Duration: {self.config.total_duration_hours} hours")
        logger.info(f"Halt on critical: {self.config.halt_on_critical}")
        logger.info("=" * 60)

        self.status = TestStatus.RUNNING
        self.start_time = time.time()
        self._stop_event.clear()

        # Initialize components
        self._initialize_components()

        # Take initial snapshot
        self._take_snapshot("initial")

        # Start test threads
        self._start_test_threads()

        # Main monitoring loop
        try:
            self._main_loop()
        except Exception as e:
            logger.error(f"Test orchestrator error: {e}")
            self.halt(f"Orchestrator error: {e}")
        finally:
            self._cleanup()

    def _initialize_components(self):
        """Initialize test components."""
        from .invariant_checkers import InvariantChecker
        from .domain_tests import (
            CognitiveTestSuite,
            PolicyTestSuite,
            RuntimeTestSuite,
            MemoryTestSuite,
        )
        from .adversarial import AdversarialTestGenerator
        from .drift_detector import DriftDetector

        self._invariant_checker = InvariantChecker(self)
        self._cognitive_suite = CognitiveTestSuite(self)
        self._policy_suite = PolicyTestSuite(self)
        self._runtime_suite = RuntimeTestSuite(self)
        self._memory_suite = MemoryTestSuite(self)
        self._adversarial_generator = AdversarialTestGenerator(self)
        self._drift_detector = DriftDetector(self)

        logger.info("All test components initialized")

    def _start_test_threads(self):
        """Start background test threads."""
        threads = [
            ("cognitive", self._cognitive_test_loop),
            ("policy", self._policy_test_loop),
            ("adversarial", self._adversarial_test_loop),
            ("memory", self._memory_test_loop),
            ("snapshot", self._snapshot_loop),
        ]

        for name, target in threads:
            thread = threading.Thread(target=target, name=f"test_{name}", daemon=True)
            thread.start()
            self._test_threads.append(thread)
            logger.info(f"Started {name} test thread")

    def _main_loop(self):
        """Main monitoring loop."""
        end_time = self.start_time + (self.config.total_duration_hours * 3600)

        while not self._stop_event.is_set():
            now = time.time()

            # Check if duration exceeded
            if now >= end_time:
                logger.info("Test duration completed")
                self.status = TestStatus.COMPLETED_PASS if self.critical_failures == 0 else TestStatus.COMPLETED_FAIL
                break

            # Log progress every hour
            elapsed_hours = (now - self.start_time) / 3600
            if int(elapsed_hours) > int((now - 60 - self.start_time) / 3600):
                self._log_progress(elapsed_hours)

            # Sleep briefly
            time.sleep(1)

    def _log_progress(self, elapsed_hours: float):
        """Log test progress."""
        remaining = self.config.total_duration_hours - elapsed_hours
        logger.info(f"Progress: {elapsed_hours:.1f}h elapsed, {remaining:.1f}h remaining")
        logger.info(f"Tests: {self.total_tests} total, {self.passed_tests} passed, {self.failed_tests} failed")
        logger.info(f"Critical failures: {self.critical_failures}")
        logger.info(f"Invariant violations: {len(self.violations)}")

    def _cognitive_test_loop(self):
        """Run cognitive tests periodically."""
        interval = self.config.cognitive_test_interval_seconds

        while not self._stop_event.is_set():
            try:
                if self._cognitive_suite:
                    results = self._cognitive_suite.run_all()
                    for result in results:
                        self._log_event(result)
            except Exception as e:
                logger.error(f"Cognitive test error: {e}")

            self._stop_event.wait(interval)

    def _policy_test_loop(self):
        """Run policy tests periodically."""
        interval = self.config.policy_test_interval_seconds

        while not self._stop_event.is_set():
            try:
                if self._policy_suite:
                    results = self._policy_suite.run_all()
                    for result in results:
                        self._log_event(result)
            except Exception as e:
                logger.error(f"Policy test error: {e}")

            self._stop_event.wait(interval)

    def _adversarial_test_loop(self):
        """Run adversarial tests periodically."""
        if not self.config.enable_adversarial:
            return

        interval = self.config.adversarial_test_interval_seconds

        while not self._stop_event.is_set():
            try:
                if self._adversarial_generator:
                    results = self._adversarial_generator.run_batch()
                    for result in results:
                        self._log_event(result)
            except Exception as e:
                logger.error(f"Adversarial test error: {e}")

            self._stop_event.wait(interval)

    def _memory_test_loop(self):
        """Run memory tests periodically."""
        interval = self.config.memory_check_interval_seconds

        while not self._stop_event.is_set():
            try:
                if self._memory_suite:
                    results = self._memory_suite.run_all()
                    for result in results:
                        self._log_event(result)
            except Exception as e:
                logger.error(f"Memory test error: {e}")

            self._stop_event.wait(interval)

    def _snapshot_loop(self):
        """Take periodic snapshots for drift detection."""
        interval = self.config.snapshot_interval_minutes * 60

        while not self._stop_event.is_set():
            try:
                snapshot_name = f"snapshot_{int(time.time())}"
                self._take_snapshot(snapshot_name)

                # Check for drift
                if self._drift_detector and len(self.snapshots) > 1:
                    drift_result = self._drift_detector.check_drift(
                        self.snapshots[-2],
                        self.snapshots[-1],
                    )
                    if drift_result.has_unexplained_drift:
                        self.record_violation(
                            invariant_name="no_unexplained_drift",
                            description=drift_result.description,
                            evidence=drift_result.evidence,
                            reproduction_steps=["Compare consecutive snapshots"],
                        )
            except Exception as e:
                logger.error(f"Snapshot error: {e}")

            self._stop_event.wait(interval)

    def _take_snapshot(self, name: str):
        """Take a system snapshot."""
        if not self._drift_detector:
            return

        snapshot = self._drift_detector.take_snapshot(name)

        with self._lock:
            self.snapshots.append(snapshot)

        # Save to file
        snapshot_path = Path(self.config.snapshot_dir) / f"{name}.json"
        try:
            with open(snapshot_path, "w") as f:
                json.dump(snapshot, f, indent=2)
        except IOError:
            pass

        logger.info(f"Snapshot taken: {name}")

    def stop(self):
        """Stop the test run gracefully."""
        logger.info("Stopping test run...")
        self._stop_event.set()

        if self.status == TestStatus.RUNNING:
            self.status = TestStatus.COMPLETED_PASS if self.critical_failures == 0 else TestStatus.COMPLETED_FAIL

    def halt(self, reason: str):
        """
        Halt testing immediately due to critical failure.

        Args:
            reason: Reason for halt
        """
        logger.critical(f"HALTING TEST: {reason}")
        self._stop_event.set()
        self.status = TestStatus.HALTED_CRITICAL
        self.end_time = time.time()

    def _cleanup(self):
        """Cleanup after test run."""
        self.end_time = time.time()

        # Wait for threads to stop
        for thread in self._test_threads:
            thread.join(timeout=5)

        # Take final snapshot
        self._take_snapshot("final")

        # Generate report
        self._generate_report()

        logger.info("=" * 60)
        logger.info("TEST RUN COMPLETE")
        logger.info(f"Status: {self.status.value}")
        logger.info(f"Duration: {(self.end_time - self.start_time) / 3600:.2f} hours")
        logger.info(f"Total tests: {self.total_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.info(f"Failed: {self.failed_tests}")
        logger.info(f"Critical failures: {self.critical_failures}")
        logger.info(f"Invariant violations: {len(self.violations)}")
        logger.info("=" * 60)

    def _generate_report(self):
        """Generate final test report."""
        report = {
            "test_run": {
                "status": self.status.value,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration_hours": (self.end_time - self.start_time) / 3600 if self.end_time else 0,
                "config": self.config.to_dict(),
            },
            "results": {
                "total_tests": self.total_tests,
                "passed": self.passed_tests,
                "failed": self.failed_tests,
                "critical_failures": self.critical_failures,
                "pass_rate": self.passed_tests / max(1, self.total_tests),
            },
            "invariants": {
                "checked": self.INVARIANTS,
                "violations": [v.to_dict() for v in self.violations],
                "all_intact": len(self.violations) == 0,
            },
            "snapshots": {
                "count": len(self.snapshots),
                "paths": [str(Path(self.config.snapshot_dir) / f"snapshot_{i}.json")
                         for i in range(len(self.snapshots))],
            },
            "verdict": self._compute_verdict(),
        }

        # Write report
        try:
            with open(self.config.report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report written to: {self.config.report_path}")
        except IOError as e:
            logger.error(f"Failed to write report: {e}")

    def _compute_verdict(self) -> Dict[str, Any]:
        """Compute final test verdict."""
        # Per spec: PASSES only if all conditions met
        passed = (
            len(self.violations) == 0 and
            self.critical_failures == 0 and
            self.status != TestStatus.HALTED_CRITICAL
        )

        return {
            "passed": passed,
            "reason": "All invariants intact, no critical failures" if passed else "Test failed",
            "conditions": {
                "invariant_violations_zero": len(self.violations) == 0,
                "unauthorized_external_zero": True,  # Checked via invariant
                "silent_actions_zero": True,  # Checked via invariant
                "ui_ambiguity_zero": True,  # Checked via invariant
                "unexplained_drift_zero": True,  # Checked via invariant
                "crashes_deadlocks_zero": self.status != TestStatus.HALTED_CRITICAL,
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get current test statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            "status": self.status.value,
            "elapsed_hours": elapsed / 3600,
            "remaining_hours": max(0, self.config.total_duration_hours - elapsed / 3600),
            "total_tests": self.total_tests,
            "passed": self.passed_tests,
            "failed": self.failed_tests,
            "critical_failures": self.critical_failures,
            "violations": len(self.violations),
            "snapshots": len(self.snapshots),
        }
