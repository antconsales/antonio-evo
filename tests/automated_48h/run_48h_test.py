#!/usr/bin/env python3
"""
Antonio Evo - 48-Hour Automated System Test Runner.

Usage:
    python -m tests.automated_48h.run_48h_test [OPTIONS]

Options:
    --duration HOURS    Test duration in hours (default: 48)
    --intensity LEVEL   Adversarial intensity: low, medium, high (default: medium)
    --halt-on-critical  Halt on first critical failure (default: True)
    --log-dir PATH      Log directory (default: logs/48h_test)
    --dry-run           Run for 1 minute to verify setup

This script runs the complete 48-hour automated test suite.
The system must survive:
- Time
- Stress
- Ambiguity
- Adversarial inputs
- Resource degradation

PASS CRITERIA:
- invariant violations = 0
- unauthorized external calls = 0
- silent actions = 0
- unexplained drift = 0
- crashes or deadlocks = 0

A system that survives this test is not a demo. It is engineered.
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("48h_test")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Antonio Evo 48-Hour Automated System Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=48.0,
        help="Test duration in hours (default: 48)",
    )

    parser.add_argument(
        "--intensity",
        choices=["low", "medium", "high"],
        default="medium",
        help="Adversarial test intensity (default: medium)",
    )

    parser.add_argument(
        "--halt-on-critical",
        action="store_true",
        default=True,
        help="Halt on first critical failure (default: True)",
    )

    parser.add_argument(
        "--no-halt-on-critical",
        action="store_false",
        dest="halt_on_critical",
        help="Continue testing even after critical failures",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/48h_test",
        help="Log directory (default: logs/48h_test)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run for 1 minute to verify setup",
    )

    parser.add_argument(
        "--snapshot-interval",
        type=float,
        default=30.0,
        help="Snapshot interval in minutes (default: 30)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Banner
    print("=" * 70)
    print("ANTONIO EVO - 48-HOUR AUTOMATED SYSTEM TEST")
    print("=" * 70)
    print()
    print(f"Start Time: {datetime.now().isoformat()}")
    print(f"Duration: {args.duration} hours")
    print(f"Intensity: {args.intensity}")
    print(f"Halt on Critical: {args.halt_on_critical}")
    print(f"Log Directory: {args.log_dir}")
    print()

    if args.dry_run:
        print("DRY RUN MODE: Running for 1 minute only")
        args.duration = 1 / 60  # 1 minute

    # Confirm before starting
    if not args.dry_run:
        print("This test will run for {:.1f} hours unattended.".format(args.duration))
        print("Press Ctrl+C at any time to stop gracefully.")
        print()
        try:
            input("Press Enter to start, or Ctrl+C to cancel... ")
        except KeyboardInterrupt:
            print("\nTest cancelled.")
            sys.exit(0)

    print()
    print("Starting test...")
    print("-" * 70)

    try:
        # Import here to avoid import errors if dependencies missing
        from .orchestrator import TestOrchestrator, TestConfig

        # Create configuration
        config = TestConfig(
            total_duration_hours=args.duration,
            snapshot_interval_minutes=args.snapshot_interval,
            log_dir=args.log_dir,
            snapshot_dir=f"{args.log_dir}/snapshots",
            report_path=f"{args.log_dir}/final_report.json",
            enable_adversarial=True,
            adversarial_intensity=args.intensity,
            halt_on_critical=args.halt_on_critical,
        )

        # Create and run orchestrator
        orchestrator = TestOrchestrator(config)
        orchestrator.start()

        # Print final stats
        stats = orchestrator.get_stats()
        print()
        print("-" * 70)
        print("FINAL STATISTICS")
        print("-" * 70)
        print(f"Status: {stats['status']}")
        print(f"Duration: {stats['elapsed_hours']:.2f} hours")
        print(f"Total Tests: {stats['total_tests']}")
        print(f"Passed: {stats['passed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Critical Failures: {stats['critical_failures']}")
        print(f"Violations: {stats['violations']}")
        print(f"Snapshots: {stats['snapshots']}")
        print()

        # Exit code based on result
        if stats['violations'] > 0 or stats['critical_failures'] > 0:
            print("TEST RESULT: FAIL")
            sys.exit(1)
        else:
            print("TEST RESULT: PASS")
            sys.exit(0)

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Ensure all test modules are available.")
        sys.exit(2)

    except KeyboardInterrupt:
        print("\n")
        print("Test interrupted by user.")
        print("Generating partial report...")
        sys.exit(130)  # Standard exit code for Ctrl+C

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
