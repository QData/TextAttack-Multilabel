#!/usr/bin/env python3
"""
Test runner script for TextAttack-Multilabel with comprehensive options.
"""

import argparse
import os
import sys
import subprocess


def run_command(cmd, check=True):
    """Run shell command."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.getcwd())
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result.returncode == 0


def run_pytest(args):
    """Run pytest with specified arguments."""
    cmd = [sys.executable, "-m", "pytest"]

    if args.verbose:
        cmd.append("-v")

    if args.coverage:
        cmd.extend(["--cov=textattack_multilabel", "--cov-report=term-missing", "--cov-report=html"])

    if args.integration_only:
        cmd.append("-k integration")
    elif args.unit_only:
        cmd.append("-k not integration")

    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])

    cmd.append("test/")

    run_command(cmd, check=True)


def run_quality_checks():
    """Run code quality checks."""
    print("\nRunning code quality checks...")

    # Run black check
    run_command(["black", "--check", "."], check=False)

    # Run isort check
    run_command(["isort", "--check-only", "."], check=False)

    # Run mypy if available
    run_command(["mypy", "textattack_multilabel/"], check=False)


def run_performance_benchmark(num_samples=50):
    """Run simple performance benchmark."""
    print(f"\nRunning performance benchmark with {num_samples} samples...")

    # This would be a full script, but for now just note
    print("Performance benchmark would run full attack pipeline here")
    print(f"With {num_samples} samples")


def main():
    parser = argparse.ArgumentParser(description="Run TextAttack-Multilabel tests")
    parser.add_argument(
        "--unit-only", action="store_true",
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration-only", action="store_true",
        help="Run only integration tests"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--parallel", type=int, metavar="N",
        help="Run tests in parallel with N workers"
    )
    parser.add_argument(
        "--quality", action="store_true",
        help="Run code quality checks"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--num-samples", type=int, default=50,
        help="Number of samples for benchmark"
    )

    args = parser.parse_args()

    if args.quality:
        run_quality_checks()
    elif args.benchmark:
        run_performance_benchmark(args.num_samples)
    else:
        run_pytest(args)

    print("\nTest run complete!")


if __name__ == "__main__":
    main()
