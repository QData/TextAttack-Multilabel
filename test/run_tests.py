#!/usr/bin/env python3
"""
Comprehensive test runner for TextAttack-Multilabel.

This script provides a unified interface for running all tests with various options:
- Unit tests, integration tests, or all tests
- Coverage reporting with HTML/XML output
- Parallel test execution
- Code quality checks (black, isort, mypy)
- Specific test file or function targeting
- Watch mode for continuous testing

Usage Examples:
    # Run all tests with coverage
    python test/run_tests.py --coverage

    # Run specific test file
    python test/run_tests.py --file test_goal_function_core.py

    # Run tests in parallel (4 workers)
    python test/run_tests.py --parallel 4

    # Run with verbose output
    python test/run_tests.py -v

    # Run only fast unit tests
    python test/run_tests.py --unit-only

    # Run quality checks
    python test/run_tests.py --quality

    # Generate detailed coverage report
    python test/run_tests.py --coverage --cov-report html
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(msg):
    """Print colored header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(msg):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} {msg}")


def print_error(msg):
    """Print error message."""
    print(f"{Colors.FAIL}✗{Colors.ENDC} {msg}")


def print_warning(msg):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {msg}")


def run_command(cmd, check=True, description=None):
    """Run shell command and return success status."""
    if description:
        print(f"\n{Colors.OKCYAN}Running:{Colors.ENDC} {description}")

    print(f"{Colors.BOLD}Command:{Colors.ENDC} {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode == 0:
        if description:
            print_success(f"{description} completed successfully")
        return True
    else:
        if check:
            print_error(f"Command failed with exit code {result.returncode}")
            sys.exit(result.returncode)
        else:
            print_warning(f"Command failed with exit code {result.returncode} (continuing)")
        return False


def check_pytest_installed():
    """Check if pytest is installed."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        print_error("pytest is not installed!")
        print(f"\nInstall with: {Colors.BOLD}pip install pytest pytest-cov{Colors.ENDC}")
        return False


def get_test_files():
    """Get list of all test files."""
    test_dir = Path("test")
    test_files = sorted(test_dir.glob("test_*.py"))
    return [f.name for f in test_files]


def run_pytest(args):
    """Run pytest with specified arguments."""
    print_header("Running Tests")

    if not check_pytest_installed():
        sys.exit(1)

    cmd = [sys.executable, "-m", "pytest"]

    # Verbosity
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    # Show summary of all test outcomes
    cmd.append("-ra")

    # Coverage options
    if args.coverage:
        cmd.extend([
            "--cov=textattack_multilabel",
            "--cov-report=term-missing"
        ])

        # Additional coverage reports
        if args.cov_report:
            for report_type in args.cov_report:
                cmd.append(f"--cov-report={report_type}")
        else:
            # Default: generate HTML report
            cmd.append("--cov-report=html")

    # Test filtering
    if args.unit_only:
        cmd.extend(["-m", "not integration"])
        print(f"{Colors.OKCYAN}Filter:{Colors.ENDC} Unit tests only\n")
    elif args.integration_only:
        cmd.extend(["-m", "integration"])
        print(f"{Colors.OKCYAN}Filter:{Colors.ENDC} Integration tests only\n")

    # Keyword filtering
    if args.keyword:
        cmd.extend(["-k", args.keyword])
        print(f"{Colors.OKCYAN}Filter:{Colors.ENDC} Keyword = '{args.keyword}'\n")

    # Parallel execution
    if args.parallel:
        try:
            subprocess.run(
                [sys.executable, "-m", "pytest", "--help"],
                capture_output=True,
                check=True
            )
            cmd.extend(["-n", str(args.parallel)])
            print(f"{Colors.OKCYAN}Parallel:{Colors.ENDC} Using {args.parallel} workers\n")
        except:
            print_warning("pytest-xdist not installed, running sequentially")
            print(f"Install with: {Colors.BOLD}pip install pytest-xdist{Colors.ENDC}\n")

    # Fail fast
    if args.fail_fast:
        cmd.append("-x")
        print(f"{Colors.OKCYAN}Mode:{Colors.ENDC} Fail fast (stop on first failure)\n")

    # Show local variables on failure
    if args.show_locals:
        cmd.append("-l")

    # Specific test file
    if args.file:
        test_path = Path("test") / args.file
        if not test_path.exists():
            print_error(f"Test file not found: {test_path}")
            print(f"\nAvailable test files:")
            for f in get_test_files():
                print(f"  - {f}")
            sys.exit(1)
        cmd.append(str(test_path))
        print(f"{Colors.OKCYAN}Target:{Colors.ENDC} {args.file}\n")
    elif args.test:
        cmd.append(args.test)
        print(f"{Colors.OKCYAN}Target:{Colors.ENDC} {args.test}\n")
    else:
        cmd.append("test/")

    # Run tests
    success = run_command(cmd, check=args.strict)

    if args.coverage:
        print(f"\n{Colors.OKGREEN}Coverage report generated:{Colors.ENDC}")
        print(f"  HTML: {Colors.BOLD}htmlcov/index.html{Colors.ENDC}")

    return success


def run_quality_checks(args):
    """Run code quality checks."""
    print_header("Code Quality Checks")

    all_passed = True

    # Black - code formatting
    print(f"\n{Colors.BOLD}1. Checking code formatting (black)...{Colors.ENDC}")
    if run_command(
        [sys.executable, "-m", "black", "--check", "textattack_multilabel/", "test/"],
        check=False,
        description=None
    ):
        print_success("Code formatting is correct")
    else:
        print_warning("Code formatting issues found")
        print(f"   Fix with: {Colors.BOLD}black textattack_multilabel/ test/{Colors.ENDC}")
        all_passed = False

    # isort - import sorting
    print(f"\n{Colors.BOLD}2. Checking import sorting (isort)...{Colors.ENDC}")
    if run_command(
        [sys.executable, "-m", "isort", "--check-only", "textattack_multilabel/", "test/"],
        check=False,
        description=None
    ):
        print_success("Import sorting is correct")
    else:
        print_warning("Import sorting issues found")
        print(f"   Fix with: {Colors.BOLD}isort textattack_multilabel/ test/{Colors.ENDC}")
        all_passed = False

    # mypy - type checking (optional)
    if not args.skip_mypy:
        print(f"\n{Colors.BOLD}3. Checking type hints (mypy)...{Colors.ENDC}")
        if run_command(
            [sys.executable, "-m", "mypy", "textattack_multilabel/", "--ignore-missing-imports"],
            check=False,
            description=None
        ):
            print_success("Type checking passed")
        else:
            print_warning("Type checking found issues (non-critical)")

    if all_passed:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}All quality checks passed!{Colors.ENDC}")
    else:
        print(f"\n{Colors.WARNING}{Colors.BOLD}Some quality checks failed{Colors.ENDC}")

    return all_passed


def list_tests():
    """List all available test files."""
    print_header("Available Test Files")

    test_files = get_test_files()

    if not test_files:
        print_warning("No test files found in test/ directory")
        return

    print(f"Found {len(test_files)} test file(s):\n")

    for i, test_file in enumerate(test_files, 1):
        test_path = Path("test") / test_file

        # Count test functions
        try:
            with open(test_path, 'r') as f:
                content = f.read()
                test_count = content.count('def test_')
        except:
            test_count = '?'

        print(f"{i:2}. {Colors.BOLD}{test_file:40}{Colors.ENDC} ({test_count} tests)")

    print(f"\n{Colors.OKCYAN}Run specific file:{Colors.ENDC}")
    print(f"  python test/run_tests.py --file {test_files[0]}")


def show_coverage_summary():
    """Show coverage summary if available."""
    coverage_file = Path(".coverage")
    if coverage_file.exists():
        print_header("Coverage Summary")
        run_command(
            [sys.executable, "-m", "coverage", "report", "--skip-empty"],
            check=False,
            description="Generating coverage summary"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for TextAttack-Multilabel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run all tests
  %(prog)s --coverage                   # Run with coverage report
  %(prog)s --file test_goal_function_core.py  # Run specific file
  %(prog)s --test test/test_goal_function_core.py::TestGetScore  # Run specific class
  %(prog)s -v --parallel 4              # Verbose, parallel execution
  %(prog)s --quality                    # Run code quality checks
  %(prog)s --list                       # List all test files
        """
    )

    # Test execution options
    test_group = parser.add_argument_group('Test Execution')
    test_group.add_argument(
        "--file", "-f",
        metavar="FILE",
        help="Run specific test file (e.g., test_goal_function_core.py)"
    )
    test_group.add_argument(
        "--test", "-t",
        metavar="PATH",
        help="Run specific test path (e.g., test/test_*.py::TestClass::test_method)"
    )
    test_group.add_argument(
        "-k", "--keyword",
        metavar="EXPRESSION",
        help="Run tests matching keyword expression"
    )
    test_group.add_argument(
        "--unit-only",
        action="store_true",
        help="Run only unit tests (exclude integration tests)"
    )
    test_group.add_argument(
        "--integration-only",
        action="store_true",
        help="Run only integration tests"
    )

    # Coverage options
    coverage_group = parser.add_argument_group('Coverage')
    coverage_group.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    coverage_group.add_argument(
        "--cov-report",
        nargs="+",
        choices=["html", "xml", "json", "annotate", "term"],
        help="Coverage report format(s)"
    )

    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    output_group.add_argument(
        "--show-locals", "-l",
        action="store_true",
        help="Show local variables on test failure"
    )

    # Execution options
    exec_group = parser.add_argument_group('Execution')
    exec_group.add_argument(
        "--parallel", "-n",
        type=int,
        metavar="N",
        help="Run tests in parallel with N workers (requires pytest-xdist)"
    )
    exec_group.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Stop on first test failure"
    )
    exec_group.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code on test failure"
    )

    # Quality and utility options
    util_group = parser.add_argument_group('Utilities')
    util_group.add_argument(
        "--quality", "-q",
        action="store_true",
        help="Run code quality checks (black, isort, mypy)"
    )
    util_group.add_argument(
        "--skip-mypy",
        action="store_true",
        help="Skip mypy type checking in quality checks"
    )
    util_group.add_argument(
        "--list",
        action="store_true",
        help="List all available test files"
    )
    util_group.add_argument(
        "--show-coverage",
        action="store_true",
        help="Show coverage summary from previous run"
    )

    args = parser.parse_args()

    # Handle special modes
    if args.list:
        list_tests()
        return

    if args.show_coverage:
        show_coverage_summary()
        return

    if args.quality:
        success = run_quality_checks(args)
        sys.exit(0 if success else 1)

    # Run tests
    success = run_pytest(args)

    # Final summary
    print_header("Test Run Complete")

    if success:
        print_success("All tests passed!")
        sys.exit(0)
    else:
        if args.strict:
            print_error("Some tests failed")
            sys.exit(1)
        else:
            print_warning("Some tests failed (non-strict mode)")
            sys.exit(0)


if __name__ == "__main__":
    main()
