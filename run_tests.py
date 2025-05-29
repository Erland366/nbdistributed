#!/usr/bin/env python3
"""
Test runner for jupyter_distributed package

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py unit         # Run unit tests only
    python run_tests.py integration  # Run integration tests only
    python run_tests.py slow         # Run slow tests
    python run_tests.py coverage     # Run with coverage
"""

import sys
import subprocess
import os
from pathlib import Path


def run_command(cmd):
    """Run a shell command and return the exit code"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    """Main test runner"""

    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)

    # Parse command line arguments
    if len(sys.argv) < 2:
        test_type = "all"
    else:
        test_type = sys.argv[1].lower()

    # Base pytest command
    base_cmd = ["python", "-m", "pytest", "tests/"]

    if test_type == "unit":
        cmd = base_cmd + ["-m", "not slow and not integration"]
    elif test_type == "integration":
        cmd = base_cmd + ["-m", "integration"]
    elif test_type == "slow":
        cmd = base_cmd + ["-m", "slow"]
    elif test_type == "coverage":
        cmd = base_cmd + [
            "--cov=src/jupyter_distributed",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]
    elif test_type == "all":
        cmd = base_cmd
    else:
        print(f"Unknown test type: {test_type}")
        print(__doc__)
        return 1

    # Run the tests
    exit_code = run_command(cmd)

    if exit_code == 0:
        print(f"\n✅ {test_type.capitalize()} tests passed!")
    else:
        print(f"\n❌ {test_type.capitalize()} tests failed!")

    return exit_code


if __name__ == "__main__":
    exit(main())
