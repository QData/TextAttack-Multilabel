#!/usr/bin/env python3
"""
Python-based environment setup to replace shell script for better security and control.
"""

import subprocess
import sys
import os


def run_command(cmd, validate=True):
    """Run a command with optional validation."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        print(f"Error output: {e.stderr}")
        if validate:
            raise
        return False


def main():
    """Set up conda environment and install packages."""

    # Validate conda is available
    if not run_command(["conda", "--version"], validate=False):
        print("Conda not found. Please install Miniconda or Anaconda first.")
        sys.exit(1)

    # Create environment
    run_command(["conda", "create", "-n", "textattackenv", "python=3.8", "-y"])

    # Activate environment and install packages
    # Note: Can't use conda activate in script, so use conda run
    run_command([
        "conda", "run", "-n", "textattackenv", "pip", "install", "detoxify", "kaggle"
    ])
    run_command([
        "conda", "run", "-n", "textattackenv", "pip", "install", "textattack[tensorflow,optional]"
    ])
    run_command([
        "conda", "run", "-n", "textattackenv", "pip", "install", "-U", "sentence-transformers"
    ])

    print("\nEnvironment setup complete!")
    print("Activate with: conda activate textattackenv")


if __name__ == "__main__":
    main()
