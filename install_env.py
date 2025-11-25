#!/usr/bin/env python3
"""
Enhanced Python-based environment setup with version constraints and cross-platform support.
"""

import subprocess
import sys
import os
import platform


def run_command(cmd, validate=True, env=None):
    """Run a command with optional validation."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        print(f"Error output: {e.stderr}")
        if validate:
            raise
        return False


def validate_system():
    """Validate system requirements."""
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required.")
        sys.exit(1)

    # Check CUDA availability for GPU support
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        print("PyTorch not yet installed - will be installed later")


def setup_conda():
    """Set up conda environment."""
    if not run_command(["conda", "--version"], validate=False):
        print("Conda not found. Installing Miniconda...")
        system = platform.system().lower()
        if system == "linux":
            conda_installer = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        elif system == "darwin":  # macOS
            conda_installer = "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        elif system == "windows":
            print("Windows support is experimental. Please use WSL or virtual environment.")
            conda_installer = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        else:
            print(f"Unsupported platform: {system}")
            sys.exit(1)

        run_command(["curl", "-fsSL", conda_installer, "-o", "miniconda.sh"])
        run_command(["bash", "miniconda.sh", "-b", "-p", "$HOME/miniconda"])
        os.environ["PATH"] = f"$HOME/miniconda/bin:{os.environ.get('PATH', '')}"

    return True


def install_packages(env_name):
    """Install required packages with version constraints."""
    print(f"Installing packages in {env_name}...")

    run_command([
        "conda", "run", "-n", env_name, "pip", "install",
        "torch>=1.9.0",
        "torchvision",
        "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])

    run_command([
        "conda", "run", "-n", env_name, "pip", "install",
        "textattack>=0.3.0"
    ])

    run_command([
        "conda", "run", "-n", env_name, "pip", "install",
        "transformers>=4.10.0",
        "datasets>=2.0.0",
        "detoxify>=0.5.0",
        "kaggle>=1.5.12",
        "sentence-transformers>=2.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0"
    ])


def verify_installation(env_name):
    """Verify installation by running basic checks."""
    print("Verifying installation...")

    try:
        # Test imports
        test_commands = [
            ["conda", "run", "-n", env_name, "python", "-c", "import textattack; print('TextAttack OK')"],
            ["conda", "run", "-n", env_name, "python", "-c", "import textattack_multilabel; print('Multilabel extension OK')"],
            ["conda", "run", "-n", env_name, "python", "-c", "import detoxify; print('Detoxify OK')"],
        ]
        for cmd in test_commands:
            run_command(cmd)

        print("✓ All packages installed and importable")
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    """Main setup process."""
    print("TextAttack-Multilabel Environment Setup")
    print("=" * 50)

    validate_system()

    env_name = "textattack-multilabel"

    if setup_conda():
        print(f"Creating conda environment '{env_name}' with Python ..")
        run_command([
            "conda", "create", "-n", env_name,
            f"python={'.'.join(map(str, sys.version_info[:2]))}", "-y"
        ])

        install_packages(env_name)

        if verify_installation(env_name):
            print("\n🚀 Environment setup complete!")
            print(f"Activate with: conda activate {env_name}")
            print(f"Run tests with: conda run -n {env_name} pytest test/")
            print(f"Generate attacks: conda run -n {env_name} python scripts/attack_multilabel.py [args]")
        else:
            print("\n⚠️  Setup completed but verification failed. Check the output above.")


if __name__ == "__main__":
    main()
