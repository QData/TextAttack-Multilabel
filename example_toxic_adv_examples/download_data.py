#!/usr/bin/env python3
"""
Secure Python-based data download script using environment variables for Kaggle API.
"""

import os
import sys
import subprocess


def check_kaggle_cli():
    """Check if kaggle CLI is installed, install if not."""
    try:
        subprocess.run(['kaggle', '--version'], check=True, capture_output=True)
        print("Kaggle CLI is available.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Kaggle CLI not found, installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle'], check=True)
        print("Kaggle CLI installed.")


def setup_kaggle_api():
    """Setup Kaggle API key from environment variables instead of file."""
    username = os.getenv('KAGGLE_USERNAME')
    key = os.getenv('KAGGLE_KEY')

    if not username or not key:
        print("Error: KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set.")
        print("Set them with:")
        print("  export KAGGLE_USERNAME='your_username'")
        print("  export KAGGLE_KEY='your_api_key'")
        print("\nOr get your API key from: https://kaggle.com → Account → Create New API Token")
        sys.exit(1)

    # Create Kaggle directory if it doesn't exist
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)

    # Write API key to file securely (temporary, could be improved with keyring)
    api_key_path = os.path.join(kaggle_dir, 'kaggle.json')
    with open(api_key_path, 'w') as f:
        f.write(f'{{"username":"{username}","key":"{key}"}}')

    # Set appropriate permissions
    os.chmod(api_key_path, 0o600)
    print(f"Kaggle API key written to {api_key_path}")


def download_dataset():
    """Download Jigsaw Toxic Comments dataset."""
    print("Downloading Jigsaw Toxic Comments dataset...")
    try:
        subprocess.run([
            'kaggle', 'datasets', 'download',
            'julian3833/jigsaw-toxic-comment-classification-challenge',
            '-p', 'data'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset: {e}")
        sys.exit(1)


def extract_files():
    """Extract downloaded zip files."""
    import zipfile

    # Create data directory
    os.makedirs('data', exist_ok=True)
    # Create subdirectory for jigsaw
    jigsaw_dir = os.path.join('data', 'jigsaw_toxic_comments')
    os.makedirs(jigsaw_dir, exist_ok=True)

    # Find and extract zip files
    for file in os.listdir('data'):
        if file.endswith('.zip'):
            zip_path = os.path.join('data', file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(jigsaw_dir)
            print(f"Extracted {file}")


def main():
    """Main download process."""
    print("Checking Kaggle CLI...")
    check_kaggle_cli()

    print("Setting up Kaggle API...")
    setup_kaggle_api()

    print("Downloading dataset...")
    download_dataset()

    print("Extracting files...")
    extract_files()

    print("Data download complete!")


if __name__ == "__main__":
    main()
