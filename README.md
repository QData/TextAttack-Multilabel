# TextAttack-Multilabel

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/textattack-multilabel.svg)](https://pypi.org/project/textattack-multilabel/)

A professional extension of [TextAttack](https://github.com/QData/TextAttack) for multi-label adversarial example generation, with focus on toxicity classification. Generate adversarial examples that flip multiple labels simultaneously while preserving semantic meaning and grammatical correctness.

## Features

- 🏗️ **Modular Architecture**: Support for multiple models (Detoxify, custom HF models) and datasets
- ⚙️ **CLI Interface**: User-friendly command-line tools for all major operations
- 📊 **Configuration-Driven**: YAML configuration for flexible attack parameters
- 🔬 **Professional Testing**: Comprehensive test suite with coverage reporting
- 📚 **Rich Examples**: Jupyter notebooks and tutorials
- 🚀 **Easy Installation**: Pip-installable with proper dependency management

## Installation

### Quick Install (Recommended)

```bash

# install from source
git clone https://github.com/QData/TextAttack-Multilabel
cd TextAttack-Multilabel
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run setup script (optional, creates conda environment)
python install_env.py
```

## Quick Start

### Using the CLI (Recommended)

```bash
# Show help
textattack-multilabel --help

# Run a basic attack on benign samples
textattack-multilabel attack --attack benign

# Preprocess and analyze data
textattack-multilabel preprocess --data data.csv --analyze --sample benign

# Run test suite
textattack-multilabel test --coverage
```

### Using Scripts Directly

```bash
# Download data
python example_toxic_adv_examples/download_data.py

# Run main attack script
python example_toxic_adv_examples/attack_multilabel_tae_main.py

# Run baseline example
python example_toxic_adv_examples/baseline_multiclass_toxic_adv_example_attack.py

# Run ACL23 example
python example_toxic_adv_examples/multilabel_acl2023.py

# Run tests
python test/run_tests.py --coverage
```

## File Structure

```
TextAttack-Multilabel/
├── .gitignore                          # Git ignore file
├── pyproject.toml                      # Package configuration
├── install_env.py                      # Environment setup script
├── LICENSE                             # License file
├── README.md                           # This file
├── example_toxic_adv_examples/         # Example scripts and configs
│   ├── attack_multilabel_tae_main.py   # Main attack script
│   ├── baseline_multiclass_toxic_adv_example_attack.py  # Baseline attack example
│   ├── download_data.py                # Data download script
│   ├── multilabel_acl2023.py           # ACL23 multilabel example
│   ├── config/                         # Configuration files
│   │   └── toxic_adv_examples_config.yaml  # Configuration for examples
│   └── __pycache__/                    # Python cache
├── textattack_multilabel/              # Main package
│   ├── __init__.py                     # Package initialization
│   ├── attack_components.py            # Attack components
│   ├── goal_function.py                # Goal function implementations
│   ├── multilabel_model_wrapper.py     # Model wrapper for multilabel
│   └── multilabel_target_attack_recipe.py  # Multilabel attack recipes
└── test/                               # Test suite
    ├── __init__.py
    ├── run_tests.py                    # Test runner
    ├── test_model_wrapper.py
    ├── test_multilabel_attack_recipes.py
    └── test_shared.py
```

## Configuration

The package uses YAML configuration files for flexible setup:

```yaml
# config/attack_config.yaml
defaults:
  model:
    type: "detoxify"  # or "custom"
    variant: "original"
  dataset:
    name: "jigsaw_toxic_comments"
    sample_size: 500
  attack:
    recipe: "MultilabelACL23"
    labels_to_maximize: []  # maximize all toxic labels
    labels_to_minimize: []  # minimize currently toxic labels
    # ... more options
```

## Examples

### Python API Usage

```python
from textattack_multilabel import MultilabelModelWrapper, MultilabelACL23

# Load your model
model_wrapper = MultilabelModelWrapper(your_model, your_tokenizer, multilabel=True)

# Create attack recipe
attack = MultilabelACL23.build(
    model_wrapper=model_wrapper,
    labels_to_maximize=[0, 1, 2, 3, 4, 5],  # maximize all toxic labels
    labels_to_minimize=[],
    wir_method="delete"
)

# Run attack
attacker = textattack.Attacker(attack, dataset)
results = attacker.attack_dataset()
```

### Jupyter Notebook Tutorial

See `examples/getting_started.ipynb` for a complete walkthrough including:
- Model setup and testing
- Creating attack recipes
- Running attacks on sample data
- Results analysis
- Configuration examples

## Scripts Overview

### Core Scripts

- **`attack_multilabel_tae_main.py`**: Modular attack generation supporting multiple models and datasets
- **`download_data.py`**: Secure Kaggle dataset download with environment variables

### Utility Scripts

- **`install_env.py`**: Cross-platform environment setup with verification
- **`run_tests.py`**: Comprehensive test runner with coverage and parallel execution

### Tests

Run the complete test suite:
```bash
python test/run_tests.py --coverage
```

Test structure:
- **Unit tests**: Individual function/component testing
- **Integration tests**: Full pipeline testing
- **Coverage reporting**: HTML coverage reports generated

## Setup Environment

```bash
python install_env.py
```

This creates a conda environment (py3.8) with required dependencies: textattack[tensorflow,optional], detoxify, kaggle, sentence-transformers. Uses Python subprocess for secure command execution.

## Download Data

Download the Jigsaw Toxic Comments dataset from Kaggle (requires KAGGLE_USERNAME and KAGGLE_KEY environment variables):

```bash
python example_toxic_adv_examples/download_data.py
```

This uses environment variables for Kaggle API credentials instead of plaintext files for improved security.

## Generate Adversarial Examples

See the example scripts in `example_toxic_adv_examples/` for running attack examples, such as `multilabel_acl2023.py`.
