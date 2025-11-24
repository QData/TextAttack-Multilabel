# TextAttack Multi-label Adversarial Examples Generator

This folder contains scripts for generating multi-label adversarial examples by extending the [TextAttack](https://github.com/QData/TextAttack) library for multi-label toxicity classification tasks.

## Python Package Structure

- `__init__.py`: Package initialization.
- `multilabel_acl2023.py`: Core attack recipes for multi-label scenarios, including constraints (grammar, semantics), transformations (word/character swaps), goal functions, and search methods (gradient, beam, genetic, etc.).
- `shared.py`: Shared utilities like `MultilabelClassificationGoalFunction`, custom search methods (`GreedyWordSwapWIRTruncated`), and result handling.
- `model.py`: `MultilabelModelWrapper` for compatibility with TextAttack's model interface.

## File Structure and Functional Modules

```
textattack_muitilabel/
├── attack_multilabel.py
├── download_data.py
├── install_env.py
├── README.md
├── test/
│   ├── __init__.py
│   ├── test_attack_recipes.py
│   ├── test_model_wrapper.py
│   └── test_shared.py
└── textattack_multilabel/
    ├── __init__.py
    ├── model.py
    ├── multiclass_acl2023.py
    ├── multilabel_acl2023.py
    └── shared.py
```

### Root Files
- `attack_multilabel.py`: Main script for generating multi-label adversarial examples using TextAttack recipes.
- `download_data.py`: Secure Python script to download Jigsaw Toxic Comments dataset using Kaggle API with environment variables.
- `install_env.py`: Secure Python script for environment setup using subprocess validation.
- `README.md`: This documentation file.

### test/ (Testing)
- `__init__.py`: Test package initialization.
- `test_attack_recipes.py`: Unit tests for attack recipe building and configuration.
- `test_model_wrapper.py`: Tests for MultilabelModelWrapper functionality.
- `test_shared.py`: Tests for shared components like goal functions and search methods.

### textattack_multilabel/ (Core Package)
- `__init__.py`: Package initialization, exposes MultilabelACL23 and related classes.
- `multiclass_acl2023.py`: Contains MultilabelACL23Transform attack recipe (alternative implementation).
- `multilabel_acl2023.py`: Core attack recipes (MultilabelACL23, MultilabelACL23Transform) with multi-label goal functions, constraints, transformations, and search methods.
- `shared.py`: Shared components like PartOfSpeechTry, GreedyWordSwapWIRTruncated, MultilabelClassificationGoalFunction, custom search algorithms.
- `model.py`: MultilabelModelWrapper adapting models for TextAttack's multi-label interface.

## Setup Environment

```bash
python install_env.py
```

This creates a conda environment (py3.8) with required dependencies: textattack[tensorflow,optional], detoxify, kaggle, sentence-transformers. Uses Python subprocess for secure command execution.

## Download Data

Download the Jigsaw Toxic Comments dataset from Kaggle (requires KAGGLE_USERNAME and KAGGLE_KEY environment variables):

```bash
python download_data.py
```

This uses environment variables for Kaggle API credentials instead of plaintext files for improved security.

## Generate Adversarial Examples

### Attack Toxic Samples (Generate Benign Adversaries)

Attack harmful examples from the Jigsaw test split to produce benign adversarial examples:

```shell
python attack_multilabel.py --output attack_harmful.parquet --attack harmful --data data/jigsaw_toxic_comments/test.csv
```

### Attack Benign Samples (Generate Toxic Adversaries)

Attack benign examples to produce harmful adversarial examples:

```shell
python attack_multilabel.py --output attack_benign.parquet --attack benign --data data/jigsaw_toxic_comments/test.csv
```

Outputs parquet files containing perturbed texts, predictions, and ground truth labels.

## Testing

Run the test suite to verify functionality:

```bash
cd textattack_muitilabel
pip install pytest  # if not already installed
python -m pytest test/
```

Tests cover:
- Attack recipe building and configuration
- Model wrapper functionality for single and multi-label modes
- Shared components (goal functions, search methods)

## Data Flow

1. Load Jigsaw dataset (6 toxicity labels: toxic, severe_toxic, obscene, threat, insult, identity_hate).
2. Sample 500 benign/toxic examples.
3. Use detoxify as target model.

## Compatibility

- **PyTorch**: Requires a PyTorch build that supports `register_full_backward_hook`. Use `PyTorch >= 1.9` (recommended) or a newer stable release to ensure backward-hook behavior used in `textattack_multilabel/shared.py` works correctly.
- **transformers**: Recommended `transformers >= 4.10`. Newer releases are supported, but ensure tokenizer and model APIs (e.g., `model.config` and tokenizer `model_max_length`) are available.
- **TextAttack**: Tested with `textattack >= 0.4.0`. The package relies on TextAttack's `Attack`, `GoalFunction`, and `SearchMethod` APIs; if you encounter attribute errors, verify your TextAttack version.

These are recommendations to reduce compatibility issues (especially around gradient hooks and model/tokenizer interfaces). If you use older library versions and encounter errors related to backward hooks or tokenizer/model attributes, upgrading to the versions above is the quickest remedy.
