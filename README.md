# TextAttack-Multilabel

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](test/)
[![Coverage](https://img.shields.io/badge/coverage-45%25-yellow.svg)](test/)

A professional extension of [TextAttack](https://github.com/QData/TextAttack) for multi-label adversarial example generation, with focus on toxicity classification. Generate adversarial examples that flip multiple labels simultaneously while preserving semantic meaning and grammatical correctness.

## ✨ Features

- 🎯 **Multi-label Attacks**: Attack multiple labels simultaneously (maximize/minimize different label sets)
- 🏗️ **Modular Architecture**: Support for multiple models (Detoxify, custom HuggingFace models)
- 🔬 **Multiple Attack Recipes**: Composite transformations and single-method attacks
- 📊 **Configuration-Driven**: YAML configuration for flexible attack parameters
- 🧪 **Comprehensive Testing**: 78+ test functions with 45% code coverage
- 📈 **Built-in Analysis**: Attack success metrics, query statistics, and result visualization
- 🚀 **Easy Installation**: Pip-installable with automatic dependency management
- 🎓 **Complete Examples**: End-to-end demos with built-in data

## 📦 Installation

### Quick Install (Recommended)

```bash
# Install from source
git clone https://github.com/QData/TextAttack-Multilabel
cd TextAttack-Multilabel
pip install -e .
```

### Enviroment Setup and Verify 
```bash
python install_env.py
```


### Development Installation

```bash
# Install with development dependencies (testing, linting, type checking)
pip install -e ".[dev]"

# Verify installation
python -c "from textattack_multilabel import MultilabelACL23; print('✓ Installation successful')"
```

### Requirements

- Python 3.8+
- PyTorch 1.9+
- TextAttack 0.3.0+
- Transformers 4.10+
- See `pyproject.toml` for complete dependencies

## 🚀 Quick Start

### Option 1: End-to-End Demo (Fastest)

Run a complete workflow with built-in sample data (no download needed):

```bash
# Quick demo (5 samples, ~2 minutes)
python example_toxic_adv_examples/run_end_to_end_demo.py --quick

# Full demo with analysis
python example_toxic_adv_examples/run_end_to_end_demo.py
```

**What this does:**
1. Creates sample benign/toxic texts
2. Loads Detoxify toxicity model
3. Runs multilabel adversarial attacks
4. Analyzes attack success rates
5. Shows example perturbations
6. Saves detailed results

### Option 2: Python API

```python
from textattack_multilabel import (
    MultilabelModelWrapper,
    MultilabelACL23_recipe,
    MultilabelACL23Transform
)
import transformers

# Load your model and tokenizer
model = transformers.AutoModelForSequenceClassification.from_pretrained("your-model")
tokenizer = transformers.AutoTokenizer.from_pretrained("your-model")

# Wrap for multilabel attacks
model_wrapper = MultilabelModelWrapper(
    model,
    tokenizer,
    multilabel=True,
    device='cuda'  # Auto-detects if None
)

# Build attack: maximize toxic labels (make benign text toxic)
mattack = MultilabelACL23_recipe.build(
    model_wrapper=model_wrapper,
    labels_to_maximize=[0, 1, 2, 3, 4, 5],  # All 6 toxic labels
    labels_to_minimize=[],
    wir_method="gradient"  # Options: unk, delete, gradient, weighted-saliency
)

# Run attack
import textattack
dataset = textattack.datasets.Dataset([("Sample text", [0.1, 0.2, 0.3, 0.1, 0.2, 0.1])])
attacker = textattack.Attacker(mattack, dataset)
results = attacker.attack_dataset()
```

### Option 3: Configuration-Based

```bash
# Run attacks with configuration file
python example_toxic_adv_examples/run_multilabel_tae_main.py \
  --config example_toxic_adv_examples/config/attack_config.yaml \
  --attack toxic
```

## 📖 Package Structure

```
textattack_multilabel/
├── __init__.py                              # Public API exports
├── multilabel_model_wrapper.py              # Model wrapper with gradient support
├── goal_function.py                         # Multi-label goal functions
├── attack_components.py                     # Search methods and components
├── multilabel_target_attack_recipe.py       # Composite attack recipe
└── multilabel_transform_attack_recipe.py    # Single-method attack recipe
```

### Core Components

#### **MultilabelModelWrapper**
Wraps HuggingFace models for multilabel classification with:
- Automatic device detection (CUDA/CPU)
- Gradient computation for gradient-based attacks
- Sigmoid activation for multilabel outputs
- Support for T5 and standard transformer models

#### **MultilabelClassificationGoalFunction**
Goal function that can:
- Maximize specific labels (make text toxic)
- Minimize specific labels (make text benign)
- Combine both objectives simultaneously
- Validate multi-label success criteria

#### **Attack Recipes**

**MultilabelACL23_recipe** (Recommended):
- Composite transformations (word swaps, character edits, homoglyphs)
- Multiple WIR methods: `unk`, `delete`, `weighted-saliency`, `gradient`
- Flexible constraint configuration

**MultilabelACL23Transform**:
- Single transformation methods: `glove`, `mlm`, `wordnet`
- Simpler, more interpretable perturbations

## 🧪 Testing

We have comprehensive test coverage with 78+ test functions:

```bash
# Run all tests
python test/run_tests.py

# Run with coverage report
python test/run_tests.py --coverage

# Run specific test file
python test/run_tests.py --file test_goal_function_core.py

# Run in parallel (4 workers)
python test/run_tests.py --parallel 4

# List all test files
python test/run_tests.py --list
```

**Test Suite Overview:**
- **test_goal_function_core.py**: 43 tests for goal function logic
- **test_model_wrapper_advanced.py**: 29 tests for model wrapper and gradients
- **test_model_wrapper.py**: Basic wrapper tests
- **test_multilabel_attack_recipes.py**: Recipe building tests

**Coverage:** ~45% (goal function: 95%, model wrapper: 85%)

## 📊 Examples

### Complete Examples Directory

See `example_toxic_adv_examples/README.md` for detailed documentation.

#### Quick Examples

```bash
# 1. End-to-end demo (no data needed)
python example_toxic_adv_examples/run_end_to_end_demo.py --quick

# 2. Custom parameters
python example_toxic_adv_examples/run_end_to_end_demo.py \
  --num-samples 20 \
  --wir-method gradient \
  --recipe-type transform

# 3. Attack only benign samples
python example_toxic_adv_examples/run_end_to_end_demo.py --no-attack-toxic
```

### Attack Direction Examples

**Make Benign Text Toxic (Maximize):**
```python
attack = MultilabelACL23_recipe.build(
    model_wrapper=model_wrapper,
    labels_to_maximize=[0, 1, 2, 3, 4, 5],  # Maximize all toxic labels
    labels_to_minimize=[],
    maximize_target_score=0.5  # Target: all labels > 0.5
)
```

**Make Toxic Text Benign (Minimize):**
```python
attack = MultilabelACL23_recipe.build(
    model_wrapper=model_wrapper,
    labels_to_maximize=[],
    labels_to_minimize=[0, 1, 2, 3, 4, 5],  # Minimize all toxic labels
    minimize_target_score=0.5  # Target: all labels < 0.5
)
```

**Mixed Objectives:**
```python
attack = MultilabelACL23_recipe.build(
    model_wrapper=model_wrapper,
    labels_to_maximize=[0, 1],  # Maximize toxic and severe_toxic
    labels_to_minimize=[2, 3],  # Minimize obscene and threat
)
```

## ⚙️ Configuration

### YAML Configuration Files

Edit `example_toxic_adv_examples/config/attack_config.yaml`:

```yaml
defaults:
  model:
    type: "detoxify"  # or "custom"
    variant: "original"

  dataset:
    name: "jigsaw_toxic_comments"
    sample_size: 500

  attack:
    wir_method: "gradient"  # unk, delete, weighted-saliency, gradient
    labels_to_maximize: []  # Empty = all labels
    labels_to_minimize: []
    maximize_target_score: 0.5
    minimize_target_score: 0.5

    constraints:
      pos_constraint: true      # Maintain part-of-speech
      sbert_constraint: false   # Semantic similarity
```

## 📈 Results and Analysis

Attack results include:

- **Success Rate**: Percentage of successful attacks
- **Query Efficiency**: Average queries per attack
- **Perturbation Quality**: Words changed, character edits
- **Label Changes**: Before/after predictions for all labels
- **Example Outputs**: Actual perturbed texts

### Output Files

Results saved as:
- `*.parquet` - Complete results with predictions
- `*.summary.txt` - Statistics and metrics
- `htmlcov/` - Test coverage reports (when using `--coverage`)

## 🔧 Advanced Usage

### Custom Models

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack_multilabel import MultilabelModelWrapper

# Load your custom model
model = AutoModelForSequenceClassification.from_pretrained(
    "path/to/your/model",
    num_labels=6
)
tokenizer = AutoTokenizer.from_pretrained("path/to/your/tokenizer")

# Wrap it
wrapper = MultilabelModelWrapper(model, tokenizer, multilabel=True)
```

### Custom Attack Parameters

```python
from textattack_multilabel import MultilabelACL23Transform

# Use WordNet transformations with beam search
attack = MultilabelACL23Transform.build(
    model_wrapper=wrapper,
    labels_to_maximize=[0, 1, 2],
    transform_method="wordnet",  # glove, mlm, or wordnet
    wir_method="beam",
    pos_constraint=True,
    sbert_constraint=True  # Add semantic similarity constraint
)
```

### Gradient-Based Attacks

```python
# Most effective but slowest
attack = MultilabelACL23_recipe.build(
    model_wrapper=wrapper,
    labels_to_maximize=[0, 1, 2, 3, 4, 5],
    wir_method="gradient",  # Gradient-guided word importance
    pos_constraint=True
)
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Force CPU mode
CUDA_VISIBLE_DEVICES="" python example_toxic_adv_examples/run_end_to_end_demo.py
```

**Import Errors:**
```bash
# Verify installation
pip install -e .

# Check imports
python -c "from textattack_multilabel import MultilabelACL23; print('OK')"
```

**Slow Attacks:**
```bash
# Use faster WIR method
python example_toxic_adv_examples/run_end_to_end_demo.py --wir-method unk
```

## 🧑‍💻 Development

### Running Tests

```bash
# All tests with coverage
python test/run_tests.py --coverage

# Quality checks (black, isort, mypy)
python test/run_tests.py --quality

# Specific test class
python test/run_tests.py --test test/test_goal_function_core.py::TestGetScore
```

### Code Quality

```bash
# Format code
black textattack_multilabel/ test/

# Sort imports
isort textattack_multilabel/ test/

# Type checking
mypy textattack_multilabel/ --ignore-missing-imports
```

## 📚 Documentation

- **Package API**: See docstrings in `textattack_multilabel/`
- **Examples**: `example_toxic_adv_examples/README.md`
- **Tests**: `test/` directory with comprehensive examples
- **TextAttack Docs**: https://textattack.readthedocs.io/

## 🔬 Research

If you use this package in your research, please cite:

```bibtex
@inproceedings{textattack-multilabel-2023,
  title={Multi-label Adversarial Attacks for Text Classification},
  author={QData Lab},
  booktitle={ACL},
  year={2023}
}
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run `python test/run_tests.py --quality` before submitting
5. Submit a pull request

## 📄 License

Apache 2.0 License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

Built on top of [TextAttack](https://github.com/QData/TextAttack) by QData Lab.

## 📞 Support

- **Issues**: https://github.com/QData/TextAttack-Multilabel/issues
- **Documentation**: See `example_toxic_adv_examples/README.md`
- **Tests**: Run `python test/run_tests.py --help`

---

**Quick Links:**
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Examples](example_toxic_adv_examples/README.md)
- [Testing](#-testing)
- [API Reference](#-package-structure)
