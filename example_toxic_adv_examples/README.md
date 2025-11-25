# TextAttack-Multilabel Examples

This directory contains example scripts demonstrating how to use TextAttack-Multilabel for generating multilabel adversarial examples.

## 📋 Quick Start

### Option 1: End-to-End Demo (Recommended)

Run the complete workflow with built-in sample data:

```bash
# Quick demo (5 samples, fast)
python example_toxic_adv_examples/run_end_to_end_demo.py --quick

# Full demo (10 samples)
python example_toxic_adv_examples/run_end_to_end_demo.py

# Custom configuration
python example_toxic_adv_examples/run_end_to_end_demo.py \
  --num-samples 20 \
  --wir-method gradient \
  --recipe-type transform
```

**What this does:**
1. ✅ Creates sample benign and toxic texts
2. ✅ Loads Detoxify model
3. ✅ Runs multilabel adversarial attacks
4. ✅ Analyzes attack success rates
5. ✅ Saves results with statistics
6. ✅ No data download needed!

### Option 2: Using Your Own Data

Run attacks on the Jigsaw Toxic Comments dataset:

```bash
# 1. Download the data (requires Kaggle API)
python example_toxic_adv_examples/download_data.py

# 2. Run attacks with configuration
python example_toxic_adv_examples/run_multilabel_tae_main.py \
  --config example_toxic_adv_examples/config/attack_config.yaml \
  --attack benign
```

---

## 📁 Files Overview

| File | Description | Use Case |
|------|-------------|----------|
| **`run_end_to_end_demo.py`** | Complete workflow with built-in data | ✅ Quick testing, demos, learning |
| **`run_multilabel_tae_main.py`** | Production script with config files | Production attacks on real data |
| **`download_data.py`** | Download Jigsaw dataset from Kaggle | Get real toxicity data |
| **`baseline_multiclass_toxic_adv_example_attack.py`** | Baseline single-label attacks | Comparison/benchmarking |
| **`config/`** | Configuration files | Customize attack parameters |

---

## 🚀 Detailed Usage

### End-to-End Demo (`run_end_to_end_demo.py`)

#### Basic Usage

```bash
# Quick demo (5 samples, unk method)
python example_toxic_adv_examples/run_end_to_end_demo.py --quick

# Standard demo (10 samples)
python example_toxic_adv_examples/run_end_to_end_demo.py

# More samples
python example_toxic_adv_examples/run_end_to_end_demo.py --num-samples 50
```

#### Advanced Options

```bash
# Use gradient-based word importance ranking
python example_toxic_adv_examples/run_end_to_end_demo.py --wir-method gradient

# Use transform recipe instead of target recipe
python example_toxic_adv_examples/run_end_to_end_demo.py --recipe-type transform

# Attack only benign samples
python example_toxic_adv_examples/run_end_to_end_demo.py --no-attack-toxic

# Attack only toxic samples
python example_toxic_adv_examples/run_end_to_end_demo.py --no-attack-benign
```

#### Available WIR Methods

- `unk` - Unknown token replacement (fastest)
- `delete` - Word deletion importance
- `weighted-saliency` - Gradient-weighted saliency
- `gradient` - Pure gradient-based (slowest, most effective)
- `random` - Random word selection (baseline)

#### Example Output

```
======================================================================
                  TextAttack-Multilabel End-to-End Demo
======================================================================

Start time: 2024-01-15 14:23:45

======================================================================
                        Step 1: Creating Sample Data
======================================================================

✓ Created 10 benign samples
✓ Created 10 toxic samples

======================================================================
                        Step 2: Loading Model
======================================================================

ℹ Using device: cuda
ℹ Loading Detoxify model...
✓ Detoxify model loaded successfully

======================================================================
                     Step 3: Building Attack Recipe
======================================================================

ℹ Attack type: maximize
ℹ WIR method: unk
ℹ Recipe: target
ℹ Goal: Maximize all toxic labels (make benign → toxic)
✓ Attack recipe built successfully
ℹ   - Goal function: MultilabelClassificationGoalFunction
ℹ   - Search method: GreedyWordSwapWIRTruncated
ℹ   - Constraints: 5 active
ℹ   - Transformation: CompositeTransformation

...

======================================================================
                         Final Summary
======================================================================

Overall Results:

Benign → Toxic Attack:
  Success rate: 80.0%
  Successful: 8/10

Toxic → Benign Attack:
  Success rate: 70.0%
  Successful: 7/10

✓ End-to-end demo completed successfully!
```

---

### Production Script (`run_multilabel_tae_main.py`)

For running attacks on real datasets with full configuration control.

#### Prerequisites

1. Download data:
   ```bash
   python example_toxic_adv_examples/download_data.py
   ```

2. Set up Kaggle API credentials (for data download):
   ```bash
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_api_key"
   ```

#### Usage

```bash
# Attack benign samples (make them toxic)
python example_toxic_adv_examples/run_multilabel_tae_main.py \
  --config example_toxic_adv_examples/config/attack_config.yaml \
  --attack benign

# Attack toxic samples (make them benign)
python example_toxic_adv_examples/run_multilabel_tae_main.py \
  --config example_toxic_adv_examples/config/attack_config.yaml \
  --attack toxic

# Attack both
python example_toxic_adv_examples/run_multilabel_tae_main.py \
  --config example_toxic_adv_examples/config/attack_config.yaml \
  --attack both

# Override data path
python example_toxic_adv_examples/run_multilabel_tae_main.py \
  --config example_toxic_adv_examples/config/attack_config.yaml \
  --attack benign \
  --data path/to/your/data.csv
```

#### Configuration File

Edit `config/attack_config.yaml` to customize:

- **Model**: Detoxify variant or custom HuggingFace model
- **Dataset**: Jigsaw or custom dataset
- **Attack**: WIR method, target scores, constraints
- **Output**: Format (parquet/csv), save location

Example config:
```yaml
defaults:
  model:
    type: "detoxify"
    variant: "original"

  dataset:
    name: "jigsaw_toxic_comments"
    sample_size: 500

  attack:
    wir_method: "gradient"
    constraints:
      pos_constraint: true
      sbert_constraint: false
```

---

## 📊 Understanding Attack Results

### Attack Success Criteria

**Benign → Toxic (Maximize):**
- Goal: ALL toxic labels > target_score (default 0.5)
- Example: `[0.1, 0.2, 0.3] → [0.6, 0.7, 0.8]` ✅ Success

**Toxic → Benign (Minimize):**
- Goal: ALL toxic labels < target_score (default 0.5)
- Example: `[0.8, 0.7, 0.9] → [0.3, 0.2, 0.1]` ✅ Success

### Output Files

Results are saved in `results/` directory:

- **`attack_*.parquet`** - Main results file
  - Original text
  - Perturbed text
  - Original predictions
  - Perturbed predictions
  - Number of queries
  - Attack success status

- **`attack_*.summary.txt`** - Statistics summary
  - Total samples
  - Success/fail/skip counts
  - Average queries
  - Average words changed

### Analyzing Results

```python
import pandas as pd

# Load results
df = pd.read_parquet('results/attack_benign_20240115_142345.parquet')

# View successful attacks
successful = df[df['result_type'] == 'Successful']

# Analyze query efficiency
print(f"Avg queries: {df['num_queries'].mean()}")

# Look at perturbations
for idx, row in successful.head(5).iterrows():
    print(f"Original: {row['original_text']}")
    print(f"Perturbed: {row['perturbed_text_clean']}")
    print(f"Queries: {row['num_queries']}\n")
```

---

## 🎯 Attack Recipes Comparison

### MultilabelACL23 (Target Recipe)

**Best for:** Most scenarios, good balance

```bash
python example_toxic_adv_examples/run_end_to_end_demo.py --recipe-type target
```

Features:
- Composite transformations (multiple perturbation types)
- Character swaps, homoglyphs, word substitutions
- Higher success rate

### MultilabelACL23Transform (Transform Recipe)

**Best for:** Specific transformation types

```bash
python example_toxic_adv_examples/run_end_to_end_demo.py --recipe-type transform
```

Features:
- Single transformation method
- Options: GloVe embeddings, MLM, WordNet
- More interpretable perturbations

---

## 🔧 Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'detoxify'`
```bash
# Solution: Install detoxify
pip install detoxify
```

**Issue:** `CUDA out of memory`
```bash
# Solution: Use CPU or reduce batch size
# The script auto-detects device, but you can force CPU mode
CUDA_VISIBLE_DEVICES="" python example_toxic_adv_examples/run_end_to_end_demo.py
```

**Issue:** `FileNotFoundError: Data file not found`
```bash
# Solution: Download data first
python example_toxic_adv_examples/download_data.py
```

**Issue:** Attack runs very slowly
```bash
# Solution: Use faster WIR method
python example_toxic_adv_examples/run_end_to_end_demo.py --wir-method unk

# Or reduce samples
python example_toxic_adv_examples/run_end_to_end_demo.py --num-samples 5
```

---

## 📚 Next Steps

1. **Run the quick demo** to see the workflow:
   ```bash
   python example_toxic_adv_examples/run_end_to_end_demo.py --quick
   ```

2. **Try different WIR methods** to compare effectiveness:
   ```bash
   for method in unk delete gradient; do
     python example_toxic_adv_examples/run_end_to_end_demo.py --wir-method $method
   done
   ```

3. **Experiment with real data** using the production script

4. **Analyze results** to understand attack patterns

5. **Customize attacks** by modifying configuration files

---

## 💡 Tips

- **Start small**: Use `--quick` mode first to verify setup
- **GPU recommended**: Attacks run 10-50x faster on GPU
- **WIR method matters**: `gradient` is most effective but slowest
- **Check constraints**: Adjust POS/SBERT constraints for quality vs. success rate
- **Save results**: All outputs include timestamps for versioning

---

## 📖 Further Reading

- Main README: `../README.md`
- Package documentation: `../textattack_multilabel/`
- TextAttack documentation: https://textattack.readthedocs.io/
- Research paper: [ACL 2023 Multilabel Attacks]

---

## 🤝 Contributing

Found issues or have improvements? Please open an issue or PR in the main repository!
