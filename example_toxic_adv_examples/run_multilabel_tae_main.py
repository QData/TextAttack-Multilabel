"""
Modular adversarial attack script for TextAttack-Multilabel.
Supports multiple models and datasets through configuration.
"""

import os
import argparse
import logging
import re
import sys
import time
import torch
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

import textattack
import transformers
from detoxify import Detoxify
from textattack import Attacker
from textattack.loggers import CSVLogger
from textattack_multilabel import MultilabelModelWrapper
from textattack_multilabel.attack_components import AttackResult_new_diff_color
from textattack_multilabel.multilabel_target_attack_recipe import MultilabelACL23_recipe as MultilabelACL23
import nltk

nltk.download('omw-1.4', quiet=True)


def get_optimal_device():
    """Get the optimal device available (MPS -> CUDA -> CPU)."""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


textattack.attack_results.AttackResult.diff_color = AttackResult_new_diff_color


def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config):
    """Load model based on configuration."""
    model_config = config['defaults']['model']

    if model_config['type'] == 'detoxify':
        print(f"Loading Detoxify model: {model_config['variant']}")

        # Check for MPS but fall back to CPU for Detoxify compatibility
        # (Detoxify may have limited MPS support)
        if torch.backends.mps.is_available():
            print("Note: Detoxify may have limited MPS support, falling back to CPU for compatibility")
            device = 'cpu'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        print(f"Using device: {device}")

        # Initialize detoxify model
        detoxify_model = Detoxify(model_config['variant'], device=device)

        # Wrap in our MultilabelModelWrapper
        # Detoxify uses RoBERTa-base, so we need to get the underlying model
        # This is a bit tricky - we'll need to extract the model from detoxify
        # For now, use the public interface
        class DetoxifyWrapper(MultilabelModelWrapper):
            def __init__(self, detoxify_model, device):
                # Get the underlying RoBERTa model
                self.detoxify = detoxify_model
                self.class_names = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
                self.device = device
                # Use RoBERTa tokenizer
                from transformers import RobertaTokenizer
                tokenizer = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base')
                tokenizer.model_max_length = 128
                super().__init__(self.detoxify.model, tokenizer, multilabel=True, device=device, max_length=128)

            def __call__(self, text_input_list):
                # Use detoxify's predict method
                predictions = self.detoxify.predict(text_input_list)
                # Convert to tensor format expected by TextAttack
                if isinstance(predictions, dict):
                    # Stack predictions
                    pred_array = np.stack([predictions[label] for label in self.class_names])
                    return torch.tensor(pred_array.T, dtype=torch.float32)
                return predictions

        # Create wrapper
        wrapper = DetoxifyWrapper(detoxify_model, device=device)

        # Warm up the model to prevent hanging during attacks
        print("Warming up Detoxify model...")
        try:
            wrapper(["This is a test sentence."])
            print("Model warmed up successfully")
        except Exception as e:
            print(f"Warning: Model warmup failed: {e}")

        return wrapper

    elif model_config['type'] == 'custom':
        print("Loading custom model...")
        custom_config = config['custom_model']

        tokenizer = transformers.AutoTokenizer.from_pretrained(custom_config['tokenizer_path'])
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            custom_config['model_path'],
            num_labels=len(custom_config['labels'])
        )

        return MultilabelModelWrapper(
            model, tokenizer,
            multilabel=custom_config['multilabel'],
            device=custom_config.get('device', get_optimal_device()),
            max_length=128
        )

    else:
        raise ValueError(f"Unsupported model type: {model_config['type']}")


def load_dataset(config, args):
    """Load and prepare dataset based on configuration."""
    dataset_config = config['defaults']['dataset']

    if dataset_config['name'] == 'jigsaw_toxic_comments':
        return load_jigsaw_dataset(args.data or dataset_config['path'], config)
    elif dataset_config['name'] == 'custom':
        return load_custom_dataset(config, args)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_config['name']}")


def load_jigsaw_dataset(data_path, config):
    """Load Jigsaw toxic comment dataset."""
    print(f"Loading Jigsaw dataset from {data_path}")

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}, attempting to download...")
        try:
            # Run download_data.py
            import subprocess
            result = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'download_data.py')], check=True, capture_output=True, text=True)
            print(result.stdout)
            print("Download completed.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download data: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            sys.exit(1)

    dataset_config = config['defaults']['dataset']
    benign_threshold = dataset_config['benign_threshold']
    toxic_threshold = dataset_config['toxic_threshold']

    df = pd.read_csv(data_path)
    df = df.assign(labels=df[['toxic', 'severe_toxic','obscene','threat','insult','identity_hate']].values.tolist())
    df = df.rename(columns={'comment_text':'text'})[['text', 'labels']]

    df_benign = df[df.labels.apply(lambda x: np.all(np.asarray(x) < benign_threshold))]
    df_toxic = df[df.labels.apply(lambda x: np.any(np.asarray(x) > toxic_threshold))]

    # Sample if configured
    sample_size = dataset_config.get('sample_size', len(df_benign))
    df_benign = df_benign.sample(min(sample_size, len(df_benign)), random_state=42)
    df_toxic = df_toxic.sample(min(sample_size, len(df_toxic)), random_state=42)

    return df_benign, df_toxic


def load_custom_dataset(config, args):
    """Load custom dataset."""
    print("Loading custom dataset...")
    custom_config = config['custom_dataset']

    df = pd.read_csv(args.data or custom_config['data_path'])

    # Assuming custom dataset format
    # This would need customization based on actual format
    if custom_config['format'] == 'csv':
        text_col = custom_config['text_column']
        label_cols = custom_config['label_columns']

        df['labels'] = df[label_cols].values.tolist()
        df = df.rename(columns={text_col: 'text'})
        df = df[['text', 'labels']].dropna()

    # This is a basic implementation - would need expansion
    df_benign = df.sample(frac=0.5, random_state=42)
    df_toxic = df.drop(df_benign.index)

    return df_benign, df_toxic


def run_attack(model_wrapper, df_samples, attack_direction, config):
    """Run attack on dataset samples."""
    import time
    start_time = time.time()

    attack_config = config['defaults']['attack']
    total_samples = len(df_samples)
    print(f"Starting attack on {total_samples} {attack_direction} samples")

    # Determine which labels to maximize/minimize based on attack direction
    if attack_direction == 'benign':
        # Attacking benign samples: maximize all toxic labels
        labels_to_maximize = list(range(6))  # Assuming 6 toxic labels
        labels_to_minimize = []
    else:  # attack == 'toxic'
        # Attacking toxic samples: minimize all toxic labels
        labels_to_maximize = []
        labels_to_minimize = list(range(6))

    attack = MultilabelACL23.build(
        model_wrapper=model_wrapper,
        labels_to_maximize=attack_config.get('labels_to_maximize', labels_to_maximize),
        labels_to_minimize=attack_config.get('labels_to_minimize', labels_to_minimize),
        maximize_target_score=attack_config.get('maximize_target_score', 0.5),
        minimize_target_score=attack_config.get('minimize_target_score', 0.5),
        wir_method=attack_config.get('wir_method', 'unk'),
        pos_constraint=attack_config['constraints']['pos_constraint'],
        sbert_constraint=attack_config['constraints']['sbert_constraint']
    )

    dataset = textattack.datasets.Dataset(
        [(x, y) for x, y in zip(df_samples["text"], df_samples["labels"])]
    )

    attack_args = textattack.AttackArgs(num_examples=-1, log_to_csv=None, csv_coloring_style=None)
    attacker = Attacker(attack, dataset, attack_args)

    print(f"Built attack with method: {attack_config.get('wir_method', 'unk')}")
    print("Starting attack_dataset...")
    attack_results = []

    # Attack each example individually to track progress
    for i, (text, label) in enumerate(zip(df_samples["text"], df_samples["labels"])):
        if i % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_sample = elapsed_time / (i + 1) if i > 0 else 0
            estimated_remaining = avg_time_per_sample * (total_samples - i)
            print(f"Attacking sample {i+1}/{total_samples} ({(i+1)/total_samples*100:.1f}%) - "
                  f"Elapsed: {elapsed_time:.1f}s, Estimated remaining: {estimated_remaining:.1f}s, "
                  f"Avg time/sample: {avg_time_per_sample:.2f}s")

        try:
            result = attacker.attack(text, label)
            attack_results.append(result)
        except Exception as e:
            print(f"Error attacking sample {i+1}: {e}")
            attack_results.append(None)

    total_elapsed = time.time() - start_time
    print(f"Attack completed in {total_elapsed:.2f}s - Average time per sample: {total_elapsed/total_samples:.2f}s")

    return attack_results


def save_results(attack_results, df_samples, output_config, attack_direction):
    # Filter out successful attack results
    successful_results = [r for r in attack_results if r is not None and r.perturbed_result is not None]
    if not successful_results:
        print("No successful attacks found, creating empty results")
        df_attacks = pd.DataFrame(columns=['ground_truth_label', 'text', 'original_output', 'perturbed_output'])
        df_attacks['ground_truth_label'] = df_samples['labels'].values
        df_attacks['text'] = df_samples['text'].values
        df_attacks['original_output'] = [None] * len(df_samples)
        df_attacks['perturbed_output'] = [None] * len(df_samples)
    else:
        attack_logger = CSVLogger(color_method='html')
        for result in successful_results:
            attack_logger.log_attack_result(result)

        df_attacks = attack_logger.df
        df_attacks.loc[:, 'ground_truth_label'] = df_samples['labels'].values[:len(df_attacks)]
        df_attacks.loc[:, 'text'] = df_attacks['perturbed_text'].apply(lambda x: re.sub(r'<font color = .{1,6}>|</font>', '', x))
        df_attacks['text'] = df_attacks['text'].str.replace('<SPLIT>', '\n', regex=True)

        # Convert outputs to numpy if they're tensors
        df_attacks["original_output"] = df_attacks["original_output"].apply(
            lambda x: x.cpu().numpy() if hasattr(x, 'cpu') else x
        )
        df_attacks["perturbed_output"] = df_attacks["perturbed_output"].apply(
            lambda x: x.cpu().numpy() if hasattr(x, 'cpu') else x
        )


    # Save based on configuration
    output_format = output_config.get('format', 'parquet')
    output_prefix = output_config.get('prefix', 'attack_results')

    output_filename = f"{output_prefix}_{attack_direction}.{output_format}"

    if output_format == 'parquet':
        df_attacks.to_parquet(output_filename)
    elif output_format == 'csv':
        df_attacks.to_csv(output_filename, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"Saved attack results to {output_filename}")
    return df_attacks


def display_attack_results(results_df, attack_direction):
    """Display attack results in a human-readable format."""
    print("\n" + "="*70)
    print(f"🎯 MULTILABEL ADVERSARIAL ATTACK RESULTS - {attack_direction.upper()}")
    print("="*70)

    if results_df.empty:
        print("❌ No attack results found.")
        return

    # Calculate success rate and metrics
    successful = len(results_df[results_df['perturbed_text'].notna()])

    print(f"📊 Summary:")
    print(f"   Total samples: {len(results_df)}")
    print(f"   Successful attacks: {successful}")
    print(f"   Success rate: {successful/len(results_df)*100:.1f}%")

    if successful > 0:
        # Calculate average toxicity reduction
        valid_results = results_df[results_df['perturbed_text'].notna()]
        original_scores = np.array(valid_results['original_output'].tolist())
        perturbed_scores = np.array(valid_results['perturbed_output'].tolist())

        # Calculate how much toxicity changed (direction depends on attack type)
        if attack_direction == 'benign':
            # For benign→toxic, higher scores = more successful
            avg_change = np.mean(perturbed_scores - original_scores)
            print(f"   Average toxicity increase: +{avg_change:.3f}")
        else:  # toxic attacks
            # For toxic→benign, lower scores = more successful
            avg_change = np.mean(original_scores - perturbed_scores)
            print(f"   Average toxicity reduction: +{avg_change:.3f}")

    print("\n🔍 Detailed Results:")
    print("-" * 50)

    for idx, row in results_df.iterrows():
        print(f"\nSample {idx + 1}:")
        print(f"  Ground truth labels: {row.ground_truth_label}")

        if pd.notna(row.perturbed_text):
            # Show original text (truncated)
            orig_text = row.get('original_text', row['text'][0] if isinstance(row['text'], list) else row['text']) if 'original_text' in results_df.columns else row['text']
            orig_text = re.sub(r'<font color = .{1,6}>|</font>', '', str(orig_text)).replace('<SPLIT>', ' ')
            print(f"  Original: {orig_text[:80]}{'...' if len(str(orig_text)) > 80 else ''}")

            # Show perturbed text (truncated)
            pert_text = re.sub(r'<font color = .{1,6}>|</font>', '', row['perturbed_text']).replace('<SPLIT>', ' ')
            print(f"  Perturbed: {pert_text[:80]}{'...' if len(str(pert_text)) > 80 else ''}")

            # Show score changes
            orig_scores = row['original_output']
            pert_scores = row['perturbed_output']
            print(f"  Original scores: [{', '.join([f'{x:.3f}' for x in orig_scores])}]")
            print(f"  Perturbed scores: [{', '.join([f'{x:.3f}' for x in pert_scores])}]")
            print(f"  Change: [{' + '.join([f'{pert_scores[i]-orig_scores[i]:+.3f}' for i in range(len(orig_scores))])}]")
            print("  ✓ SUCCESSFUL ATTACK")
        else:
            print(f"  Original: {row['text'][:80]}{'...' if len(str(row['text'])) > 80 else ''}")
            print("  Perturbed: N/A")
            print("  ✗ FAILED ATTACK")

    print("\n" + "="*70)
    print("💾 Results saved to: attack_results_{}.{}".format(
        attack_direction,
        'parquet'  # Could make this configurable based on output_format
    ))


def attack_model(config_path, args):
    """Main attack function with configuration support."""
    # Load configuration
    config = load_config(config_path)

    # Load model
    model_wrapper = load_model(config)

    # Load dataset
    df_benign, df_toxic = load_dataset(config, args)

    print(f"Benign samples: {len(df_benign)}")
    print(f"Toxic samples: {len(df_toxic)}")

    # Run attacks
    if args.attack in ['benign', 'both']:
        print("\nRunning attack on benign samples...")
        benign_results = run_attack(model_wrapper, df_benign, 'benign', config)
        benign_df = save_results(benign_results, df_benign, config['defaults']['output'], 'benign')
        display_attack_results(benign_df, 'benign')

    if args.attack in ['toxic', 'both']:
        print("\nRunning attack on toxic samples...")
        toxic_results = run_attack(model_wrapper, df_toxic, 'toxic', config)
        toxic_df = save_results(toxic_results, df_toxic, config['defaults']['output'], 'toxic')
        display_attack_results(toxic_df, 'toxic')


if __name__ == "__main__":
    ## Argument parser
    parser = argparse.ArgumentParser(description="Run TextAttack-Multilabel adversarial attacks")
    parser.add_argument(
        "--config",
        type=str,
        default="config/attack_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--attack",
        type=str,
        required=True,
        choices=['benign', 'toxic', 'both'],
        help="Attack benign or toxic examples, or both",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to input dataset (overrides config)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()
    print(f"Arguments: {args}")

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('attack_multilabel.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        attack_model(args.config, args)
        print("Attack completed successfully!")
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory. Try reducing batch size or free GPU memory.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
