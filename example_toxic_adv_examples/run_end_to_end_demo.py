#!/usr/bin/env python3
"""
Complete End-to-End Demo for TextAttack-Multilabel

This script demonstrates the full workflow:
1. Download/prepare sample data
2. Load model (Detoxify or custom)
3. Run multilabel adversarial attacks
4. Evaluate attack success
5. Generate analysis and visualization
6. Save results

Usage:
    # Quick demo with small dataset
    python example_toxic_adv_examples/run_end_to_end_demo.py --quick

    # Full demo with configuration
    python example_toxic_adv_examples/run_end_to_end_demo.py --config config/toxic_adv_examples_config.yaml

    # Custom attack parameters
    python example_toxic_adv_examples/run_end_to_end_demo.py --num-samples 100 --wir-method gradient
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import yaml

# TextAttack imports
import textattack
from textattack import Attacker
from textattack.loggers import CSVLogger

# TextAttack-Multilabel imports
from textattack_multilabel import (
    MultilabelModelWrapper,
    MultilabelACL23_recipe,
    MultilabelACL23Transform
)
from textattack_multilabel.attack_components import AttackResult_new_diff_color

# Apply custom diff color
textattack.attack_results.AttackResult.diff_color = AttackResult_new_diff_color


class Colors:
    """Terminal color codes."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg):
    """Print colored header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(msg):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} {msg}")


def print_info(msg):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ{Colors.ENDC} {msg}")


def print_warning(msg):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {msg}")


def create_sample_data(num_samples=10):
    """Create sample toxic/benign data for demonstration."""
    print_header("Step 1: Creating Sample Data")

    benign_samples = [
        "I really enjoyed this movie",
        "The weather is nice today",
        "This is a helpful tutorial",
        "Thank you for your assistance",
        "I agree with your point",
        "Great work on this project",
        "The food was delicious",
        "I love spending time with friends",
        "This book is very informative",
        "Have a wonderful day"
    ][:num_samples]

    toxic_samples = [
        "You are such an idiot",
        "I hate everything about this",
        "This is complete garbage",
        "Shut up and leave",
        "You're totally useless",
        "What a stupid comment",
        "This makes me so angry",
        "I can't stand people like you",
        "This is the worst thing ever",
        "Go away and never come back"
    ][:num_samples]

    # Create dummy labels (6 toxicity categories)
    benign_labels = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in benign_samples]
    toxic_labels = [[0.8, 0.1, 0.7, 0.05, 0.6, 0.1] for _ in toxic_samples]

    df_benign = pd.DataFrame({'text': benign_samples, 'labels': benign_labels})
    df_toxic = pd.DataFrame({'text': toxic_samples, 'labels': toxic_labels})

    print_success(f"Created {len(benign_samples)} benign samples")
    print_success(f"Created {len(toxic_samples)} toxic samples")

    return df_benign, df_toxic


def load_model(model_type='detoxify', device=None):
    """Load the model for attacks."""
    print_header("Step 2: Loading Model")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print_info(f"Using device: {device}")

    if model_type == 'detoxify':
        from detoxify import Detoxify
        from transformers import RobertaTokenizer

        print_info("Loading Detoxify model...")
        detoxify_model = Detoxify('original', device=device)

        # Wrap for TextAttack
        class DetoxifyWrapper(MultilabelModelWrapper):
            def __init__(self, detoxify_model, device):
                self.detoxify = detoxify_model
                self.class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

                # Get underlying model and tokenizer
                tokenizer = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base')
                tokenizer.model_max_length = 512

                super().__init__(
                    detoxify_model.model,
                    tokenizer,
                    multilabel=True,
                    device=device,
                    max_length=512
                )

            def __call__(self, text_input_list):
                # Use Detoxify's predict method
                predictions = self.detoxify.predict(text_input_list)

                # Convert dict to tensor
                if isinstance(predictions, dict):
                    pred_array = np.stack([predictions[label] for label in self.class_names])
                    return torch.tensor(pred_array.T, dtype=torch.float32)
                return predictions

        model_wrapper = DetoxifyWrapper(detoxify_model, device)
        print_success("Detoxify model loaded successfully")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model_wrapper


def build_attack(model_wrapper, attack_type='maximize', wir_method='unk', recipe_type='target'):
    """Build the attack recipe."""
    print_header("Step 3: Building Attack Recipe")

    print_info(f"Attack type: {attack_type}")
    print_info(f"WIR method: {wir_method}")
    print_info(f"Recipe: {recipe_type}")

    # Configure labels based on attack type
    if attack_type == 'maximize':
        # Make benign text toxic
        labels_to_maximize = [0, 1, 2, 3, 4, 5]  # All 6 toxic labels
        labels_to_minimize = []
        print_info("Goal: Maximize all toxic labels (make benign → toxic)")
    elif attack_type == 'minimize':
        # Make toxic text benign
        labels_to_maximize = []
        labels_to_minimize = [0, 1, 2, 3, 4, 5]
        print_info("Goal: Minimize all toxic labels (make toxic → benign)")
    else:
        raise ValueError(f"Invalid attack_type: {attack_type}")

    # Build appropriate recipe
    if recipe_type == 'target':
        attack = MultilabelACL23_recipe.build(
            model_wrapper=model_wrapper,
            labels_to_maximize=labels_to_maximize,
            labels_to_minimize=labels_to_minimize,
            maximize_target_score=0.5,
            minimize_target_score=0.5,
            wir_method=wir_method,
            pos_constraint=True,
            sbert_constraint=False
        )
    elif recipe_type == 'transform':
        attack = MultilabelACL23Transform.build(
            model_wrapper=model_wrapper,
            labels_to_maximize=labels_to_maximize,
            labels_to_minimize=labels_to_minimize,
            maximize_target_score=0.5,
            minimize_target_score=0.5,
            wir_method=wir_method,
            transform_method='glove',
            pos_constraint=True,
            sbert_constraint=False
        )
    else:
        raise ValueError(f"Invalid recipe_type: {recipe_type}")

    print_success("Attack recipe built successfully")
    print_info(f"  - Goal function: {type(attack.goal_function).__name__}")
    print_info(f"  - Search method: {type(attack.search_method).__name__}")
    print_info(f"  - Constraints: {len(attack.constraints)} active")
    print_info(f"  - Transformation: {type(attack.transformation).__name__}")

    return attack


def run_attack(attack, df_samples, attack_name):
    """Run the attack on samples."""
    print_header(f"Step 4: Running Attack - {attack_name}")

    # Create TextAttack dataset
    dataset = textattack.datasets.Dataset(
        [(text, labels) for text, labels in zip(df_samples['text'], df_samples['labels'])]
    )

    print_info(f"Dataset size: {len(dataset)} samples")

    # Configure attack
    attack_args = textattack.AttackArgs(
        num_examples=len(dataset),
        disable_stdout=False,
        random_seed=42
    )

    # Run attack
    print_info("Starting attack (this may take a few minutes)...")
    attacker = Attacker(attack, dataset, attack_args)
    attack_results = attacker.attack_dataset()

    print_success(f"Attack completed: {len(attack_results)} results")

    return attack_results


def analyze_results(attack_results, attack_name):
    """Analyze and display attack results."""
    print_header(f"Step 5: Analyzing Results - {attack_name}")

    # Categorize results
    successful = [r for r in attack_results if isinstance(r, textattack.attack_results.SuccessfulAttackResult)]
    failed = [r for r in attack_results if isinstance(r, textattack.attack_results.FailedAttackResult)]
    skipped = [r for r in attack_results if isinstance(r, textattack.attack_results.SkippedAttackResult)]

    total = len(attack_results)

    # Calculate statistics
    success_rate = len(successful) / total * 100 if total > 0 else 0

    print(f"\n{Colors.BOLD}Attack Statistics:{Colors.ENDC}")
    print(f"  Total attacks: {total}")
    print(f"  {Colors.OKGREEN}✓ Successful: {len(successful)} ({success_rate:.1f}%){Colors.ENDC}")
    print(f"  {Colors.FAIL}✗ Failed: {len(failed)} ({len(failed)/total*100:.1f}%){Colors.ENDC}")
    print(f"  {Colors.WARNING}⊘ Skipped: {len(skipped)} ({len(skipped)/total*100:.1f}%){Colors.ENDC}")

    if successful:
        # Analyze successful attacks
        num_queries = [r.num_queries for r in successful]
        words_changed = [r.original_result.attacked_text.words_diff_num(r.perturbed_result.attacked_text) for r in successful]

        print(f"\n{Colors.BOLD}Successful Attacks:{Colors.ENDC}")
        print(f"  Avg queries: {np.mean(num_queries):.1f} (min: {min(num_queries)}, max: {max(num_queries)})")
        print(f"  Avg words changed: {np.mean(words_changed):.1f} (min: {min(words_changed)}, max: {max(words_changed)})")

        # Show example
        if len(successful) > 0:
            example = successful[0]
            print(f"\n{Colors.BOLD}Example Successful Attack:{Colors.ENDC}")
            print(f"  Original: {example.original_result.attacked_text.text[:100]}...")
            print(f"  Perturbed: {example.perturbed_result.attacked_text.text[:100]}...")
            print(f"  Queries: {example.num_queries}")
            print(f"  Words changed: {example.original_result.attacked_text.words_diff_num(example.perturbed_result.attacked_text)}")

    return {
        'total': total,
        'successful': len(successful),
        'failed': len(failed),
        'skipped': len(skipped),
        'success_rate': success_rate
    }


def save_results(attack_results, df_samples, output_path, attack_name):
    """Save attack results to file."""
    print_header(f"Step 6: Saving Results - {attack_name}")

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log results to CSV
    logger = CSVLogger(color_method='html')
    for result in attack_results:
        logger.log_attack_result(result)

    df_results = logger.df

    # Add ground truth labels
    df_results['ground_truth_labels'] = df_samples['labels'].values[:len(df_results)]

    # Clean up text formatting
    df_results['perturbed_text_clean'] = df_results['perturbed_text'].replace(
        '<font color = .{1,6}>|</font>', '', regex=True
    )

    # Save to parquet (more efficient than CSV for arrays)
    output_file = Path(output_path).with_suffix('.parquet')
    df_results.to_parquet(output_file)
    print_success(f"Results saved to: {output_file}")

    # Also save summary
    summary_file = Path(output_path).with_suffix('.summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Attack Results Summary - {attack_name}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total samples: {len(attack_results)}\n")
        f.write(f"Successful: {len([r for r in attack_results if isinstance(r, textattack.attack_results.SuccessfulAttackResult)])}\n")
        f.write(f"Failed: {len([r for r in attack_results if isinstance(r, textattack.attack_results.FailedAttackResult)])}\n")
        f.write(f"Skipped: {len([r for r in attack_results if isinstance(r, textattack.attack_results.SkippedAttackResult)])}\n")

    print_success(f"Summary saved to: {summary_file}")

    return df_results


def run_full_workflow(args):
    """Run complete end-to-end workflow."""
    print_header("TextAttack-Multilabel End-to-End Demo")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results_summary = {}

    # Step 1: Prepare data
    df_benign, df_toxic = create_sample_data(num_samples=args.num_samples)

    # Step 2: Load model
    model_wrapper = load_model(model_type='detoxify')

    # Step 3-6: Run attacks on benign samples (maximize toxicity)
    if args.attack_benign:
        print_info("\n" + "="*80)
        print_info("ATTACKING BENIGN SAMPLES (Goal: Make them toxic)")
        print_info("="*80)

        attack_benign = build_attack(
            model_wrapper,
            attack_type='maximize',
            wir_method=args.wir_method,
            recipe_type=args.recipe_type
        )

        results_benign = run_attack(attack_benign, df_benign, "Benign → Toxic")
        stats_benign = analyze_results(results_benign, "Benign → Toxic")
        df_benign_results = save_results(
            results_benign,
            df_benign,
            f"results/attack_benign_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "Benign → Toxic"
        )

        results_summary['benign'] = stats_benign

    # Step 3-6: Run attacks on toxic samples (minimize toxicity)
    if args.attack_toxic:
        print_info("\n" + "="*80)
        print_info("ATTACKING TOXIC SAMPLES (Goal: Make them benign)")
        print_info("="*80)

        attack_toxic = build_attack(
            model_wrapper,
            attack_type='minimize',
            wir_method=args.wir_method,
            recipe_type=args.recipe_type
        )

        results_toxic = run_attack(attack_toxic, df_toxic, "Toxic → Benign")
        stats_toxic = analyze_results(results_toxic, "Toxic → Benign")
        df_toxic_results = save_results(
            results_toxic,
            df_toxic,
            f"results/attack_toxic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "Toxic → Benign"
        )

        results_summary['toxic'] = stats_toxic

    # Final summary
    print_header("Final Summary")
    print(f"\n{Colors.BOLD}Overall Results:{Colors.ENDC}")

    if 'benign' in results_summary:
        stats = results_summary['benign']
        print(f"\n{Colors.OKCYAN}Benign → Toxic Attack:{Colors.ENDC}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        print(f"  Successful: {stats['successful']}/{stats['total']}")

    if 'toxic' in results_summary:
        stats = results_summary['toxic']
        print(f"\n{Colors.OKCYAN}Toxic → Benign Attack:{Colors.ENDC}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        print(f"  Successful: {stats['successful']}/{stats['total']}")

    print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ End-to-end demo completed successfully!{Colors.ENDC}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete end-to-end demo of TextAttack-Multilabel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo with 5 samples
  python %(prog)s --quick

  # Full demo with custom parameters
  python %(prog)s --num-samples 20 --wir-method gradient --recipe-type transform

  # Attack only benign samples
  python %(prog)s --attack-benign --no-attack-toxic
        """
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick demo mode (5 samples, unk method)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to attack (default: 10)'
    )
    parser.add_argument(
        '--wir-method',
        type=str,
        default='unk',
        choices=['unk', 'delete', 'weighted-saliency', 'gradient', 'random'],
        help='Word importance ranking method (default: unk)'
    )
    parser.add_argument(
        '--recipe-type',
        type=str,
        default='target',
        choices=['target', 'transform'],
        help='Attack recipe type (default: target)'
    )
    parser.add_argument(
        '--attack-benign',
        action='store_true',
        default=True,
        help='Attack benign samples (default: True)'
    )
    parser.add_argument(
        '--no-attack-benign',
        action='store_false',
        dest='attack_benign',
        help='Skip attacking benign samples'
    )
    parser.add_argument(
        '--attack-toxic',
        action='store_true',
        default=True,
        help='Attack toxic samples (default: True)'
    )
    parser.add_argument(
        '--no-attack-toxic',
        action='store_false',
        dest='attack_toxic',
        help='Skip attacking toxic samples'
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.num_samples = 5
        args.wir_method = 'unk'

    try:
        run_full_workflow(args)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Demo interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
