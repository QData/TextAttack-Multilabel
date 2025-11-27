import numpy as np
import pandas as pd
import textattack
import os
import time
import random
import re
from pathlib import Path

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


def analyze_results(attack_results, attack_name):
    """Analyze and display attack results."""
    print_header(f"Step: Analyzing Results - {attack_name}")

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



def display_attack_results(results_df, attack_direction, stats=None):
    """Display attack results in a human-readable format."""
    print("\n" + "="*70)
    print(f"🎯 MULTILABEL ADVERSARIAL ATTACK RESULTS - {attack_direction.upper()}")
    print("="*70)

    if results_df.empty:
        print("❌ No attack results found.")
        return

    # Use stats if provided, otherwise fall back to counting non-null perturbed_text
    if stats:
        total_samples = stats['total']
        successful = stats['successful']
        success_rate = stats['success_rate']
    else:
        successful = len(results_df[results_df['perturbed_text'].notna()])
        total_samples = len(results_df)
        success_rate = successful / total_samples * 100 if total_samples > 0 else 0

    print(f"📊 Summary:")
    print(f"   Total samples: {total_samples}")
    print(f"   Successful attacks: {successful}")
    print(f"   Success rate: {success_rate:.1f}%")

    if successful > 0:
        # Calculate average toxicity reduction
        valid_results = results_df[results_df['perturbed_text'].notna()]
        original_scores = np.array(valid_results['original_output'].tolist())
        perturbed_scores = np.array(valid_results['perturbed_output'].tolist())

        # Calculate how much toxicity changed (direction depends on attack type)
        if attack_direction.lower().startswith('benign'):
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
        print(f"  Original output scores: {row.original_output}")
        print(f"  Perturbed output scores: {row.perturbed_output}")

        if pd.notna(row.perturbed_text):
            # Show original text without highlighting (plain)
            original_text_clean = re.sub(r'<font[^>]*>', '', str(row.iloc[0]))
            original_text_clean = re.sub(r'</font>', '', original_text_clean)[:80]
            print(f"  Original: {original_text_clean}{('...' if len(str(row.iloc[0])) > 80 else '')}")

            # Show perturbed text with color differentiation: original words in blue, changed words in red
            def replace_font_tag(match):
                color = match.group(1).lower()
                if color == "red":
                    return Colors.FAIL  # Red for perturbed/changed words
                elif color == "blue":
                    return Colors.OKBLUE  # Blue for original highlighting (if any, but shouldn't be in perturbed)
                else:
                    return Colors.BOLD  # Fallback

            pert_text = Colors.OKBLUE  # Start with blue for original words
            pert_text += re.sub(r'<font color = ([^>]+)>', replace_font_tag, str(row.perturbed_text))
            pert_text = re.sub(r'</font>', f'{Colors.ENDC}{Colors.OKBLUE}', pert_text)  # Reset to blue after highlight
            pert_text += Colors.ENDC  # Reset at end
            pert_text = pert_text.replace('<SPLIT>', ' ')
            print(f"  Perturbed: {pert_text[:80]}{'...' if len(str(pert_text)) > 80 else ''}")

            # Show score changes (original output might be in different columns)
            if 'original_output' in row.index:
                print(".3f")
                print(".3f")
        else:
            print(f"  Original: {row.iloc[0][:80]}{'...' if len(str(row.iloc[0])) > 80 else ''}")
            print("  Perturbed: N/A")
            print("  ✗ FAILED ATTACK")

    print("\n" + "="*70)
    dirname = f"attack_{attack_direction.lower().replace(' ', '_').replace('→', 'to')}"
    print(f"💾 Results saved to: {Path('results') / dirname}")
