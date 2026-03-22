"""Experiment 2: Effect of data duplication on memorization speed and forgetting.

Tests H2: Duplicated examples are memorized faster with fewer forgetting events,
and this hurts generalization.
"""
import sys
sys.path.insert(0, '/workspaces/llm-forget-better-claude/src')

import torch
import numpy as np
import json
import random
from pathlib import Path
from model import ModularArithmeticTransformer, create_modular_dataset
from train import train_with_forgetting_tracking

# Configuration
P = 97
OPERATION = 'add'
FRAC_TRAIN = 0.5
DUPLICATION_FACTORS = [1, 2, 4, 8]
SEEDS = [42, 123, 456]
LR = 1e-3
WEIGHT_DECAY = 1.0  # Use the best WD from exp1 or a reasonable default
EPOCHS = 50000
BATCH_SIZE = 512
EVAL_EVERY = 50
LOG_EVERY = 500
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DUP_FRACTION = 0.25  # Duplicate 25% of training data

RESULTS_DIR = Path('/workspaces/llm-forget-better-claude/results/exp2_duplication')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_duplicated_dataset(p, operation, frac_train, dup_factor, dup_fraction, seed):
    """Create dataset with controlled duplication of a subset."""
    train_x, train_y, test_x, test_y = create_modular_dataset(
        p=p, operation=operation, frac_train=frac_train, seed=seed
    )

    n_train = len(train_x)
    n_to_dup = int(n_train * dup_fraction)

    # Select examples to duplicate
    rng = random.Random(seed)
    dup_indices = list(range(n_train))
    rng.shuffle(dup_indices)
    dup_indices = dup_indices[:n_to_dup]

    # Create duplicated training set
    if dup_factor > 1:
        extra_x = train_x[dup_indices].repeat(dup_factor - 1, 1)
        extra_y = train_y[dup_indices].repeat(dup_factor - 1)
        train_x_dup = torch.cat([train_x, extra_x], dim=0)
        train_y_dup = torch.cat([train_y, extra_y], dim=0)
    else:
        train_x_dup = train_x
        train_y_dup = train_y

    # Track which original indices were duplicated
    is_duplicated = torch.zeros(n_train, dtype=torch.bool)
    is_duplicated[dup_indices] = True

    return train_x_dup, train_y_dup, test_x, test_y, is_duplicated, n_train


print(f"Device: {DEVICE}")
print(f"Testing duplication factors: {DUPLICATION_FACTORS}")
print("=" * 80)

all_results = {}

for dup_factor in DUPLICATION_FACTORS:
    all_results[str(dup_factor)] = []

    for seed in SEEDS:
        print(f"\n{'='*80}")
        print(f"Duplication Factor: {dup_factor}x, Seed: {seed}")
        print(f"{'='*80}")

        train_x, train_y, test_x, test_y, is_dup, n_orig = create_duplicated_dataset(
            P, OPERATION, FRAC_TRAIN, dup_factor, DUP_FRACTION, seed
        )
        print(f"Original train: {n_orig}, After duplication: {len(train_x)}, Test: {len(test_x)}")

        model = ModularArithmeticTransformer(
            p=P, d_model=128, n_heads=4, d_ff=512, dropout=0.0
        )

        results = train_with_forgetting_tracking(
            model, train_x, train_y, test_x, test_y,
            lr=LR, weight_decay=WEIGHT_DECAY, epochs=EPOCHS,
            batch_size=BATCH_SIZE, device=DEVICE,
            log_every=LOG_EVERY, eval_every=EVAL_EVERY,
            seed=seed, early_stop_test_acc=0.99,
            patience=3000,
        )

        # Analyze forgetting events for duplicated vs non-duplicated (original indices only)
        fe = np.array(results['forgetting_events_per_example'][:n_orig])
        dup_mask = is_dup.numpy()

        dup_fe = fe[dup_mask]
        nondup_fe = fe[~dup_mask]

        summary = {
            'dup_factor': dup_factor,
            'seed': seed,
            'n_train_total': len(train_x),
            'n_train_original': n_orig,
            'grokked': results['grokked'],
            'grok_step': results['grok_step'],
            'total_steps': results['total_steps'],
            'elapsed_seconds': results['elapsed_seconds'],
            'best_test_acc': results['best_test_acc'],
            'final_test_acc': results['final_test_acc'],
            'duplicated_examples': {
                'count': int(dup_mask.sum()),
                'mean_forgetting_events': float(dup_fe.mean()) if len(dup_fe) > 0 else 0,
                'std_forgetting_events': float(dup_fe.std()) if len(dup_fe) > 0 else 0,
            },
            'non_duplicated_examples': {
                'count': int((~dup_mask).sum()),
                'mean_forgetting_events': float(nondup_fe.mean()) if len(nondup_fe) > 0 else 0,
                'std_forgetting_events': float(nondup_fe.std()) if len(nondup_fe) > 0 else 0,
            },
            'overall_forgetting_stats': results['forgetting_stats'],
        }
        all_results[str(dup_factor)].append(summary)

        # Save history
        history_path = RESULTS_DIR / f"history_dup{dup_factor}_seed{seed}.json"
        with open(history_path, 'w') as f:
            json.dump(results['history'], f)

        print(f"  Grokked: {results['grokked']} at step {results['grok_step']}")
        print(f"  Best test acc: {results['best_test_acc']:.4f}")
        print(f"  Dup examples mean FE: {summary['duplicated_examples']['mean_forgetting_events']:.2f}")
        print(f"  Non-dup examples mean FE: {summary['non_duplicated_examples']['mean_forgetting_events']:.2f}")

# Save all results
with open(RESULTS_DIR / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Print summary
print("\n" + "=" * 80)
print("EXPERIMENT 2 SUMMARY")
print("=" * 80)
print(f"{'Dup':>5} | {'Grok Rate':>10} | {'Avg Grok Step':>14} | {'Dup FE':>8} | {'Non-Dup FE':>10} | {'Test Acc':>8}")
print("-" * 80)

for df in DUPLICATION_FACTORS:
    runs = all_results[str(df)]
    grok_rate = sum(1 for r in runs if r['grokked']) / len(runs)
    grok_steps = [r['grok_step'] for r in runs if r['grokked']]
    avg_grok = np.mean(grok_steps) if grok_steps else float('inf')
    avg_dup_fe = np.mean([r['duplicated_examples']['mean_forgetting_events'] for r in runs])
    avg_nondup_fe = np.mean([r['non_duplicated_examples']['mean_forgetting_events'] for r in runs])
    avg_test = np.mean([r['best_test_acc'] for r in runs])
    print(f"{df:>5}x | {grok_rate:>10.2f} | {avg_grok:>14.0f} | {avg_dup_fe:>8.2f} | {avg_nondup_fe:>10.2f} | {avg_test:>8.4f}")

print("\nExperiment 2 complete!")
