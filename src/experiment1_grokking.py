"""Experiment 1: Effect of forgetting pressure (weight decay) on grokking.

Tests H1: Higher forgetting pressure leads to faster grokking and more forgetting events.
"""
import sys
sys.path.insert(0, '/workspaces/llm-forget-better-claude/src')

import torch
import numpy as np
import json
from pathlib import Path
from model import ModularArithmeticTransformer, create_modular_dataset
from train import train_with_forgetting_tracking

# Configuration
P = 97
OPERATION = 'add'
FRAC_TRAIN = 0.5
WEIGHT_DECAYS = [0.0, 0.001, 0.01, 0.1, 1.0]
SEEDS = [42, 123, 456]
LR = 1e-3
EPOCHS = 50000
BATCH_SIZE = 512
EVAL_EVERY = 50
LOG_EVERY = 500
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RESULTS_DIR = Path('/workspaces/llm-forget-better-claude/results/exp1_grokking')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Testing weight decays: {WEIGHT_DECAYS}")
print(f"Seeds: {SEEDS}")
print("=" * 80)

all_results = {}

for wd in WEIGHT_DECAYS:
    all_results[str(wd)] = []

    for seed in SEEDS:
        print(f"\n{'='*80}")
        print(f"Weight Decay: {wd}, Seed: {seed}")
        print(f"{'='*80}")

        # Create dataset
        train_x, train_y, test_x, test_y = create_modular_dataset(
            p=P, operation=OPERATION, frac_train=FRAC_TRAIN, seed=seed
        )
        print(f"Train: {len(train_x)}, Test: {len(test_x)}")

        # Create model
        model = ModularArithmeticTransformer(
            p=P, d_model=128, n_heads=4, d_ff=512, dropout=0.0
        )

        # Train
        results = train_with_forgetting_tracking(
            model, train_x, train_y, test_x, test_y,
            lr=LR, weight_decay=wd, epochs=EPOCHS,
            batch_size=BATCH_SIZE, device=DEVICE,
            log_every=LOG_EVERY, eval_every=EVAL_EVERY,
            seed=seed, early_stop_test_acc=0.99,
            patience=3000,
        )

        # Save individual result
        summary = {
            'weight_decay': wd,
            'seed': seed,
            'grokked': results['grokked'],
            'grok_step': results['grok_step'],
            'total_steps': results['total_steps'],
            'elapsed_seconds': results['elapsed_seconds'],
            'best_test_acc': results['best_test_acc'],
            'final_train_acc': results['final_train_acc'],
            'final_test_acc': results['final_test_acc'],
            'forgetting_stats': results['forgetting_stats'],
        }
        all_results[str(wd)].append(summary)

        # Save history for plotting
        history_path = RESULTS_DIR / f"history_wd{wd}_seed{seed}.json"
        with open(history_path, 'w') as f:
            json.dump(results['history'], f)

        print(f"  Grokked: {results['grokked']} at step {results['grok_step']}")
        print(f"  Best test acc: {results['best_test_acc']:.4f}")
        print(f"  Mean forgetting events: {results['forgetting_stats']['mean']:.2f}")
        print(f"  Unforgettable fraction: {results['forgetting_stats']['unforgettable_frac']:.4f}")

# Save all results
with open(RESULTS_DIR / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Print summary table
print("\n" + "=" * 80)
print("EXPERIMENT 1 SUMMARY")
print("=" * 80)
print(f"{'WD':>8} | {'Grok Rate':>10} | {'Avg Grok Step':>14} | {'Avg Forget Events':>18} | {'Avg Test Acc':>12}")
print("-" * 80)

for wd in WEIGHT_DECAYS:
    runs = all_results[str(wd)]
    grok_rate = sum(1 for r in runs if r['grokked']) / len(runs)
    grok_steps = [r['grok_step'] for r in runs if r['grokked']]
    avg_grok = np.mean(grok_steps) if grok_steps else float('inf')
    avg_forget = np.mean([r['forgetting_stats']['mean'] for r in runs])
    avg_test = np.mean([r['best_test_acc'] for r in runs])
    print(f"{wd:>8.3f} | {grok_rate:>10.2f} | {avg_grok:>14.0f} | {avg_forget:>18.2f} | {avg_test:>12.4f}")

print("\nExperiment 1 complete!")
