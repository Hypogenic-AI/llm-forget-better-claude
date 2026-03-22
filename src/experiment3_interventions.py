"""Experiment 3: Forgetting interventions and data efficiency.

Tests H3: Explicit forgetting mechanisms improve data efficiency.
Compares: baseline, dropout, noise injection, periodic weight perturbation.
"""
import sys
sys.path.insert(0, '/workspaces/llm-forget-better-claude/src')

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from model import ModularArithmeticTransformer, create_modular_dataset

P = 97
OPERATION = 'add'
FRAC_TRAIN = 0.5
SEEDS = [42, 123, 456]
LR = 1e-3
WEIGHT_DECAY = 1.0
EPOCHS = 50000
BATCH_SIZE = 512
EVAL_EVERY = 50
LOG_EVERY = 500
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RESULTS_DIR = Path('/workspaces/llm-forget-better-claude/results/exp3_interventions')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def add_noise_to_weights(model, noise_std=0.01):
    """Add Gaussian noise to all model weights."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * noise_std)


def partial_weight_reset(model, reset_fraction=0.1, seed=None):
    """Reset a fraction of weights to initial values."""
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            mask = torch.rand_like(p) < reset_fraction
            p[mask] = torch.randn_like(p[mask]) * 0.02


def train_with_intervention(
    model, train_x, train_y, test_x, test_y,
    intervention='none', intervention_kwargs=None,
    lr=1e-3, weight_decay=0.0, epochs=10000,
    batch_size=512, device='cuda',
    log_every=500, eval_every=50,
    seed=42, early_stop_test_acc=0.99, patience=3000,
):
    """Train with various forgetting interventions."""
    if intervention_kwargs is None:
        intervention_kwargs = {}

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    n_train = len(train_x)
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'steps': []}

    step = 0
    grokked = False
    grok_step = None
    best_test_acc = 0.0
    start_time = time.time()

    # Intervention schedule
    noise_every = intervention_kwargs.get('noise_every', 100)
    noise_std = intervention_kwargs.get('noise_std', 0.01)
    reset_every = intervention_kwargs.get('reset_every', 500)
    reset_fraction = intervention_kwargs.get('reset_fraction', 0.1)

    for epoch in range(epochs):
        perm = torch.randperm(n_train, device=device)

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = train_x[idx]
            batch_y = train_y[idx]

            model.train()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            # Apply intervention
            if intervention == 'noise' and step % noise_every == 0:
                add_noise_to_weights(model, noise_std=noise_std)
            elif intervention == 'reset' and step % reset_every == 0:
                partial_weight_reset(model, reset_fraction=reset_fraction, seed=seed + step)

            # Evaluate
            if step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    train_logits = model(train_x)
                    train_loss = criterion(train_logits, train_y).item()
                    train_acc = (train_logits.argmax(-1) == train_y).float().mean().item()

                    test_logits = model(test_x)
                    test_loss = criterion(test_logits, test_y).item()
                    test_acc = (test_logits.argmax(-1) == test_y).float().mean().item()

                history['train_loss'].append(train_loss)
                history['test_loss'].append(test_loss)
                history['train_acc'].append(train_acc)
                history['test_acc'].append(test_acc)
                history['steps'].append(step)

                if test_acc > best_test_acc:
                    best_test_acc = test_acc

                if test_acc >= early_stop_test_acc and not grokked:
                    grokked = True
                    grok_step = step

                if step % log_every == 0:
                    elapsed = time.time() - start_time
                    print(f"  Step {step:6d} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | {elapsed:.1f}s")

                if grokked and step - grok_step > patience:
                    break

        if grokked and step - grok_step > patience:
            break

    elapsed = time.time() - start_time

    # Compute data efficiency: area under test accuracy curve
    steps_arr = np.array(history['steps'])
    test_acc_arr = np.array(history['test_acc'])
    if len(steps_arr) > 1:
        from scipy.integrate import trapezoid
        auc = trapezoid(test_acc_arr, steps_arr) / (steps_arr[-1] - steps_arr[0])
    else:
        auc = 0.0

    return {
        'history': history,
        'grokked': grokked,
        'grok_step': grok_step,
        'total_steps': step,
        'elapsed_seconds': elapsed,
        'best_test_acc': best_test_acc,
        'final_test_acc': history['test_acc'][-1] if history['test_acc'] else 0,
        'data_efficiency_auc': auc,
    }


# Define interventions
INTERVENTIONS = {
    'baseline': {'intervention': 'none', 'kwargs': {}},
    'dropout_0.1': {'intervention': 'none', 'kwargs': {}, 'dropout': 0.1},
    'dropout_0.3': {'intervention': 'none', 'kwargs': {}, 'dropout': 0.3},
    'noise_small': {'intervention': 'noise', 'kwargs': {'noise_every': 100, 'noise_std': 0.005}},
    'noise_large': {'intervention': 'noise', 'kwargs': {'noise_every': 100, 'noise_std': 0.02}},
    'reset_gentle': {'intervention': 'reset', 'kwargs': {'reset_every': 500, 'reset_fraction': 0.05}},
    'reset_aggressive': {'intervention': 'reset', 'kwargs': {'reset_every': 500, 'reset_fraction': 0.15}},
}

print(f"Device: {DEVICE}")
print(f"Testing interventions: {list(INTERVENTIONS.keys())}")
print("=" * 80)

all_results = {}

for name, config in INTERVENTIONS.items():
    all_results[name] = []

    for seed in SEEDS:
        print(f"\n{'='*80}")
        print(f"Intervention: {name}, Seed: {seed}")
        print(f"{'='*80}")

        train_x, train_y, test_x, test_y = create_modular_dataset(
            p=P, operation=OPERATION, frac_train=FRAC_TRAIN, seed=seed
        )

        dropout = config.get('dropout', 0.0)
        model = ModularArithmeticTransformer(
            p=P, d_model=128, n_heads=4, d_ff=512, dropout=dropout
        )

        results = train_with_intervention(
            model, train_x, train_y, test_x, test_y,
            intervention=config['intervention'],
            intervention_kwargs=config['kwargs'],
            lr=LR, weight_decay=WEIGHT_DECAY, epochs=EPOCHS,
            batch_size=BATCH_SIZE, device=DEVICE,
            log_every=LOG_EVERY, eval_every=EVAL_EVERY,
            seed=seed, early_stop_test_acc=0.99, patience=3000,
        )

        summary = {
            'intervention': name,
            'seed': seed,
            'grokked': results['grokked'],
            'grok_step': results['grok_step'],
            'total_steps': results['total_steps'],
            'elapsed_seconds': results['elapsed_seconds'],
            'best_test_acc': results['best_test_acc'],
            'final_test_acc': results['final_test_acc'],
            'data_efficiency_auc': results['data_efficiency_auc'],
        }
        all_results[name].append(summary)

        # Save history
        history_path = RESULTS_DIR / f"history_{name}_seed{seed}.json"
        with open(history_path, 'w') as f:
            json.dump(results['history'], f)

        print(f"  Grokked: {results['grokked']} at step {results['grok_step']}")
        print(f"  Best test acc: {results['best_test_acc']:.4f}")
        print(f"  Data efficiency AUC: {results['data_efficiency_auc']:.4f}")

# Save all
with open(RESULTS_DIR / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Summary
print("\n" + "=" * 80)
print("EXPERIMENT 3 SUMMARY")
print("=" * 80)
print(f"{'Intervention':>20} | {'Grok Rate':>10} | {'Avg Grok Step':>14} | {'Avg AUC':>10} | {'Avg Test Acc':>12}")
print("-" * 80)

for name in INTERVENTIONS:
    runs = all_results[name]
    grok_rate = sum(1 for r in runs if r['grokked']) / len(runs)
    grok_steps = [r['grok_step'] for r in runs if r['grokked']]
    avg_grok = np.mean(grok_steps) if grok_steps else float('inf')
    avg_auc = np.mean([r['data_efficiency_auc'] for r in runs])
    avg_test = np.mean([r['best_test_acc'] for r in runs])
    print(f"{name:>20} | {grok_rate:>10.2f} | {avg_grok:>14.0f} | {avg_auc:>10.4f} | {avg_test:>12.4f}")

print("\nExperiment 3 complete!")
