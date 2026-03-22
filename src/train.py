"""Training loop with per-example forgetting event tracking."""
import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path


def train_with_forgetting_tracking(
    model, train_x, train_y, test_x, test_y,
    lr=1e-3, weight_decay=0.0, epochs=10000,
    batch_size=512, device='cuda',
    log_every=100, eval_every=100,
    seed=42, early_stop_test_acc=0.99,
    patience=2000,  # stop if no improvement for this many steps after grokking starts
):
    """Train model and track per-example forgetting events.

    A forgetting event occurs when an example transitions from correctly
    classified to incorrectly classified between consecutive evaluations.

    Returns dict with training history and forgetting statistics.
    """
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
    n_test = len(test_x)

    # Per-example tracking
    prev_correct_train = torch.zeros(n_train, dtype=torch.bool, device=device)
    forgetting_events = torch.zeros(n_train, dtype=torch.long, device=device)
    learning_events = torch.zeros(n_train, dtype=torch.long, device=device)

    # History
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
        'steps': [],
        'forgetting_events_total': [],
        'forgetting_events_mean': [],
    }

    step = 0
    grokked = False
    grok_step = None
    best_test_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        # Shuffle training data
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

            # Evaluate and track forgetting events
            if step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # Full train eval
                    train_logits = model(train_x)
                    train_loss = criterion(train_logits, train_y).item()
                    train_preds = train_logits.argmax(dim=-1)
                    curr_correct_train = (train_preds == train_y)
                    train_acc = curr_correct_train.float().mean().item()

                    # Track forgetting events: was correct, now wrong
                    forgot = prev_correct_train & ~curr_correct_train
                    learned = ~prev_correct_train & curr_correct_train
                    forgetting_events += forgot.long()
                    learning_events += learned.long()
                    prev_correct_train = curr_correct_train.clone()

                    # Full test eval
                    test_logits = model(test_x)
                    test_loss = criterion(test_logits, test_y).item()
                    test_preds = test_logits.argmax(dim=-1)
                    test_acc = (test_preds == test_y).float().mean().item()

                history['train_loss'].append(train_loss)
                history['test_loss'].append(test_loss)
                history['train_acc'].append(train_acc)
                history['test_acc'].append(test_acc)
                history['steps'].append(step)
                history['forgetting_events_total'].append(forgetting_events.sum().item())
                history['forgetting_events_mean'].append(forgetting_events.float().mean().item())

                if test_acc > best_test_acc:
                    best_test_acc = test_acc

                if test_acc >= early_stop_test_acc and not grokked:
                    grokked = True
                    grok_step = step

                if step % log_every == 0:
                    elapsed = time.time() - start_time
                    print(f"Step {step:6d} | Train: {train_acc:.4f} ({train_loss:.4f}) | "
                          f"Test: {test_acc:.4f} ({test_loss:.4f}) | "
                          f"Forget events: {forgetting_events.sum().item():6d} | "
                          f"Time: {elapsed:.1f}s")

                # Early stopping: if grokked and no improvement
                if grokked and step - grok_step > patience:
                    break

        if grokked and step - grok_step > patience:
            break

    elapsed = time.time() - start_time

    # Final forgetting statistics
    fe_np = forgetting_events.cpu().numpy()
    le_np = learning_events.cpu().numpy()

    results = {
        'history': history,
        'grokked': grokked,
        'grok_step': grok_step,
        'total_steps': step,
        'elapsed_seconds': elapsed,
        'best_test_acc': best_test_acc,
        'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
        'final_test_acc': history['test_acc'][-1] if history['test_acc'] else 0,
        'forgetting_stats': {
            'mean': float(fe_np.mean()),
            'std': float(fe_np.std()),
            'max': int(fe_np.max()),
            'min': int(fe_np.min()),
            'median': float(np.median(fe_np)),
            'unforgettable_frac': float((fe_np == 0).mean()),
            'distribution': np.bincount(fe_np).tolist(),
        },
        'learning_stats': {
            'mean': float(le_np.mean()),
            'std': float(le_np.std()),
        },
        'forgetting_events_per_example': fe_np.tolist(),
    }

    return results
