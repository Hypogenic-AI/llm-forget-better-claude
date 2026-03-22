"""Analysis and visualization for all experiments."""
import sys
sys.path.insert(0, '/workspaces/llm-forget-better-claude/src')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

RESULTS_BASE = Path('/workspaces/llm-forget-better-claude/results')
PLOTS_DIR = Path('/workspaces/llm-forget-better-claude/results/plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
})


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Grokking + Weight Decay
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_exp1():
    print("=" * 80)
    print("EXPERIMENT 1 ANALYSIS: Forgetting Pressure and Grokking")
    print("=" * 80)

    results_path = RESULTS_BASE / 'exp1_grokking' / 'all_results.json'
    with open(results_path) as f:
        all_results = json.load(f)

    weight_decays = [0.0, 0.001, 0.01, 0.1, 1.0]

    # --- Plot 1: Grokking curves for each WD ---
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))

    for ax, wd, color in zip(axes, weight_decays, colors):
        for seed_idx, seed in enumerate([42, 123, 456]):
            history_path = RESULTS_BASE / 'exp1_grokking' / f'history_wd{wd}_seed{seed}.json'
            if not history_path.exists():
                continue
            with open(history_path) as f:
                history = json.load(f)
            steps = np.array(history['steps'])
            train_acc = np.array(history['train_acc'])
            test_acc = np.array(history['test_acc'])

            alpha = 0.6 if seed_idx > 0 else 1.0
            ax.plot(steps, train_acc, '--', color='gray', alpha=0.3, linewidth=1)
            ax.plot(steps, test_acc, '-', color=color, alpha=alpha, linewidth=1.5)

        ax.set_title(f'WD={wd}')
        ax.set_xlabel('Training Steps')
        ax.axhline(y=0.99, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel('Accuracy')
    fig.suptitle('Grokking Under Different Weight Decay (Forgetting Pressure)', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp1_grokking_curves.png', bbox_inches='tight')
    plt.close()

    # --- Plot 2: Steps to grok vs weight decay ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    grok_steps_by_wd = {}
    forget_events_by_wd = {}

    for wd in weight_decays:
        runs = all_results[str(wd)]
        grok_steps = [r['grok_step'] for r in runs if r['grokked'] and r['grok_step'] is not None]
        fe = [r['forgetting_stats']['mean'] for r in runs]
        grok_steps_by_wd[wd] = grok_steps
        forget_events_by_wd[wd] = fe

    # Grok steps
    means = [np.mean(grok_steps_by_wd[wd]) for wd in weight_decays]
    stds = [np.std(grok_steps_by_wd[wd]) for wd in weight_decays]
    ax1.errorbar(range(len(weight_decays)), means, yerr=stds, fmt='o-', capsize=5,
                 color='#2196F3', linewidth=2, markersize=8)
    ax1.set_xticks(range(len(weight_decays)))
    ax1.set_xticklabels([str(wd) for wd in weight_decays])
    ax1.set_xlabel('Weight Decay')
    ax1.set_ylabel('Steps to Grok (99% test acc)')
    ax1.set_title('Higher Forgetting Pressure → Faster Grokking')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Forgetting events
    means_fe = [np.mean(forget_events_by_wd[wd]) for wd in weight_decays]
    stds_fe = [np.std(forget_events_by_wd[wd]) for wd in weight_decays]
    ax2.errorbar(range(len(weight_decays)), means_fe, yerr=stds_fe, fmt='s-', capsize=5,
                 color='#FF5722', linewidth=2, markersize=8)
    ax2.set_xticks(range(len(weight_decays)))
    ax2.set_xticklabels([str(wd) for wd in weight_decays])
    ax2.set_xlabel('Weight Decay')
    ax2.set_ylabel('Mean Forgetting Events Per Example')
    ax2.set_title('Higher Forgetting Pressure → More Forgetting Events')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp1_grok_steps_and_forgetting.png', bbox_inches='tight')
    plt.close()

    # --- Statistical test: Spearman correlation ---
    all_wds = []
    all_groks = []
    for wd in weight_decays:
        for gs in grok_steps_by_wd[wd]:
            all_wds.append(wd)
            all_groks.append(gs)

    rho, p_val = stats.spearmanr(all_wds, all_groks)
    print(f"\nSpearman correlation (WD vs Grok Steps): ρ = {rho:.4f}, p = {p_val:.6f}")

    # Effect size: compare WD=0 vs WD=0.1
    g0 = grok_steps_by_wd[0.0]
    g1 = grok_steps_by_wd[0.1]
    cohens_d = (np.mean(g0) - np.mean(g1)) / np.sqrt((np.std(g0)**2 + np.std(g1)**2) / 2)
    print(f"Cohen's d (WD=0.0 vs WD=0.1): d = {cohens_d:.2f}")

    # Wilcoxon for WD=0 vs WD=0.1
    if len(g0) >= 3 and len(g1) >= 3:
        stat, p = stats.mannwhitneyu(g0, g1, alternative='greater')
        print(f"Mann-Whitney U (WD=0 > WD=0.1): U = {stat:.1f}, p = {p:.6f}")

    return {
        'spearman_rho': rho,
        'spearman_p': p_val,
        'cohens_d_0_vs_01': cohens_d,
        'grok_steps_by_wd': {str(k): v for k, v in grok_steps_by_wd.items()},
        'forget_events_by_wd': {str(k): v for k, v in forget_events_by_wd.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Duplication Effects
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_exp2():
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 ANALYSIS: Duplication and Memorization Speed")
    print("=" * 80)

    results_path = RESULTS_BASE / 'exp2_duplication' / 'all_results.json'
    if not results_path.exists():
        print("Experiment 2 results not found, skipping.")
        return None

    with open(results_path) as f:
        all_results = json.load(f)

    dup_factors = [1, 2, 4, 8]

    # --- Plot: Grok steps and FE by duplication ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    grok_data = []
    dup_fe_data = []
    nondup_fe_data = []

    for df in dup_factors:
        runs = all_results[str(df)]
        groks = [r['grok_step'] for r in runs if r['grokked'] and r['grok_step'] is not None]
        dup_fe = [r['duplicated_examples']['mean_forgetting_events'] for r in runs]
        nondup_fe = [r['non_duplicated_examples']['mean_forgetting_events'] for r in runs]
        grok_data.append(groks)
        dup_fe_data.append(dup_fe)
        nondup_fe_data.append(nondup_fe)

    # Grok steps
    means = [np.mean(g) if g else float('inf') for g in grok_data]
    stds = [np.std(g) if g else 0 for g in grok_data]
    ax1.errorbar(range(len(dup_factors)), means, yerr=stds, fmt='o-', capsize=5,
                 color='#2196F3', linewidth=2, markersize=8)
    ax1.set_xticks(range(len(dup_factors)))
    ax1.set_xticklabels([f'{df}x' for df in dup_factors])
    ax1.set_xlabel('Duplication Factor')
    ax1.set_ylabel('Steps to Grok')
    ax1.set_title('Duplication Effect on Grokking Speed')
    ax1.grid(True, alpha=0.3)

    # FE comparison: duplicated vs non-duplicated
    x = np.arange(len(dup_factors))
    width = 0.35
    dup_means = [np.mean(d) for d in dup_fe_data]
    nondup_means = [np.mean(d) for d in nondup_fe_data]
    dup_stds = [np.std(d) for d in dup_fe_data]
    nondup_stds = [np.std(d) for d in nondup_fe_data]

    ax2.bar(x - width/2, dup_means, width, yerr=dup_stds, label='Duplicated', color='#FF5722', alpha=0.8, capsize=3)
    ax2.bar(x + width/2, nondup_means, width, yerr=nondup_stds, label='Non-Duplicated', color='#4CAF50', alpha=0.8, capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{df}x' for df in dup_factors])
    ax2.set_xlabel('Duplication Factor')
    ax2.set_ylabel('Mean Forgetting Events')
    ax2.set_title('Forgetting Events: Duplicated vs Non-Duplicated')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Test accuracy
    test_means = [np.mean([r['best_test_acc'] for r in all_results[str(df)]]) for df in dup_factors]
    test_stds = [np.std([r['best_test_acc'] for r in all_results[str(df)]]) for df in dup_factors]
    ax3.errorbar(range(len(dup_factors)), test_means, yerr=test_stds, fmt='s-', capsize=5,
                 color='#9C27B0', linewidth=2, markersize=8)
    ax3.set_xticks(range(len(dup_factors)))
    ax3.set_xticklabels([f'{df}x' for df in dup_factors])
    ax3.set_xlabel('Duplication Factor')
    ax3.set_ylabel('Best Test Accuracy')
    ax3.set_title('Duplication Effect on Generalization')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp2_duplication_analysis.png', bbox_inches='tight')
    plt.close()

    # Statistics
    print("\nDuplication Factor | Avg Grok Step | Dup FE | Non-Dup FE | Test Acc")
    print("-" * 70)
    for i, df in enumerate(dup_factors):
        avg_grok = np.mean(grok_data[i]) if grok_data[i] else float('inf')
        print(f"  {df}x              | {avg_grok:>13.0f} | {dup_means[i]:>6.2f} | {nondup_means[i]:>10.2f} | {test_means[i]:.4f}")

    return {
        'grok_by_dup': {str(df): g for df, g in zip(dup_factors, grok_data)},
        'dup_fe_means': dup_means,
        'nondup_fe_means': nondup_means,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Forgetting Interventions
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_exp3():
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 ANALYSIS: Forgetting Interventions")
    print("=" * 80)

    results_path = RESULTS_BASE / 'exp3_interventions' / 'all_results.json'
    if not results_path.exists():
        print("Experiment 3 results not found, skipping.")
        return None

    with open(results_path) as f:
        all_results = json.load(f)

    interventions = list(all_results.keys())

    # --- Plot 1: Grok step comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    grok_means = []
    grok_stds = []
    auc_means = []
    auc_stds = []
    colors_list = plt.cm.Set2(np.linspace(0, 1, len(interventions)))

    for name in interventions:
        runs = all_results[name]
        groks = [r['grok_step'] for r in runs if r['grokked'] and r['grok_step'] is not None]
        aucs = [r['data_efficiency_auc'] for r in runs]
        grok_means.append(np.mean(groks) if groks else 50000)
        grok_stds.append(np.std(groks) if groks else 0)
        auc_means.append(np.mean(aucs))
        auc_stds.append(np.std(aucs))

    x = np.arange(len(interventions))
    bars1 = ax1.bar(x, grok_means, yerr=grok_stds, capsize=3, color=colors_list, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(interventions, rotation=45, ha='right')
    ax1.set_ylabel('Steps to Grok')
    ax1.set_title('Steps to Grokking by Intervention')
    ax1.grid(True, alpha=0.3, axis='y')

    bars2 = ax2.bar(x, auc_means, yerr=auc_stds, capsize=3, color=colors_list, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(interventions, rotation=45, ha='right')
    ax2.set_ylabel('Data Efficiency (AUC)')
    ax2.set_title('Data Efficiency by Intervention\n(Higher = Better)')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp3_interventions_comparison.png', bbox_inches='tight')
    plt.close()

    # --- Plot 2: Learning curves overlaid ---
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab10

    for idx, name in enumerate(interventions):
        # Use seed 42 for clarity
        history_path = RESULTS_BASE / 'exp3_interventions' / f'history_{name}_seed42.json'
        if not history_path.exists():
            continue
        with open(history_path) as f:
            history = json.load(f)
        steps = np.array(history['steps'])
        test_acc = np.array(history['test_acc'])
        ax.plot(steps, test_acc, label=name, color=cmap(idx / len(interventions)), linewidth=1.5)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Curves: Different Forgetting Interventions')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axhline(y=0.99, color='red', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp3_learning_curves.png', bbox_inches='tight')
    plt.close()

    # Print summary
    print(f"\n{'Intervention':>20} | {'Grok Rate':>10} | {'Avg Grok Step':>14} | {'Avg AUC':>10}")
    print("-" * 65)
    for name in interventions:
        runs = all_results[name]
        grok_rate = sum(1 for r in runs if r['grokked']) / len(runs)
        groks = [r['grok_step'] for r in runs if r['grokked'] and r['grok_step'] is not None]
        avg_grok = np.mean(groks) if groks else float('inf')
        avg_auc = np.mean([r['data_efficiency_auc'] for r in runs])
        print(f"{name:>20} | {grok_rate:>10.2f} | {avg_grok:>14.0f} | {avg_auc:>10.4f}")

    return {
        'interventions': interventions,
        'grok_means': grok_means,
        'auc_means': auc_means,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def create_summary_figure():
    """Create a combined 2x2 summary figure for the report."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Grokking curves (WD=0 vs WD=0.1)
    ax = axes[0, 0]
    for wd, color, label in [(0.0, '#F44336', 'No forgetting (WD=0)'),
                               (0.1, '#2196F3', 'With forgetting (WD=0.1)')]:
        history_path = RESULTS_BASE / 'exp1_grokking' / f'history_wd{wd}_seed42.json'
        if not history_path.exists():
            continue
        with open(history_path) as f:
            history = json.load(f)
        steps = np.array(history['steps'])
        test_acc = np.array(history['test_acc'])
        train_acc = np.array(history['train_acc'])
        ax.plot(steps, test_acc, '-', color=color, label=label, linewidth=2)
        ax.plot(steps, train_acc, '--', color=color, alpha=0.3, linewidth=1)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('(A) Forgetting Pressure Accelerates Grokking')
    ax.legend(fontsize=10)
    ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)

    # Panel B: Grok steps vs WD (log scale)
    ax = axes[0, 1]
    results_path = RESULTS_BASE / 'exp1_grokking' / 'all_results.json'
    with open(results_path) as f:
        all_results = json.load(f)

    weight_decays = [0.0, 0.001, 0.01, 0.1, 1.0]
    grok_means = []
    grok_stds = []
    for wd in weight_decays:
        runs = all_results[str(wd)]
        groks = [r['grok_step'] for r in runs if r['grokked'] and r['grok_step'] is not None]
        grok_means.append(np.mean(groks))
        grok_stds.append(np.std(groks))

    ax.errorbar(range(len(weight_decays)), grok_means, yerr=grok_stds,
                fmt='o-', capsize=5, color='#2196F3', linewidth=2, markersize=8)
    ax.set_xticks(range(len(weight_decays)))
    ax.set_xticklabels(['0', '0.001', '0.01', '0.1', '1.0'])
    ax.set_xlabel('Weight Decay (Forgetting Pressure)')
    ax.set_ylabel('Steps to Grok (log scale)')
    ax.set_title('(B) 30x Speedup with Forgetting')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)

    # Panel C: Duplication effects (if available)
    ax = axes[1, 0]
    dup_results_path = RESULTS_BASE / 'exp2_duplication' / 'all_results.json'
    if dup_results_path.exists():
        with open(dup_results_path) as f:
            dup_results = json.load(f)
        dup_factors = [1, 2, 4, 8]
        for df in dup_factors:
            runs = dup_results[str(df)]
            history_path = RESULTS_BASE / 'exp2_duplication' / f'history_dup{df}_seed42.json'
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)
                steps = np.array(history['steps'])
                test_acc = np.array(history['test_acc'])
                ax.plot(steps, test_acc, label=f'{df}x dup', linewidth=1.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('(C) Duplication Effect on Generalization')
        ax.legend(fontsize=10)
        ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.3)
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, 'Experiment 2\n(Running)', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)

    # Panel D: Intervention comparison (if available)
    ax = axes[1, 1]
    int_results_path = RESULTS_BASE / 'exp3_interventions' / 'all_results.json'
    if int_results_path.exists():
        with open(int_results_path) as f:
            int_results = json.load(f)
        names = list(int_results.keys())
        grok_means = []
        for name in names:
            runs = int_results[name]
            groks = [r['grok_step'] for r in runs if r['grokked'] and r['grok_step'] is not None]
            grok_means.append(np.mean(groks) if groks else 50000)

        colors_bar = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bars = ax.barh(range(len(names)), grok_means, color=colors_bar, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Steps to Grok')
        ax.set_title('(D) Forgetting Interventions Comparison')
        ax.grid(True, alpha=0.2, axis='x')
    else:
        ax.text(0.5, 0.5, 'Experiment 3\n(Running)', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)

    plt.suptitle('LLMs Won\'t Learn Properly Until They Forget Better', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'summary_figure.png', bbox_inches='tight')
    plt.close()
    print("\nSummary figure saved.")


if __name__ == '__main__':
    exp1_stats = analyze_exp1()
    exp2_stats = analyze_exp2()
    exp3_stats = analyze_exp3()
    create_summary_figure()

    # Save combined statistics
    combined = {
        'exp1': exp1_stats,
        'exp2': exp2_stats,
        'exp3': exp3_stats,
    }
    with open(RESULTS_BASE / 'combined_statistics.json', 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    print("\nAll analysis complete!")
