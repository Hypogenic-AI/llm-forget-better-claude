# LLMs Won't Learn Properly Until They Forget Better

Research project investigating whether rapid memorization limits neural network generalization, and whether inducing "beneficial forgetting" improves data efficiency.

## Key Findings

- **30x speedup in generalization** when increasing forgetting pressure (weight decay 0 to 1.0) in modular arithmetic grokking experiments
- **Strong statistical evidence** (Spearman rho = -0.88, p < 0.00002, Cohen's d = 3.91) that forgetting pressure accelerates the transition from memorization to generalization
- **Data duplication hurts**: 8x duplication slows grokking by 37%, supporting the claim that redundant examples provide diminishing returns
- **Optimal forgetting exists**: Too little = prolonged memorization; too much = can't learn. Weight decay is a surprisingly effective forgetting mechanism
- **Unforgettable examples are diagnostic**: Models with low forgetting pressure have ~90% "unforgettable" examples (learned once, never forgotten), while high forgetting pressure reduces this to ~1%, forcing continuous re-learning that drives generalization

## How to Reproduce

```bash
# Setup environment
uv venv && source .venv/bin/activate
uv add torch numpy matplotlib scipy tqdm pandas scikit-learn

# Run experiments
python src/experiment1_grokking.py      # ~5 min, tests forgetting pressure
python src/experiment2_duplication.py    # ~3 min, tests data redundancy
python src/experiment3_interventions.py  # ~5 min, compares forgetting mechanisms

# Generate analysis and plots
python src/analyze_and_plot.py
```

Requires GPU (tested on NVIDIA RTX A6000). Total runtime: ~15 minutes.

## File Structure

```
├── REPORT.md                    # Full research report with results
├── planning.md                  # Research plan and methodology
├── literature_review.md         # Literature review (pre-gathered)
├── resources.md                 # Resource catalog
├── src/
│   ├── model.py                 # Transformer model + dataset creation
│   ├── train.py                 # Training loop with forgetting tracking
│   ├── experiment1_grokking.py  # Exp 1: Weight decay vs grokking
│   ├── experiment2_duplication.py # Exp 2: Data duplication effects
│   ├── experiment3_interventions.py # Exp 3: Forgetting interventions
│   └── analyze_and_plot.py      # Analysis and visualization
├── results/
│   ├── exp1_grokking/           # Experiment 1 results
│   ├── exp2_duplication/        # Experiment 2 results
│   ├── exp3_interventions/      # Experiment 3 results
│   ├── plots/                   # All visualizations
│   └── combined_statistics.json # Statistical test results
├── papers/                      # Downloaded research papers
├── datasets/                    # Pre-downloaded datasets
└── code/                        # Cloned baseline repositories
```

## Full Report

See [REPORT.md](REPORT.md) for complete methodology, results, analysis, and discussion.
