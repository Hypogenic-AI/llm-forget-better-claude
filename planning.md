# Research Plan: LLMs Won't Learn Properly Until They Forget Better

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are trained on trillions of tokens yet remain less data-efficient than humans. If the bottleneck is not architectural but behavioral — models memorize too fast, collapsing similar examples into single representations before extracting the generalizable structure — then interventions that induce "beneficial forgetting" could dramatically improve data efficiency. This has direct implications for training cost, model generalization, and the scaling laws that govern LLM development.

### Gap in Existing Work
The literature studies memorization (Carlini et al.), forgetting events (Toneva et al.), grokking (Power et al.), and compression cycles (Yu 2025) **separately**. No work has:
1. Systematically varied "forgetting pressure" and measured its effect on data efficiency
2. Tracked per-example forgetting events in the grokking setting to understand the memorize→forget→generalize transition
3. Tested whether duplicated (highly similar) examples are "factored out" too quickly, reducing learning signal

### Our Novel Contribution
We provide the first controlled experiment linking forgetting pressure → forgetting events → data efficiency, using modular arithmetic as a clean testbed. We test three specific predictions:
1. More forgetting pressure (weight decay) → faster grokking → better data efficiency
2. Duplicated examples are memorized with fewer forgetting events and contribute less to generalization
3. Forgetting interventions (dropout, noise) improve test performance per training token

### Experiment Justification
- **Experiment 1 (Grokking + Weight Decay):** Directly tests whether forgetting pressure accelerates generalization. Grokking is the cleanest natural example of the memorize-then-generalize phenomenon.
- **Experiment 2 (Duplication and Memorization Speed):** Tests the core claim that similar examples are "factored out too fast." By controlling duplication, we isolate the effect of redundancy on memorization speed and generalization.
- **Experiment 3 (Forgetting Interventions):** Tests whether explicit forgetting mechanisms improve data efficiency, supporting the causal direction of the hypothesis.

## Research Question
Does rapid memorization limit LLM generalization, and can inducing beneficial forgetting improve data efficiency?

## Hypothesis Decomposition
- **H1:** Higher forgetting pressure (weight decay) leads to faster grokking and better generalization
- **H2:** Duplicated examples are memorized faster (fewer forgetting events) and contribute less to generalization
- **H3:** Forgetting interventions (dropout, noise injection, periodic perturbation) improve data efficiency (test accuracy per training step)

## Proposed Methodology

### Approach
Train small transformers on modular arithmetic (addition mod 97), which is the gold-standard setting for studying grokking. This setting is:
- Fully controlled (no confounds from data quality)
- Fast to iterate (minutes per run)
- Well-studied (baselines exist)
- Directly relevant to the hypothesis (grokking = memorize then generalize)

### Experimental Steps

**Experiment 1: Forgetting Pressure and Grokking**
1. Train 1-layer transformer on modular addition (mod 97)
2. Vary weight decay: {0, 0.001, 0.01, 0.1, 1.0}
3. Track: train acc, test acc, per-example forgetting events at each step
4. 3 random seeds per condition
5. Measure: steps to 99% test accuracy, total forgetting events

**Experiment 2: Duplication Effects**
1. Create datasets with controlled duplication: {1x, 2x, 4x, 8x} of random 25% subset
2. Train with fixed hyperparameters
3. Track memorization speed (steps to 100% train acc) on duplicated vs unique examples
4. Measure forgetting events on duplicated vs unique examples
5. Compare generalization (test accuracy at fixed steps)

**Experiment 3: Forgetting Interventions**
1. Compare: baseline, dropout (0.1, 0.3), Gaussian noise injection, periodic weight reset
2. Fixed training budget (same total steps)
3. Measure test accuracy curves and data efficiency

### Baselines
- Standard training with AdamW, no regularization beyond default
- Standard weight decay (0.01) as baseline forgetting mechanism

### Evaluation Metrics
- **Steps to grokking**: First step where test acc > 99%
- **Forgetting events**: Per-example count of correct→incorrect transitions
- **Data efficiency**: Test accuracy as function of training steps
- **Memorization speed**: Steps to 100% train accuracy
- **Generalization gap**: Train acc - test acc over training

### Statistical Analysis Plan
- 3 seeds per condition, report mean ± std
- Wilcoxon rank-sum test for comparing conditions (non-parametric, small N)
- Spearman correlation between forgetting events and generalization
- Effect size: Cohen's d where appropriate

## Expected Outcomes
- H1: Weight decay 0.01-0.1 should show fastest grokking; 0 (no forgetting) should be slowest
- H2: Duplicated examples should have fewer forgetting events and removing them shouldn't hurt test accuracy
- H3: Moderate forgetting interventions should improve data efficiency; too much should hurt

## Timeline
- Planning: 20 min ✓
- Implementation: 60 min
- Experiments: 60 min
- Analysis: 30 min
- Documentation: 20 min

## Potential Challenges
- Grokking can be sensitive to hyperparameters — mitigate with grid search
- Per-example forgetting tracking adds memory overhead — use efficient implementation
- Some interventions may prevent convergence — monitor and adjust

## Success Criteria
- Clear evidence that forgetting pressure affects grokking speed
- Quantitative relationship between forgetting events and generalization
- At least one forgetting intervention that improves data efficiency
