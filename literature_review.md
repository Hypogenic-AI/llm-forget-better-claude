# Literature Review: LLMs Won't Learn Properly Until They Forget Better

## Research Area Overview

This review examines the intersection of memorization, forgetting, and generalization in large language models (LLMs). The hypothesis under investigation is that rapid memorization limits LLM generalization, making them less data-efficient compared to humans. We survey work on: (1) how and why LLMs memorize training data, (2) the role of forgetting in learning, (3) data efficiency and deduplication, and (4) the memorization-to-generalization transition.

---

## Key Papers

### 1. Toneva et al. (2019) — "An Empirical Study of Example Forgetting during Deep Neural Network Learning" (ICLR 2019)
- **arXiv:** 1812.05159
- **Key Contribution:** Introduces the concept of "forgetting events" — transitions from correct to incorrect classification during training. Shows that examples vary dramatically in how often they are forgotten, from "unforgettable" (learned once, never forgotten) to highly forgettable.
- **Methodology:** Track per-example classification accuracy at every SGD step. A forgetting event occurs when acc_i^t > acc_i^{t+1}. Count forgetting events per example across training.
- **Datasets:** CIFAR-10, CIFAR-100, permuted MNIST
- **Key Findings:**
  - ~30% of CIFAR-10 examples are unforgettable (learned once and never forgotten)
  - Forgetting events are highly consistent across runs with different random seeds (Spearman ρ ≈ 0.9)
  - Removing unforgettable examples (keeping only the "hard" forgettable ones) still allows near-full test accuracy, enabling up to 30% dataset compression
  - Noisy/mislabeled examples have the highest forgetting counts
  - Forgetting events correlate with classification margin (ρ = -0.74)
- **Relevance:** Directly supports the hypothesis — examples learned too quickly (unforgettable) contribute less to generalization. The "hard" examples that are repeatedly forgotten and relearned are the ones that shape the decision boundary.
- **Code:** Not released with paper, but methodology is simple to reproduce

### 2. Feldman & Zhang (2020) — "Does Learning Require Memorization? A Short Tale about a Long Tail" (STOC 2020 / arXiv 2008.03703)
- **Key Contribution:** Provides theoretical and empirical evidence that memorization is *necessary* for good generalization on long-tailed data distributions.
- **Methodology:** Influence estimation via leave-one-out retraining (~300K model trainings on CIFAR-100, Imagenet). Measures "memorization" as the influence of individual examples.
- **Key Findings:**
  - Long-tailed data distributions require memorizing rare subpopulations
  - ~5% of training examples are "high-memorization" — accuracy on these drops by ≥40% when removed
  - Models memorize both useful rare examples AND useless outliers because they're statistically indistinguishable
  - Large generalization gaps are the *price* of achieving near-optimal accuracy on long-tailed data
- **Relevance:** Challenges the simplistic view that memorization = bad. Suggests the problem isn't memorization per se, but *undiscriminating* memorization — LLMs memorize too quickly without distinguishing useful rare examples from noise.
- **Datasets:** CIFAR-100, ImageNet (with extensive leave-one-out analysis)

### 3. Carlini et al. (2023) — "Quantifying Memorization Across Neural Language Models" (ICLR 2023)
- **arXiv:** 2202.07646
- **Key Contribution:** Three log-linear relationships governing memorization in LMs: memorization grows with (1) model capacity, (2) data duplication, (3) context length.
- **Key Findings:**
  - Memorization is more prevalent than previously believed
  - Larger models memorize more of their training data
  - Duplicated examples are memorized superlinearly more
  - Will likely worsen as models scale, without active mitigations
- **Relevance:** Quantifies the memorization problem at scale — larger LLMs memorize faster and more, supporting the hypothesis that rapid memorization limits data efficiency.

### 4. Zhang et al. (2023) — "Counterfactual Memorization in Neural Language Models" (ICLR 2024)
- **arXiv:** 2112.12938
- **Key Contribution:** Introduces *counterfactual memorization* — measuring memorization by comparing a model trained with vs. without a specific example, accounting for the fact that some sequences are inherently predictable.
- **Key Findings:**
  - Distinguishes "extractable memorization" (can regenerate verbatim) from "counterfactual memorization" (model behavior changes due to a specific example)
  - Many sequences that appear memorized are actually just predictable from other data
  - True memorization is concentrated in specific, often duplicated or distinctive examples
- **Relevance:** Refines our understanding of what "memorization" means — critical for designing experiments that test whether forgetting improves learning.

### 5. Yu (2025) — "Memorization-Compression Cycles Improve Generalization" (NeurIPS 2025 Workshop)
- **arXiv:** 2505.08727
- **Key Contribution:** Shows that during LLM pretraining, models naturally oscillate between "memorization" phases (expanding representations) and "compression" phases (contracting representations). Amplifying compression phases improves generalization.
- **Methodology:** Tracks cosine similarity between cross-entropy gradients and Matrix-Based Entropy (MBE) gradients during training. Positive = memorization phase, negative = compression phase.
- **Key Findings:**
  - Generalization bound: Error ≤ Empirical Error + O(log(N) * 2^(αH(R_l)) / √N)
  - Two levers: more data (N) OR lower representation entropy H(R_l)
  - Adding explicit compression loss (MBE minimization) improves generalization significantly
  - On FineWeb training: +1.8% improvement on MMLU, +2.2% on HellaSwag
  - Most of the gain comes from compression, not just more training
- **Relevance:** **Directly supports the hypothesis.** Shows that LLMs that "forget" (compress) better do learn better. The memorization-compression cycle is essentially a formalization of "forgetting to learn."
- **Datasets:** FineWeb (0.73B tokens), evaluated on MMLU, HellaSwag, ARC

### 6. Li, Fan & Zhou (2025) — "Grokking in LLM Pretraining? Monitor Memorization-to-Generalization"
- **arXiv:** 2506.21551
- **Key Contribution:** Demonstrates that grokking (delayed generalization long after memorization) occurs in practical LLM pretraining, not just toy settings. Proposes monitoring via MoE routing pathways.
- **Methodology:** Uses OLMoE-1B-7B (MoE) model with 154 checkpoints. Tracks per-sample memorization (loss < ε and stable) and generalization (via LoRA instruction tuning + benchmark evaluation).
- **Key Findings:**
  - Clear temporal gap between memorization and generalization onset
  - Routing pathway entropy correlates with the memorization-to-generalization transition
  - Memorized samples that later generalize develop more diverse routing pathways
  - This provides a test-free way to monitor when a model shifts from memorizing to generalizing
- **Relevance:** Shows that the memorize-then-generalize transition is a real phenomenon in LLMs, and that models need time (and implicit "forgetting" of surface patterns) to transition from memorization to true generalization.

### 7. Wang et al. (2025) — "Breaking Memorization Barriers in LLM Code Fine-Tuning via Information Bottleneck"
- **arXiv:** 2510.16022
- **Key Contribution:** Identifies the "memorization barrier" — LLMs pre-memorize fine-tuning data during pretraining, preventing effective learning during fine-tuning. Proposes Information Bottleneck (IB) based approach to break through it.
- **Methodology:** Uses Min-K% Prob to measure memorization. Applies IB principle: minimize I(X;Z) while maximizing I(Z;Y) — compress input representation while preserving task-relevant signal.
- **Key Findings:**
  - Base models already heavily memorize downstream code datasets
  - This prior memorization creates bad local optima that SFT cannot escape
  - IB-based fine-tuning (ForgetFilter) forces model to "forget" memorized patterns and learn generalizable features
  - Significant improvements on Pass@1 and strict multi-sample metrics
- **Relevance:** **Directly supports the hypothesis.** Demonstrates that forcing models to forget (via information bottleneck) improves generalization in code tasks.

### 8. Lee et al. (2021) — "Deduplicating Training Data Makes Language Models Better" (ACL 2022)
- **arXiv:** 2107.06499  | **Citations:** 804
- **Key Contribution:** Shows massive duplication in standard datasets. Deduplication reduces memorized output by 10x and improves training efficiency.
- **Key Findings:**
  - Over 1% of unprompted LM output is verbatim from training data
  - C4 contains a single 61-word sentence repeated >60,000 times
  - Deduplication: models need fewer steps for same/better accuracy
  - Train-test overlap affects >4% of validation sets
- **Code:** https://github.com/google-research/deduplicate-text-datasets
- **Relevance:** Reducing redundancy (a form of preventing over-memorization) directly improves both efficiency and quality.

### 9. Abbas et al. (2023) — "SemDeDup: Data-efficient Learning at Web-scale through Semantic Deduplication" (ICLR 2024)
- **arXiv:** 2303.09540 | **Citations:** 258
- **Key Contribution:** Goes beyond exact deduplication to remove *semantic* duplicates using pre-trained embeddings. Can remove 50% of data with minimal performance loss.
- **Key Findings:**
  - Removing semantic duplicates preserves in-distribution performance
  - Performance *increases* out of distribution — less memorization → better generalization
  - Effectively halves training time
- **Relevance:** Stronger evidence that redundant memorization hurts. When similar examples are consolidated, models learn more efficiently.

### 10. Muennighoff et al. (2023) — "Scaling Data-Constrained Language Models"
- **arXiv:** 2305.16264 | **Citations:** Large-scale (400 training runs)
- **Key Contribution:** Scaling laws for data-constrained regimes. Up to 4 epochs of repetition ≈ unique data; beyond that, returns diminish to zero.
- **Key Findings:**
  - Training with up to 4 epochs yields negligible loss difference vs unique data
  - At 40+ epochs, additional compute provides zero benefit
  - Code/multilingual data augmentation helps mitigate data scarcity
  - Proposes scaling law accounting for diminishing returns of repeated tokens
- **Code:** https://github.com/huggingface/datablations
- **Relevance:** Quantifies when repeated exposure (memorization) stops helping — directly informs the "learning more from less" aspect of the hypothesis.

### 11. Tirumala et al. (2023) — "D4: Improving LLM Pretraining via Document De-Duplication and Diversification"
- **arXiv:** 2308.12284
- **Key Findings:** Smart data selection via embeddings provides 20% efficiency gains and up to 2% downstream accuracy improvement. Intelligent repetition outperforms random data while random repetition is worse than baseline.
- **Relevance:** Data diversity (the opposite of redundant memorization) consistently helps.

### 12. Wang et al. (2024) — "Generalization v.s. Memorization: Tracing Language Models' Capabilities Back to Pretraining Data" (ICLR 2025)
- **arXiv:** 2407.14985
- **Key Contribution:** Uses "distributional memorization" to measure correlation between LLM outputs and pretraining data frequency via task-gram language models.
- **Key Findings:**
  - TriviaQA: strongest memorization effect
  - Math/reasoning tasks: greater generalization, producing more novel outputs
  - Memorization plays larger role in simpler, knowledge-intensive tasks
  - Generalization is key for harder, reasoning-based tasks
- **Relevance:** Not all tasks are equal — memorization is more/less harmful depending on task type.

### 13. Power et al. (2022) — "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" (ICLR 2022)
- **arXiv:** 2201.02177
- **Key Contribution:** Discovery of "grokking" — models achieve perfect training accuracy (memorization) long before test accuracy improves (generalization). With continued training, generalization eventually emerges.
- **Key Findings:**
  - On modular arithmetic tasks, models memorize quickly but generalize only after much more training
  - Weight decay is critical — it forces a form of "forgetting" that enables generalization
  - The memorize-then-generalize pattern suggests these are distinct learning phases
- **Relevance:** **Foundational for the hypothesis.** Grokking is the clearest example of "learning requires forgetting" — the model must move beyond memorized solutions to find generalizable ones.

### 14. Zheng et al. (2025) — "Spurious Forgetting in Continual Learning of Language Models" (ICLR 2025)
- **arXiv:** 2501.13453
- **Key Contribution:** Distinguishes "spurious forgetting" (loss of task alignment, not knowledge) from true catastrophic forgetting in continual learning.
- **Key Findings:**
  - Performance drops during continual learning often reflect alignment disruption, not knowledge loss
  - Freezing bottom layers substantially improves continual learning
  - Knowledge is retained even when performance drops
- **Relevance:** Important nuance — not all "forgetting" is equal. The type of forgetting matters for whether it helps or hurts learning.

---

## Common Methodologies

1. **Memorization measurement:** Min-K% Prob (Shi et al.), membership inference attacks, counterfactual memorization, distributional memorization
2. **Forgetting measurement:** Per-example forgetting event counting (Toneva), influence estimation (Feldman), routing pathway analysis (Li et al.)
3. **Deduplication:** Exact (MinHash), near-duplicate (suffix arrays), semantic (SemDeDup via embeddings)
4. **Information-theoretic approaches:** Information Bottleneck, Matrix-Based Entropy, representation compression
5. **Scaling analysis:** Chinchilla-style scaling laws extended to data-constrained regimes

## Standard Baselines

- **Pythia model suite** (70M–12B): Standard for studying learning dynamics, has 154 checkpoints per model
- **GPT-2/GPT-Neo:** Commonly used for memorization extraction studies
- **Standard SFT vs. SFT + regularization:** For fine-tuning memorization studies
- **Random replay vs. smart replay:** For continual learning baselines

## Evaluation Metrics

- **Memorization rate:** Fraction of training data extractable/memorized
- **Forgetting events:** Count of correct→incorrect transitions during training
- **Min-K% Prob score:** Average log-likelihood of least-likely K% tokens
- **Generalization gap:** Train accuracy - test accuracy
- **Data efficiency:** Performance per token/FLOP

## Datasets in the Literature

| Dataset | Used For | Papers |
|---------|----------|--------|
| The Pile (+ deduped) | LLM pretraining, memorization | Carlini, Pythia, many |
| C4 | LM pretraining, dedup studies | Lee, SemDeDup, D4 |
| CIFAR-10/100 | Forgetting event tracking | Toneva, Feldman |
| WikiMIA | Memorization detection benchmark | Shi et al. |
| GSM8K/GSM1K | Math reasoning memorization | Wang et al. |
| TriviaQA | Knowledge memorization | Wang et al. |
| Modular arithmetic | Grokking experiments | Power et al. |
| FineWeb | Compression cycle training | Yu |

## Gaps and Opportunities

1. **No unified framework** connecting forgetting events, grokking, compression cycles, and data deduplication — these are studied separately but likely manifestations of the same underlying phenomenon
2. **Limited work on *controlled* forgetting** — most work observes forgetting or prevents it; few papers explore *inducing* beneficial forgetting during training
3. **Scale gap:** Forgetting events studied at CIFAR scale; memorization studied at LLM scale; no work connecting the two
4. **No comparison to human learning:** The hypothesis claims LLMs are less data-efficient than humans *because* of memorization patterns, but no empirical comparison exists
5. **Missing: forgetting-aware curriculum learning** — scheduling training to exploit the memorize-compress cycle

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **Modular arithmetic** (synthetic) — for grokking/forgetting dynamics in controlled setting
2. **Pythia checkpoints + The Pile** — for studying memorization dynamics across training
3. **WikiMIA** — for measuring memorization at specific training stages
4. **GSM8K** — for testing memorization vs generalization on reasoning tasks
5. **C4 (subset)** — for deduplication experiments if compute allows

### Recommended Baselines
1. Standard training (no forgetting intervention)
2. Weight decay as implicit forgetting mechanism
3. SemDeDup / exact dedup as data-side forgetting
4. Information Bottleneck regularization (ForgetFilter from Wang et al.)

### Recommended Metrics
1. Forgetting events per example (Toneva-style)
2. Min-K% Prob for memorization detection
3. Test accuracy / loss for generalization
4. Data efficiency: performance-per-token curves
5. Representation entropy (MBE) for compression

### Methodological Considerations
- Use Pythia for reproducibility (all checkpoints available)
- Control for data ordering (Pythia trained on same data in same order)
- Track both per-example and aggregate metrics
- Compare forgetting patterns between deduplicated and non-deduplicated training
- Consider MoE routing analysis for mechanistic insights (if using MoE models)
