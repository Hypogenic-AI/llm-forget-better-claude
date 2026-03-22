# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "LLMs Won't Learn Properly Until They Forget Better." The hypothesis is that rapid memorization limits LLM generalization and data efficiency.

---

## Papers
Total papers downloaded: 22 (+ 1 failed download)

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Quantifying Memorization Across Neural LMs | Carlini et al. | 2023 | carlini2023_quantifying_memorization.pdf | Log-linear memorization relationships |
| 2 | Extracting Training Data from LLMs | Carlini et al. | 2021 | carlini2021_extracting_training_data.pdf | Practical extraction attacks on GPT-2 |
| 3 | Counterfactual Memorization in Neural LMs | Zhang et al. | 2023 | zhang2023_counterfactual_memorization_correct.pdf | Counterfactual definition of memorization |
| 4 | Example Forgetting during DNN Learning | Toneva et al. | 2019 | toneva2019_example_forgetting.pdf | **Foundational**: forgetting events, unforgettable examples |
| 5 | Does Learning Require Memorization? | Feldman & Zhang | 2020 | feldman2020_does_learning_require_memorization.pdf | Long tail theory: memorization is necessary |
| 6 | Memorization-Compression Cycles | Yu | 2025 | memorization_compression_cycles.pdf | **Key**: compression improves generalization |
| 7 | Grokking in LLM Pretraining | Li et al. | 2025 | grokking_llm_pretraining.pdf | Grokking occurs in real LLM training |
| 8 | Grokking: Generalization Beyond Overfitting | Power et al. | 2022 | power2022_grokking_original.pdf | **Foundational**: delayed generalization |
| 9 | Grokking Modular Arithmetic | Gromov | 2023 | power2022_grokking.pdf | Interpretable grokking analysis |
| 10 | Breaking Memorization Barriers via IB | Wang et al. | 2025 | breaking_memorization_barriers_llm.pdf | IB overcomes memorization in fine-tuning |
| 11 | Deduplicating Training Data Makes LMs Better | Lee et al. | 2021 | lee2021_dedup_training_data.pdf | 804 cites; dedup reduces memorization 10x |
| 12 | Dedup Mitigates Privacy Risks | Kandpal et al. | 2022 | kandpal2022_dedup_privacy.pdf | Memorization superlinear in duplication |
| 13 | SemDeDup | Abbas et al. | 2023 | abbas2023_semdedup.pdf | Semantic dedup: 50% data reduction |
| 14 | Scaling Data-Constrained LMs | Muennighoff et al. | 2023 | muennighoff2023_scaling_data_constrained.pdf | 4 epochs ≈ unique data; 40+ = worthless |
| 15 | D4: Document De-Duplication & Diversification | Tirumala et al. | 2023 | tirumala2023_d4.pdf | Smart selection: 20% efficiency gains |
| 16 | Generalization or Memorization: Data Contamination | Yang et al. | 2024 | yang2024_generalization_or_memorization.pdf | Contamination detection via output distribution |
| 17 | Gen vs Mem: Tracing LM Capabilities | Wang et al. | 2024 | jiang2024_gen_vs_mem_tracing.pdf | Memorization stronger for factual QA |
| 18 | LLM Graph Reasoning Generalization | Zhang et al. | 2024 | llm_graph_reasoning_generalization.pdf | LLMs memorize graph patterns |
| 19 | Spurious Forgetting in Continual Learning | Zheng et al. | 2025 | spurious_forgetting_continual.pdf | Task alignment vs knowledge loss |
| 20 | Continual Learning of LLMs: Survey | Wang et al. | 2024 | wang2024_continual_learning_survey.pdf | Comprehensive CL survey |
| 21 | InsCL: Data-efficient Continual Learning | Wang et al. | 2024 | inscl_data_efficient_continual.pdf | Instruction-based replay |
| 22 | Forgetting before Learning | Ni et al. | 2024 | forgetting_before_learning_llm.pdf | Parametric arithmetic for knowledge update |

See `papers/README.md` for detailed descriptions.

---

## Datasets
Total datasets downloaded: 4 locally + 2 streaming-accessible

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| WikiMIA | HuggingFace swj0419/WikiMIA | 250 examples (len128) | Memorization detection | datasets/wikimia/ | Binary: seen/unseen by model |
| GSM8K | HuggingFace openai/gsm8k | 7,473 train / 1,319 test | Math reasoning | datasets/gsm8k/ | Tests memorization vs generalization |
| TriviaQA (sample) | HuggingFace trivia_qa | 5,000 examples | Factual QA | datasets/triviaqa_sample/ | Strongest memorization effect (Wang 2024) |
| Modular Arithmetic | Generated | 9,409 addition / 9,312 division | Grokking | datasets/modular_arithmetic/ | p=97, 50/50 split |
| The Pile (deduped) | EleutherAI/the_pile_deduplicated | ~207B tokens | LM pretraining | Streaming only | Pythia training data |
| C4 | allenai/c4 | ~305GB | LM pretraining | Streaming only | Deduplication studies |

See `datasets/README.md` for download instructions and loading code.

---

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| Pythia | github.com/EleutherAI/pythia | Model suite for learning dynamics | code/pythia/ | 8 sizes × 154 checkpoints |
| deduplicate-text-datasets | github.com/google-research/deduplicate-text-datasets | Text deduplication tools | code/deduplicate-text-datasets/ | Suffix array based |
| semantic-memorization | github.com/EleutherAI/semantic-memorization | Memorization taxonomy | code/semantic-memorization/ | Recite/Reconstruct/Recollect |
| SemDeDup | github.com/facebookresearch/SemDeDup | Semantic deduplication | code/semdedup/ | Embedding-based |
| datablations | github.com/huggingface/datablations | Data repetition scaling laws | code/datablations/ | 400 training runs |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
- Semantic Scholar API for initial broad search (100 papers from 5 queries)
- arXiv API for recent papers (60 papers from 3 queries)
- Manual curation of 11 must-have foundational papers
- Ranked by keyword relevance and citation count
- Downloaded top 22 most relevant papers

### Selection Criteria
Papers were selected based on:
1. Direct relevance to memorization-forgetting-generalization dynamics
2. Citation impact (prioritized highly-cited foundational works)
3. Recency (2023-2025 for state-of-the-art, 2019-2022 for foundations)
4. Availability of code/data for reproducibility
5. Diversity of perspectives (theory, empirical, applications)

### Challenges Encountered
- Semantic Scholar API rate limiting (429 errors) — only 5/10 queries succeeded
- Some arXiv IDs in search results pointed to wrong papers
- Paper-finder service was unavailable (timed out)
- One target paper ("On the Similarity of Memorization and Generalization") downloaded incorrectly

### Gaps and Workarounds
- No unified benchmark specifically for "beneficial forgetting" — used combination of existing benchmarks
- Limited work on controlled forgetting experiments — this IS the research gap to fill
- Most forgetting work is at small scale (CIFAR) while memorization work is at LLM scale — bridging needed

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **Modular arithmetic** for controlled grokking experiments (fast, reproducible)
- **Pythia checkpoints** for studying memorization dynamics across LLM training
- **WikiMIA** for measuring memorization detection accuracy
- **GSM8K + TriviaQA** for comparing memorization effects across task types

### 2. Baseline Methods
- Standard training (no intervention)
- Weight decay as implicit forgetting
- Deduplication (exact + semantic) as data-side forgetting
- Information Bottleneck regularization
- Dropout as stochastic forgetting

### 3. Evaluation Metrics
- **Forgetting events** (Toneva-style per-example tracking)
- **Min-K% Prob** (memorization detection)
- **Test accuracy/loss** (generalization)
- **Performance-per-token** (data efficiency)
- **Matrix-Based Entropy** (representation compression)

### 4. Code to Adapt/Reuse
- **Pythia**: Load any checkpoint for analysis, training dynamics studies
- **semantic-memorization**: Memorization taxonomy pipeline
- **deduplicate-text-datasets**: Pre/post deduplication comparisons
- **datablations**: Data repetition experimental framework

### 5. Suggested Experiment Pipeline
1. **Small-scale proof of concept**: Modular arithmetic + grokking (fast iteration)
2. **Medium-scale validation**: Pythia 70M-410M with forgetting event tracking
3. **Analysis**: Compare memorization dynamics in standard vs deduped Pythia models
4. **Intervention**: Apply compression/IB regularization during training, measure effect on forgetting events and generalization
5. **Data efficiency comparison**: Measure performance-per-token with and without forgetting interventions
