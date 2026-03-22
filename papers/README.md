# Downloaded Papers

## Core Papers on Memorization and Forgetting

1. **Quantifying Memorization Across Neural Language Models** (Carlini et al., 2023)
   - File: `carlini2023_quantifying_memorization.pdf`
   - ICLR 2023 | arXiv: 2202.07646 | ~380+ citations
   - Log-linear relationships governing LM memorization: model size, duplication, context length

2. **Extracting Training Data from Large Language Models** (Carlini et al., 2021)
   - File: `carlini2021_extracting_training_data.pdf`
   - USENIX Security 2021 | arXiv: 2012.07805
   - Demonstrates practical training data extraction attacks on GPT-2

3. **Counterfactual Memorization in Neural Language Models** (Zhang et al., 2023)
   - File: `zhang2023_counterfactual_memorization_correct.pdf`
   - arXiv: 2112.12938
   - Defines counterfactual memorization: comparing model with/without specific examples

4. **An Empirical Study of Example Forgetting during Deep Neural Network Learning** (Toneva et al., 2019)
   - File: `toneva2019_example_forgetting.pdf`
   - ICLR 2019 | arXiv: 1812.05159
   - Foundational: defines forgetting events, unforgettable examples, dataset compression

5. **Does Learning Require Memorization? A Short Tale about a Long Tail** (Feldman & Zhang, 2020)
   - File: `feldman2020_does_learning_require_memorization.pdf`
   - STOC 2020 | arXiv: 2008.03703
   - Theory: memorization is necessary for long-tailed distributions

## Memorization-Generalization Transition

6. **Memorization-Compression Cycles Improve Generalization** (Yu, 2025)
   - File: `memorization_compression_cycles.pdf`
   - arXiv: 2505.08727
   - Key: compression phases during training improve generalization

7. **Grokking in LLM Pretraining? Monitor Memorization-to-Generalization** (Li et al., 2025)
   - File: `grokking_llm_pretraining.pdf`
   - arXiv: 2506.21551
   - Shows grokking occurs in practical LLM pretraining

8. **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets** (Power et al., 2022)
   - File: `power2022_grokking_original.pdf`
   - ICLR 2022 | arXiv: 2201.02177
   - Original grokking discovery: delayed generalization after memorization

9. **Grokking Modular Arithmetic** (Gromov, 2023)
   - File: `power2022_grokking.pdf`
   - Interpretable grokking analysis with analytic weight solutions

10. **Breaking Memorization Barriers in LLM Code Fine-Tuning via Information Bottleneck** (Wang et al., 2025)
    - File: `breaking_memorization_barriers_llm.pdf`
    - arXiv: 2510.16022
    - IB-based approach to overcome memorization barriers in fine-tuning

11. **On the Similarity of Memorization and Generalization in Training LLMs** (2024)
    - File: *(download was incorrect — wrong PDF retrieved)*
    - arXiv: 2410.01524

## Data Efficiency and Deduplication

12. **Deduplicating Training Data Makes Language Models Better** (Lee et al., 2021)
    - File: `lee2021_dedup_training_data.pdf`
    - ACL 2022 | arXiv: 2107.06499 | 804 citations
    - Near-duplicate removal reduces memorization 10x, improves efficiency

13. **Deduplicating Training Data Mitigates Privacy Risks in Language Models** (Kandpal et al., 2022)
    - File: `kandpal2022_dedup_privacy.pdf`
    - ICML 2022 | arXiv: 2202.06539 | 380 citations
    - Memorization superlinearly related to duplication count

14. **SemDeDup: Data-efficient Learning through Semantic Deduplication** (Abbas et al., 2023)
    - File: `abbas2023_semdedup.pdf`
    - ICLR 2024 | arXiv: 2303.09540 | 258 citations
    - Semantic deduplication: 50% data removal with minimal loss, improved OOD

15. **Scaling Data-Constrained Language Models** (Muennighoff et al., 2023)
    - File: `muennighoff2023_scaling_data_constrained.pdf`
    - arXiv: 2305.16264
    - Scaling laws for data repetition: 4 epochs ≈ unique data, 40+ epochs = worthless

16. **D4: Improving LLM Pretraining via Document De-Duplication and Diversification** (Tirumala et al., 2023)
    - File: `tirumala2023_d4.pdf`
    - arXiv: 2308.12284
    - Smart data selection via embeddings: 20% efficiency gains

## Generalization and Contamination

17. **Generalization or Memorization: Data Contamination and Trustworthy Evaluation** (Yang et al., 2024)
    - File: `yang2024_generalization_or_memorization.pdf`
    - arXiv: 2402.15938 | 143 citations
    - Contamination detection via output distribution

18. **Generalization v.s. Memorization: Tracing LM Capabilities to Pretraining Data** (Wang et al., 2024)
    - File: `jiang2024_gen_vs_mem_tracing.pdf`
    - ICLR 2025 | arXiv: 2407.14985
    - Distributional memorization: stronger for factual QA, weaker for reasoning

19. **Can LLM Graph Reasoning Generalize beyond Pattern Memorization?** (Zhang et al., 2024)
    - File: `llm_graph_reasoning_generalization.pdf`
    - arXiv: 2406.15992
    - LLMs struggle to generalize graph reasoning beyond memorized patterns

## Continual Learning and Forgetting

20. **Spurious Forgetting in Continual Learning of Language Models** (Zheng et al., 2025)
    - File: `spurious_forgetting_continual.pdf`
    - ICLR 2025 | arXiv: 2501.13453
    - Distinguishes task alignment loss from true knowledge forgetting

21. **Continual Learning of Large Language Models: A Comprehensive Survey** (Wang et al., 2024)
    - File: `wang2024_continual_learning_survey.pdf`
    - arXiv: 2404.16789
    - Comprehensive survey of continual learning methods for LLMs

22. **InsCL: Data-efficient Continual Learning for Fine-tuning LLMs** (Wang et al., 2024)
    - File: `inscl_data_efficient_continual.pdf`
    - arXiv: 2403.11435
    - Instruction-based replay with Wasserstein distance for task similarity

23. **Forgetting before Learning: Utilizing Parametric Arithmetic for Knowledge Manipulation** (Ni et al., 2024)
    - File: `forgetting_before_learning_llm.pdf`
    - arXiv: 2311.08011
    - F-Learning: parametric arithmetic to forget old knowledge before learning new
