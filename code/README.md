# Cloned Repositories

## 1. Pythia (EleutherAI)
- **URL**: https://github.com/EleutherAI/pythia
- **Purpose**: Model suite for studying learning dynamics, interpretability, and memorization. 8 model sizes (14M–12B) with 154 checkpoints each, trained on The Pile.
- **Location**: `code/pythia/`
- **Key features**:
  - All models trained on same data in same order
  - Both standard and deduplicated versions available
  - Includes code for "Emergent and Predictable Memorization in LLMs"
  - Models on HuggingFace: `EleutherAI/pythia-{size}` and `EleutherAI/pythia-{size}-deduped`
- **How to use**: Load any checkpoint via HuggingFace transformers:
  ```python
  from transformers import GPTNeoXForCausalLM
  model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-410m", revision="step100000")
  ```

## 2. Deduplicate Text Datasets (Google Research)
- **URL**: https://github.com/google-research/deduplicate-text-datasets
- **Purpose**: Tools for exact and near-duplicate detection in text datasets. Used in Lee et al. (2021).
- **Location**: `code/deduplicate-text-datasets/`
- **Key features**:
  - Suffix array based deduplication
  - Scales to terabyte-scale datasets
  - Used to create deduplicated versions of C4, The Pile, etc.
- **Requirements**: Rust compiler for suffix array construction

## 3. Semantic Memorization (EleutherAI)
- **URL**: https://github.com/EleutherAI/semantic-memorization
- **Purpose**: Taxonomy of memorization types in LMs: recitation (duplicated), reconstruction (predictable), recollection (rare/episodic). Predictive model for memorization.
- **Location**: `code/semantic-memorization/`
- **Key features**:
  - Code/NL classifier for memorized sequence categorization
  - Duplication counting via sequence hashing
  - Predictive model for which sequences will be memorized
  - Based on Pythia models

## 4. SemDeDup (Meta FAIR)
- **URL**: https://github.com/facebookresearch/SemDeDup
- **Purpose**: Semantic deduplication using pre-trained embeddings. Removes semantically similar (not just exact) duplicates.
- **Location**: `code/semdedup/`
- **Key features**:
  - Clustering + cosine similarity for semantic duplicate detection
  - Demonstrated 50% data reduction with minimal quality loss
  - Works at web-scale (LAION-440M, C4)

## 5. Datablations (Hugging Face)
- **URL**: https://github.com/huggingface/datablations
- **Purpose**: Code for "Scaling Data-Constrained Language Models" (Muennighoff et al., 2023). 400 training runs studying data repetition and scaling laws.
- **Location**: `code/datablations/`
- **Key features**:
  - Scaling law implementation for data-constrained regimes
  - Training scripts for various repetition rates (1–100 epochs)
  - Analysis of when data repetition stops helping
