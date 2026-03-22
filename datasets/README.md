# Downloaded Datasets

This directory contains datasets for the research project "LLMs Won't Learn Properly Until They Forget Better." Large data files are NOT committed to git due to size. Follow the download instructions below.

---

## Dataset 1: WikiMIA (Memorization Detection Benchmark)

### Overview
- **Source**: HuggingFace `swj0419/WikiMIA`
- **Size**: 250 examples (length=128 split), also has length 32/64/256 splits
- **Format**: HuggingFace Dataset
- **Task**: Binary classification — was this text in the model's training data?
- **Features**: `input` (text string), `label` (0=unseen, 1=seen)
- **License**: Open

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("swj0419/WikiMIA", split="WikiMIA_length128")
dataset.save_to_disk("datasets/wikimia")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wikimia")
```

### Notes
- Created from Wikipedia articles written before/after LLM training cutoffs
- Used with Min-K% Prob method for memorization detection
- Good for evaluating memorization at different sequence lengths

---

## Dataset 2: GSM8K (Math Reasoning)

### Overview
- **Source**: HuggingFace `openai/gsm8k`
- **Size**: train=7,473, test=1,319
- **Format**: HuggingFace Dataset
- **Task**: Grade school math word problems
- **Features**: `question` (string), `answer` (string with chain-of-thought + final answer)
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main")
dataset.save_to_disk("datasets/gsm8k")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/gsm8k")
```

### Notes
- Key benchmark for studying memorization vs generalization in reasoning tasks
- Wang et al. (2024) showed GSM8K exhibits more generalization than memorization
- Compare with GSM1K (contamination-free mirror) to detect memorization

---

## Dataset 3: TriviaQA Sample (Knowledge Memorization)

### Overview
- **Source**: HuggingFace `trivia_qa` (rc.nocontext config)
- **Size**: 5,000 examples (subset of 138K total)
- **Format**: HuggingFace Dataset
- **Task**: Factual question answering
- **Features**: `question`, `answer` (with aliases), entity pages
- **License**: Apache 2.0

### Download Instructions

```python
from datasets import load_dataset
# Full dataset
dataset = load_dataset("trivia_qa", "rc.nocontext")
# Or just a sample
dataset = load_dataset("trivia_qa", "rc.nocontext", split="train[:5000]")
dataset.save_to_disk("datasets/triviaqa_sample")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/triviaqa_sample")
```

### Notes
- Wang et al. (2024) found TriviaQA shows the STRONGEST memorization effect among tested tasks
- Ideal for studying knowledge-intensive memorization patterns

---

## Dataset 4: Modular Arithmetic (Grokking Experiments)

### Overview
- **Source**: Generated locally (see code below)
- **Size**: Addition: train=4,704, test=4,705; Division: train=4,656, test=4,656
- **Format**: JSON
- **Task**: Modular arithmetic (a op b mod p, p=97)
- **Features**: `a`, `b`, `result`, `operation`
- **License**: N/A (synthetic)

### Generation Code
```python
import random, json

p = 97  # prime modulus
data = [{"a": a, "b": b, "result": (a + b) % p, "operation": "add"} for a in range(p) for b in range(p)]
random.seed(42)
random.shuffle(data)
split = int(0.5 * len(data))
# Save train/test splits
```

### Loading
```python
import json
with open("datasets/modular_arithmetic/train.json") as f:
    train = json.load(f)
```

### Notes
- Reproduces the grokking setup from Power et al. (2022)
- 50/50 train/test split as in original paper
- Both addition and division operations available
- Key for studying memorization-to-generalization transition

---

## Dataset 5: The Pile (Streaming Access)

### Overview
- **Source**: HuggingFace `EleutherAI/the_pile_deduplicated`
- **Size**: ~800GB text, ~207B tokens (deduplicated version)
- **Format**: Streaming HuggingFace Dataset
- **Task**: Language modeling pretraining corpus
- **Features**: `text`
- **License**: Various (multi-source corpus)

### Loading (Streaming — no full download needed)
```python
from datasets import load_dataset
dataset = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)
for example in dataset:
    print(example["text"][:200])
    break
```

### Notes
- Training data for Pythia model suite
- Both standard and deduplicated versions available
- Use streaming to avoid downloading full 800GB
- Essential for studying deduplication effects on memorization

---

## Dataset 6: C4 (Streaming Access)

### Overview
- **Source**: HuggingFace `allenai/c4` (en config)
- **Size**: ~305GB text
- **Format**: Streaming HuggingFace Dataset
- **Task**: Language modeling pretraining corpus
- **Features**: `text`, `timestamp`, `url`
- **License**: ODC-BY

### Loading (Streaming)
```python
from datasets import load_dataset
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
```

### Notes
- Used in SemDeDup and deduplication studies
- Lee et al. found a single 61-word sentence repeated >60,000 times
- Good for deduplication experiments
