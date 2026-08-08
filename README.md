# TCCBench: A Benchmark for Constrained Language Organization via Unordered Token Assembly

TCCBench is a benchmark inspired by the *Thousand Character Classic* (Qianziwen) that evaluates large language models (LLMs) through an unordered token assembly task. Given a shuffled multiset of characters or words extracted from a valid sentence, models must reconstruct a fluent and factually consistent sentence using **exactly** the provided elements — no more, no less.

The benchmark contains **7,200 bilingual (Chinese/English) samples** covering six disciplines: physics, chemistry, biology, geography, history, and daily-life text. Evaluation uses a multi-dimensional protocol (CQF) that jointly measures constraint compliance, generation quality, and factual consistency.

## Repository Structure

```
TCCBench/
├── data/
│   └── TCCBench.jsonl                 # Final 7,200-sample dataset
├── src/
│   ├── dataset_construction/          # Pipeline for building the benchmark
│   │   ├── fetch_wikipedia.py         # Fetch Wikipedia articles via outlines
│   │   ├── clean_rawdata.py           # Clean and segment raw text
│   │   ├── screen_sentences_en.py     # 3-agent screening (English)
│   │   ├── screen_sentences_cn.py     # 3-agent screening (Chinese)
│   │   ├── classify_daily_life.py     # Classify daily-life sentences
│   │   ├── deduplicate.py             # Jaccard-similarity de-duplication
│   │   ├── split_by_length.py         # Stratify by sequence-length level
│   │   ├── dataset_statistics.py      # Compute dataset statistics
│   │   ├── tokenize_shuffle.py        # Tokenize sentences and shuffle tokens
│   │   └── merge_dataset.py           # Merge discipline files into final dataset
│   ├── inference/                     # Run LLMs on the benchmark
│   │   ├── run_deepseek.py            # DeepSeek-V3.2
│   │   ├── run_deepseek_think.py      # DeepSeek-V3.2 (thinking mode)
│   │   ├── run_deepseek_t0.py         # DeepSeek-V3.2 (greedy, T=0)
│   │   ├── run_gpt.py                 # GPT-5.2
│   │   ├── run_glm.py                 # GLM-4.7
│   │   ├── run_glm_thinking.py        # GLM-4.7 (thinking mode)
│   │   └── run_qwen.py               # Qwen3-max
│   ├── evaluation/                    # CQF evaluation protocol
│   │   ├── run_evaluation.py          # 3-agent evaluation pipeline
│   │   ├── compliance_evaluator.py    # Multiset-matching compliance (F1)
│   │   └── score_fusion.py            # Fuse compliance × quality × factuality
├── config/
│   └── config.example.yaml            # Example configuration (no real keys)

```

## Dataset Format

Each line in `data/TCCBench.jsonl` is a JSON object:

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Stable sample identifier |
| `discipline` | string | Source domain (daily life, Biology, Chemistry, Physics, History, Geography) |
| `language` | string | `"en"` or `"zh"` |
| `original_sentence` | string | The source sentence before discretization |
| `word_count` | string | Number of tokens in the reference sentence |
| `word_only_once` | string | `"yes"` if all tokens are distinct, `"no"` otherwise |
| `word_list` | list[string] | The shuffled token multiset shown to the model |

**Statistics:** 7,200 samples = 600 × 12 (6 disciplines × 2 languages). Sequence lengths span 4–65 tokens across six levels. 1,727 samples (24.0%) have all-distinct tokens; 5,473 (76.0%) contain repeats.

## Installation

```bash
pip install -r requirements.txt
```

Python ≥ 3.10 is required. For Chinese text processing, `jieba` and `opencc` are needed. For GLM model access, the `zai` package provides the ZhipuAI client.

## Usage

### 1. Dataset Construction

The dataset was built through a multi-stage pipeline:

```
fetch_wikipedia.py → clean_rawdata.py → screen_sentences_*.py
    → deduplicate.py → split_by_length.py → tokenize_shuffle.py
    → merge_dataset.py
```

- **Fetch:** For each discipline, an LLM generates a hierarchical outline; each keyword maps to a Wikipedia page whose body is extracted. Daily-life text is sampled from CC-Stories (English) and the Zh-hans corpus (Chinese).
- **Screen:** Three LLM agents independently screen each candidate sentence for grammaticality, fluency, and factual correctness.
- **De-duplicate:** Exact duplicate removal followed by structural de-duplication using token-set Jaccard similarity (threshold τ = 0.95). Chinese text is segmented with `jieba`; English text is whitespace-tokenized after punctuation removal.
- **Tokenize & Shuffle:** Sentences are tokenized (character-level for Chinese, word-level for English) with protected entities (chemical formulae, mathematical expressions, proper nouns). Tokens are then randomly shuffled using a cryptographically secure seed.
- **Merge:** The water-filling sampler balances sequence length, repeated-token prevalence, and keyword coverage, yielding 600 samples per language–discipline cell.

To re-run the pipeline, update the file paths in each script's configuration section and execute them in order.

### 2. LLM Inference

Each script in `src/inference/` runs one model on the benchmark. Before running, replace `YOUR_API_KEY` with your actual API key and update the input/output paths:

```python
API_CONFIG = {
    "api_key": "YOUR_API_KEY",       # Replace with your key
    "base_url": "https://api.deepseek.com/v1/",
    "model_name": "deepseek-chat"
}

INPUT_PATH = "data/TCCBench.jsonl"
OUTPUT_PATH = "output/TCCBench_deepseek.jsonl"
```

The inference scripts support:
- Bilingual prompts (Chinese and English)
- Resume from checkpoint (appends to existing output file)
- Configurable retry strategy and concurrency

### 3. Response Postprocessing

Model responses are tokenized using the same tokenizer as the dataset:

```bash
python src/postprocessing/split_response.py
```

This adds a `response_list` field to each output record, enabling multiset-matching compliance computation.

### 4. Evaluation

The evaluation pipeline (`src/evaluation/run_evaluation.py`) computes three scores per sample:

| Score | Range | Method |
|-------|-------|--------|
| **Compliance** ($S_C$) | [0, 1] | F1 over exact multiset matching between input tokens and response tokens |
| **Quality** ($S_Q$) | [0, 100] | Mean of three LLM-agent scores (0.6 × grammar + 0.4 × coherence) |
| **Factuality** ($S_F$) | {0, 1} | Majority vote of three agents (PASS/FAIL); gated by $S_Q \geq 50$ |

The overall **CQF** score is:

$$S_{\text{CQF}} = S_C \cdot S_Q \cdot S_F$$

where $S_C$ is a proportional multiplier, $S_Q$ is the quality score, and $S_F$ is a binary veto.

To run evaluation, configure the three judge agents in the script:

```python
LLM_AGENTS = {
    "agent01": {"api_key": "YOUR_API_KEY", "base_url": "...", "model": "gpt-5-mini"},
    "agent02": {"api_key": "YOUR_API_KEY", "base_url": "...", "model": "deepseek-chat"},
    "agent03": {"api_key": "YOUR_API_KEY", "base_url": "...", "model": "qwen-plus"},
}
```

All judges use temperature $T = 0$ for evaluation consistency.

### 5. Analysis

- `compute_statistics.py`: Recomputes all paper metrics (main table, per-language, per-level, compliance intervals, inter-judge agreement, etc.) from raw evaluation JSONL files.
- `generate_figures.py`: Produces all paper figures (per-model scores, compliance vs. quality, error analysis, difficulty curves, etc.).
- `extract_case_studies.py`: Extracts representative best-case, failure, and creative-reorganization examples.

## Evaluated Models

| Model | API Provider | Model ID |
|-------|-------------|----------|
| DeepSeek-V3.2 | DeepSeek | `deepseek-chat` |
| DS-V3.2 (Think) | DeepSeek | `deepseek-reasoner` |
| Qwen3-max | DashScope | `qwen-max` |
| GLM-4.7 | ZhipuAI | `glm-4.7` |
| GPT-5.2 | OpenAI | `gpt-5.2` |

All models were accessed via official APIs in May 2026 using sampling (temperature 0.7, top-p 1.0), zero-shot, with language-specific prompts stating the exact token-level constraint.

## Key Findings

- The best model achieves only **32.2 CQF** out of 100, showing the task is challenging.
- A **precision–quality pattern** exists among non-reasoning models: higher compliance correlates with lower quality.
- **Reasoning** improves compliance (8.8% → 46.0% perfect-compliance rate) but not fluency.
- Chinese character-level assembly is **harder** than English word-level assembly for every model.

## Notes

- All API keys in the released code have been replaced with `YOUR_API_KEY`. Users must provide their own keys.
- Hardcoded file paths in the scripts reflect the original development environment. Update them to match your local setup before running.
- Evaluation results and raw model outputs are not included in this release; only the dataset and code are provided.

## Citation

If you use TCCBench in your research, please cite:

```bibtex
@inproceedings{tccbench2027,
  title     = {TCCBench: A Benchmark for Constrained Language Organization via Unordered Token Assembly},
  author    = {Anonymous Submission},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2027}
}
```

## License

This dataset and code are released for research purposes.
