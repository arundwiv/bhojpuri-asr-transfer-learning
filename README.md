# Bhojpuri AI — ASR Transfer Learning & Production Eval Framework

End-to-end AI engineering project: low-resource Bhojpuri ASR (MSc dissertation) extended into a
production voice assistant with a rigorous LLM evaluation framework and multi-arm experiment design.

---

## What's in this repo

| Directory | Contents |
|---|---|
| `notebooks/` | MSc dissertation — IndicWav2Vec fine-tuning, tokenizer analysis, WER/CER evaluation |
| `eval-monitoring/` | LLM evaluation harness, A/B experiment results, Claude API corpus generator |

---

## Part 1 — MSc Dissertation: Bhojpuri ASR via Transfer Learning

Adapts IndicWav2Vec and Whisper models for Bhojpuri automatic speech recognition using transfer
learning on the SpeeD-IA Bhojpuri corpus.

**Pipeline**

1. Dataset preparation and LOSO (leave-one-speaker-out) speaker split
2. Fine-tuning IndicWav2Vec on Bhojpuri speech
3. Evaluation with WER / CER metrics
4. Optional KenLM shallow-fusion decoding for improved robustness

---

## Part 2 — Production LLM Eval Framework & A/B Experiments

The ASR work was extended into a production voice assistant (Android + cloud GPU backend) with a
formal evaluation framework for the LLM component. The eval harness and experiment results are in
`eval-monitoring/`.

### Architecture

```
Android app  →  FastAPI backend (Colab L4 GPU)
                  │
                  ├─ Whisper ASR
                  ├─ Keyword intent router
                  ├─ Tool calls: WeatherAPI / Tavily live search (with TTL cache)
                  └─ LLM (fine-tuned LLaMA-3.1-8B) → Kokoro TTS → SSE stream
```

### Six-arm A/B experiment — quality vs latency

Each arm was evaluated against a fixed 50-prompt test suite (5 categories: weather, current facts,
general QA, conversational, out-of-scope). Scores are 1–3 (1 = wrong, 2 = acceptable, 3 = good).

| Arm | Model | Avg LLM latency | Bhojpuri quality | Factual accuracy | Result |
|---|---|---|---|---|---|
| **A** (baseline) | Llama 3.1 8B NF4 | 3.45 s | 1.70 | 2.44 | baseline |
| D | Llama 3.1 8B BF16 | 2.47 s | 1.84 | 2.52 | faster; more honest on unknowns |
| E | Llama 3.1 8B GPTQ gs32 | 2.01–2.29 s | — | — | fails — noun corruption |
| F | Qwen2.5-7B BF16 | 3.26 s | — | — | fails — garbled Bhojpuri output |
| **G** | Llama 3.1 8B BF16 + QLoRA | 2.62 s | 2.18 | 2.38 | +28% dialect quality vs baseline |
| **G+Tavily** ✓ | Llama 3.1 8B BF16 + QLoRA + live search | 2.54 s | 2.18 | ~2.60 | best overall |

**Key finding:** Tavily grounding eliminates current-facts hallucination (factual accuracy on
current-events queries: 1.50 → ~2.70). QLoRA fine-tune improves Bhojpuri dialect consistency
(+28% quality score) but cannot fix knowledge cutoff — requires live retrieval.

### Blind validation set

To check for overfitting to the 50-prompt development set, a separate 30-prompt blind set
(`llm_eval_blind.py`) was held out during all development. Scores on the blind set matched the
development set within ±0.05 on both dimensions, confirming results generalise.

### Eval harness files

| File | Purpose |
|---|---|
| `llm_eval.py` | 50-prompt dev eval — routing accuracy, LLM quality, latency |
| `llm_eval_blind.py` | 30-prompt blind validation set (held out during development) |
| `batch_eval.py` | Batch runner for multi-arm comparison |
| `score_fluency.py` | Bhojpuri dialect fluency scoring |
| `router_audit.py` | Intent routing accuracy audit |
| `e2e_latency.py` | End-to-end latency profiling (ASR / router / tool / LLM / TTS) |
| `latency_report.py` | Latency summary report |
| `smoke_test.py` | Smoke test against live backend |
| `spike_int4_benchmark.py` | INT4 quantisation latency spike benchmark |
| `generate_finetune_corpus.py` | Claude API corpus generator (see below) |
| `bhojpuri_qlora_finetune.ipynb` | QLoRA fine-tuning notebook (Colab, L4 GPU) |
| `results/*/analysis.md` | Per-arm experiment analysis |

### Claude API — synthetic training data generation

The QLoRA fine-tuning corpus (498 Bhojpuri Q&A examples) was generated using the Anthropic Claude
API (`claude-haiku-4-5`). The script in `generate_finetune_corpus.py`:

- Seeds generation from 20 native-speaker-verified gold examples
- Uses **prompt caching** (`cache_control: ephemeral`) to share the gold examples and language
  rules across all batch API calls, reducing token cost
- Generates 129 additional examples across 8 categories (agriculture, health, government schemes,
  general knowledge, conversational, local culture, maths/units, refusal)
- Outputs TRL SFTTrainer format ready for QLoRA fine-tuning

This approach — using a frontier API model to synthesise a domain-specific fine-tuning corpus for
a smaller local model — cut annotation time from weeks to hours while maintaining dialect quality
(verified by a native Bhojpuri speaker).

---

## Running the eval harness

```bash
# Against a live backend (ngrok URL from Colab)
python eval-monitoring/llm_eval.py --backend https://xxx.ngrok-free.app

# Blind validation set
python eval-monitoring/llm_eval_blind.py --backend https://xxx.ngrok-free.app

# End-to-end latency profiling
python eval-monitoring/e2e_latency.py --backend https://xxx.ngrok-free.app
```

Results are written to `eval-monitoring/results/` (CSV + JSONL, gitignored — only analysis.md
files are tracked).

---

## Dataset

Experiments conducted using the SpeeD-IA Bhojpuri corpus.
