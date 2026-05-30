# E2E Latency Analysis — Arm A (bnb-nf4, main branch)

**Date:** 2026-05-29  
**Script:** `eval-monitoring/e2e_latency.py`  
**Input:** 30 synthesised Hindi WAVs (edge-tts hi-IN-SwaraNeural → 16 kHz mono)  
**Pipeline:** WAV → ASR → router → tool → LLM → TTS → `/chat_debug`

---

## Warm-request latency (excluding nw01 cold-start)

| Stage | p50 | p95 | mean | % of total |
|---|---|---|---|---|
| ASR | 1.056s | 1.268s | 1.010s | **17.4%** |
| True tool overhead* | — | — | 0.128s | 2.2% |
| LLM | — | — | 4.551s | **78.5%** |
| TTS | — | — | 0.186s | 3.2% |
| **Total** | **5.38s** | **11.03s** | **5.80s** | — |

*True tool overhead = `tool_seconds − llm_seconds`. The `/chat_debug` endpoint wraps both tool and LLM into one timer; this separates them.

**LLM is the dominant bottleneck at 78.5% of total latency.**

---

## Cold-start finding

| Request | ASR time | Total |
|---|---|---|
| nw01 (1st ever) | **14.52s** | 22.2s |
| nw02 onwards | 1.06s p50 | 5.4s p50 |

The first Whisper inference after Colab startup takes ~14s (model warm-up + GPU memory
allocation). Subsequent calls stabilise at ~1.06s. TTS shows the same pattern: nw01 TTS =
3.2s vs 0.18s for all others (Kokoro first-inference overhead).

**Fix**: call `maybe_warmup_tts()` at startup and add a silent Whisper dummy pass to
pre-warm the GPU before serving traffic. Estimated saving: ~13s off first real user request.

---

## Tool overhead

| Category | Mean tool overhead | Note |
|---|---|---|
| weather | ~0.19s | WeatherAPI cached after first per-location call |
| current_facts | ~0.1s | Tavily/disk cache hits (ncf01: 0.61s, rest ~5ms) |
| general_qa | ~0s | No tool call |

Tool overhead is negligible when caches are warm. Tavily adds ~0.6s on cache miss (first
call per fact_key per session); subsequent hits are disk reads at ~5ms.

---

## p95 = 11s: the advisory-length driver

The p95 (11s) is driven by questions that trigger `MAX_NEW_TOKENS_ADVISORY = 300`:
- ngq07 (mustard farming): LLM = 10.6s
- ngq09 (electricity bill): LLM = 11.7s
- ngq02 (cow milk): LLM = 7.0s

These are `how-to` / `benefits` questions hitting advisory keywords ("कैसे", "फायदा",
"उपाय"). The current ceiling of 300 tokens produces ~3× longer answers and ~2.5× longer
LLM times vs simple QA (128-token ceiling).

For a voice assistant, 11s is unacceptable. Consider lowering `MAX_NEW_TOKENS_ADVISORY`
to 200 or adding a voice-specific token budget.

---

## NEW finding: ASR error caused routing miss

The text-only eval showed 27/30 routing (90%). E2E eval shows the **same** 27/30 — but
the routing misses are partially different causes:

| ID | ASR transcript | Expected route | Actual | Cause |
|---|---|---|---|---|
| nw04 | "आज बनारस में **आन्धि** आई का" | weather | general_qa | **ASR spelled "आंधी" → "आन्धि"** — not in weather keywords |
| ncf04 | "आजकल **बेहार** के मुखिया…" | current_facts | general_qa | router keyword gap (fixed in arm-g branch) |
| ngq10 | correct | general_qa | weather | known ambiguity (गर्मी) |

nw04 is a new finding from e2e: ASR transcribed "आंधी" (storm) as "आन्धि", a variant
spelling not present in the weather keywords. The text eval never surfaced this because it
bypasses ASR entirely.

**Fix**: add "आन्धि", "आँधी" to weather keywords as ASR spelling variants.

---

## Routing accuracy post-ASR

27/30 (90.0%) — identical to the text eval score. No additional routing degradation
from ASR on these 30 questions (beyond the one ASR-variant miss above).

---

## Baseline summary for arm comparison

| Metric | Arm A |
|---|---|
| E2E p50 (warm) | 5.38s |
| E2E p95 (warm) | 11.03s |
| ASR p50 | 1.06s |
| LLM mean | 4.55s |
| TTS mean | 0.19s |
| Post-ASR routing | 27/30 (90%) |
| Cold-start penalty | +13s ASR, +3s TTS |
