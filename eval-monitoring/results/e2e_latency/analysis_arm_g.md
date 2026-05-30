# E2E Latency Analysis — Arm G (BF16 + QLoRA) vs Arm A baseline

**Date:** 2026-05-29  
**Script:** `eval-monitoring/e2e_latency.py`  
**Input:** 30 synthesised Hindi WAVs (same set as Arm A run)

---

## Warm-request comparison (excluding nw01 cold-start)

| Stage | Arm A | Arm G | Delta |
|---|---|---|---|
| ASR | 1.010s | 1.006s | −0.004s |
| LLM | 4.551s | **3.262s** | **−1.289s** |
| TTS | 0.186s | 0.169s | −0.017s |
| **E2E mean** | 5.797s | **4.713s** | **−1.084s** |
| **E2E p50** | 5.381s | **4.568s** | **−0.813s** |
| **E2E p95** | 11.028s | **6.514s** | **−4.514s** |

**Arm G is faster on every stage.** LLM is the biggest gain: −1.3s mean.

---

## Why Arm G's LLM is 1.3s faster

Two compounding effects:

1. **BF16 vs nf4**: BF16 has no dequantisation overhead per token. Arm A's nf4 pays
   a per-token compute penalty that adds up over a full response.

2. **Shorter answers**: Arm G's QLoRA training produces tighter Bhojpuri responses.
   Arm A generates longer, more verbose Hindi answers — more tokens → more time.
   The effect is largest on general_qa (−1.45s), where Arm A's advisory-path answers
   are longest.

| Category | A LLM | G LLM | Delta |
|---|---|---|---|
| weather | 4.41s | 3.65s | −0.76s |
| current_facts | 4.51s | 3.37s | −1.14s |
| general_qa | 4.60s | **3.15s** | **−1.45s** |

---

## p95 improvement: 11.0s → 6.5s

Arm A p95 was driven by advisory-length answers (mustard farming: 10.6s LLM,
electricity bill: 11.7s LLM). Arm G produces shorter answers for the same questions,
keeping p95 at 6.5s — a 4.5s improvement at the tail.

This is significant for voice UX: the worst-case user experience improved from 11s to 6.5s.

---

## Routing accuracy

| | Post-ASR routing | Misses |
|---|---|---|
| Arm A | 27/30 (90%) | nw04 (ASR variant), ncf04 (keyword gap), ngq10 (ambiguity) |
| **Arm G** | **29/30 (96.7%)** | ngq10 only |

Two of Arm A's three misses are fixed in the arm-g branch:
- nw04: "आन्धि" spelling variant now in weather keywords
- ncf04: "मुखिया" now in current_facts keywords

Only ngq10 ("गर्मी में लू") remains — known ambiguity, left open.

---

## ncf04 tool overhead spike (10.9s)

ncf04 routed correctly to current_facts for the first time (मुखिया fix active) and
triggered a live Tavily call (cache miss — this fact_key had never been fetched before).
Tool overhead = 10.9 − 1.18 − 3.14 − 0.17 ≈ **6.4s** (Tavily live fetch).

Subsequent runs will hit the disk cache (~5ms). This is expected behaviour, not a regression.

---

## Cold-start

Both arms show identical cold-start penalty on nw01 (~14.8s ASR + ~2.4s TTS).
`warmup_pipeline()` is committed but Arm G's Colab was started before the fix was pulled.
Next session with `warmup_pipeline()` active should eliminate this.

---

## Summary: Arm A vs Arm G e2e

| Metric | Arm A | Arm G | Winner |
|---|---|---|---|
| E2E p50 (warm) | 5.38s | **4.57s** | G −0.81s |
| E2E p95 (warm) | 11.03s | **6.51s** | G −4.51s |
| LLM mean | 4.55s | **3.26s** | G −1.29s |
| Routing (post-ASR) | 90% | **96.7%** | G +2 correct |
| Cold-start penalty | ~17s | ~17s | tie (warmup not yet active) |

Arm G wins on all e2e latency metrics. The quality improvements from QLoRA
(Bhojpuri, shorter answers) directly translate into better latency at runtime.
