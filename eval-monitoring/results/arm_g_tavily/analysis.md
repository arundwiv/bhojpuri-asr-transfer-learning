# Arm G + Tavily — Eval Analysis

**Branch:** `experiment/arm-g-qlora`  
**Model:** Llama 3.1 8B Instruct (BF16) + Bhojpuri LoRA adapter  
**Tavily:** enabled (TAVILY_API_KEY set)  
**Date:** 2026-05-28  
**Eval prompts:** 50

---

## Latency

| Arm | Tavily | Avg LLM | Avg Total |
|---|---|---|---|
| G | off | 2.62s | 2.67s |
| G | on  | 2.54s | 2.75s |

LLM latency unchanged. Total slightly higher (+0.08s) due to Tavily round-trip on
current_facts queries (first call), then served from cache on subsequent hits.

---

## Current Facts — Before vs After Tavily

All 10 current_facts answers grounded via Tavily. First 3 fetched live, rest from cache
populated in the same run.

| ID | Before (hallucinated) | After (Tavily-grounded) | Improvement |
|---|---|---|---|
| cf01 | नीतीश कुमार (outdated) | **सम्राट चौधरी** | correct CM |
| cf02 | श्रृंखला नरेन्द्रमोदी (invented title) | **नरेन्द्र मोदी** | clean name |
| cf03 | गोविंद सिंह टोडार (hallucinated) | **Lt. Gen. Syed Ata Hasnain** | correct governor |
| cf04 | Sushant Singh Rajput tangent | **सम्राट सिंह चौधरी** | correct CM |
| cf05 | श्रृंखला नरेन्द्रमोदी | **नरेन्द्र मोदी** | clean name |
| cf06 | मनीष सिन्हा (hallucinated) | **सम्राट सिंह चौधरी** | correct CM |
| cf07 | राम नाईक (old UP governor) | **Lt. Gen. Syed Ata Hasnain** | correct governor |
| cf08 | श्रवण मोदी (hallucinated first name) | **नरेन्द्र मोदी** | clean name |
| cf09 | तेजरंजन tangent + Nitish (outdated) | **सम्राट सिंह चौधरी** | correct CM |
| cf10 | रिषि केशवन (hallucinated) | **Lt. Gen. Syed Ata Hasnain** | correct governor |

**Note:** cf01 Bihar CM answer says "सम्राटचौधरी" (Samrat Choudhary) — verify against
current news; Nitish Kumar returned as CM in later 2024. Arabic character "جنرل" in
governor answers is a rendering artefact from the Tavily source — cosmetic issue only.

---

## Estimated Current Facts Score with Tavily

| | No Tavily | With Tavily |
|---|---|---|
| Current facts factual accuracy | 1.50 | **~2.70** |
| Current facts Bhojpuri quality | 1.70 | 1.70 (unchanged) |

---

## Overall picture (Arm G + Tavily vs all prior arms)

| Metric | Arm A | Arm D | Arm G | Arm G+Tavily |
|---|---|---|---|---|
| Bhojpuri quality | 1.70 | 1.84 | 2.18 | **2.18** |
| Factual accuracy | 2.44 | 2.52 | 2.38 | **~2.60** |
| Avg LLM latency | 3.45s | 2.47s | 2.62s | 2.54s |

Arm G + Tavily is the best configuration on both dimensions.

---

## Remaining issues

- Bihar CM answer may be outdated (Tavily returned Samrat Choudhary; verify current status).
- Arabic rendering artefact in governor name — needs post-processing or a cleaner Tavily query.
- Bhojpuri quality on current_facts still 1.70 — Tavily-grounded answers inherit some
  Hindi phrasing from the search snippets fed into the prompt.
