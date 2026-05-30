# Arm D — BF16 Baseline Eval Analysis

**Branch:** `experiment/arm-d`  
**Model:** Llama 3.1 8B Instruct (BF16, no quantization, no fine-tuning)  
**Date:** 2026-05-28  
**Eval prompts:** 50 (weather×8, current_facts×10, general_qa×15, conversational×10, out_of_scope×7)

---

## Latency

| Arm | Model | Avg LLM | Avg Total | vs Baseline |
|---|---|---|---|---|
| A | Llama 3.1 8B bnb-nf4 | 3.45s | ~3.60s | baseline |
| D | Llama 3.1 8B BF16 | 2.47s | 2.50s | **-28%** |

---

## Bhojpuri Language Quality (scored 1–3)

| Category | Arm A | Arm D | Delta |
|---|---|---|---|
| Weather | 1.50 | 2.00 | +0.50 |
| Current Facts | 1.40 | 1.50 | +0.10 |
| General QA | 1.40 | 1.47 | +0.07 |
| Conversational | 3.00 | 3.00 | 0.00 |
| Out-of-scope | 1.14 | 1.29 | +0.15 |
| **Overall** | **1.70** | **1.84** | **+0.14** |

BF16 precision gives modest dialect improvement over 4-bit quantization, but Hindi drift
remains prevalent in general QA and out-of-scope categories.

---

## Factual Accuracy (scored 1–3)

| Category | Arm A | Arm D | Delta |
|---|---|---|---|
| Weather | 3.00 | 3.00 | 0.00 |
| Current Facts | 1.40 | 2.10 | **+0.70** |
| General QA | 2.53 | 2.40 | -0.13 |
| Conversational | 3.00 | 3.00 | 0.00 |
| Out-of-scope | 2.29 | 2.14 | -0.15 |
| **Overall** | **2.44** | **2.52** | **+0.08** |

BF16 notably improves current-facts accuracy (+0.70) — the model correctly declines
to answer when it doesn't know (Bihar governor, some CM variants) rather than hallucinating.

---

## Notable Behaviours

**Improvements over Arm A:**
- `cf03/cf04/cf06/cf10`: Arm D correctly says "जानकारि उपलब्ध नहीं" rather than
  hallucinating governor/CM names. More conservative and honest.
- `w01/w05/w06`: Weather answers use "रहल बा" (Bhojpuri progressive) rather than pure Hindi.

**Remaining issues:**
- `w02/w08`: "है/है" Hindi leaks in weather answers.
- `gq01/gq02/gq04/gq11/gq13/gq15`: General QA mostly Hindi throughout.
- `os01/os02/os05`: Out-of-scope refusals in Hindi ("नहीं है", "मैं/मुझे").
- `gq12`: Sunrise direction wrong — "सूर्य ग्रह से उगले ला" (invented "सूर्य ग्रह").

---

## Verdict

Arm D is faster than Arm A (-28% latency) and marginally better on both dialect and
factual accuracy. However, the dialect improvement (+0.14) is modest — BF16 precision
alone does not solve Hindi drift. Arm G (LoRA fine-tuned) is the better choice for
production.
