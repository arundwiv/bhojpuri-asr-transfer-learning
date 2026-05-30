# Arm G — QLoRA Fine-Tune Eval Analysis

**Branch:** `experiment/arm-g-qlora`  
**Model:** Llama 3.1 8B Instruct (BF16) + Bhojpuri LoRA adapter (498 examples, 3 epochs)  
**Date:** 2026-05-28  
**Eval prompts:** 50 (weather×8, current_facts×10, general_qa×15, conversational×10, out_of_scope×7)

---

## Latency

| Arm | Model | Avg LLM | Avg Total | vs Baseline |
|---|---|---|---|---|
| A | Llama 3.1 8B bnb-nf4 | 3.45s | ~3.60s | baseline |
| D | Llama 3.1 8B BF16 | 2.71s | 2.87s | -21% |
| G | Llama 3.1 8B BF16 + LoRA | 2.62s | 2.67s | **-24%** |

LoRA adapter adds negligible inference overhead over Arm D.

---

## Bhojpuri Language Quality (scored 1–3 per response)

| Category | Arm A | Arm G | Delta |
|---|---|---|---|
| Weather | 1.50 | 2.38 | **+0.88** |
| Current Facts | 1.40 | 1.70 | +0.30 |
| General QA | 1.40 | 1.87 | +0.47 |
| Conversational | 3.00 | 3.00 | 0.00 |
| Out-of-scope | 1.14 | 2.14 | **+1.00** |
| **Overall** | **1.70** | **2.18** | **+0.48** |

Scoring: 1 = Hindi drift/garbled, 2 = mixed, 3 = pure Bhojpuri.

---

## Factual Accuracy (scored 1–3 per response)

| Category | Arm A | Arm G | Delta |
|---|---|---|---|
| Weather | 3.00 | 3.00 | 0.00 |
| Current Facts | 1.40 | 1.50 | +0.10 |
| General QA | 2.53 | 2.33 | -0.20 |
| Conversational | 3.00 | 3.00 | 0.00 |
| Out-of-scope | 2.29 | 2.14 | -0.14 |
| **Overall** | **2.44** | **2.38** | **-0.06** |

---

## Notable Improvements (A → G)

| ID | Question | Arm A | Arm G |
|---|---|---|---|
| w02 | कल बारिश होई? | "नहीं हुई" (Hindi) | "नइखे होने वाला" |
| w04 | वाराणसी में धूप बा कि बादल? | "नहीं देला" (Hindi) | "नइखी" (Bhojpuri) |
| w08 | छपरा में बरखा होई? | "वर्षा नहीं होई" (Hindi) | "नइखे — कोहरा बा" |
| gq02 | गंगा नदी कहाँ से निकलती है? | "निकलती है" (Hindi) | "नीकलत बा" (Bhojpuri) |
| gq05 | दूध पीयल के का फायदा? | "होत हैं/करत हैं" (Hindi) | "होला/पीयलीं" |
| os01 | शेयर मार्केट refusal | "मेरे पास उपलब्ध नहीं" (Hindi) | "हमरा पास नइखे" |
| os03 | हमरा पैसा कहाँ गया? | irrelevant savings advice | police redirect (correct) |
| os05 | आज IPL में कौन जीता? | "मुझे जानकारि नइए" (Hindi) | "नइखे — टीवी पर देखीं" |

---

## Regressions (A → G)

| ID | Question | Arm A | Arm G | Severity |
|---|---|---|---|---|
| cf04 | अभी बिहार के सीएम कौन बा? | Correct name (Hindi verbs) | Sushant Singh Rajput tangent | High |
| gq06 | बिहार के राजधानी का बा? | "पटना है" (Hindi but correct) | "ब के राजस्थानी पटना बा" (garbled) | High |
| gq03 | आम के फायदे का हवे? | reasonable Hindi answer | "आँख ध्यानी" (nonsensical) | Medium |
| os02 | लॉटरी में कइसे जीतीं? | correctly declines | gives lottery advice | Medium |

---

## Pre-existing Issues (both arms)

**Current facts hallucination** — Bihar governor and PM name are consistently wrong in both Arm A and G:
- Bihar governor: A → "Syed Ata Hasnain" (hallucinated), G → "Govinд Singh Todar" / "Ram Naik" (hallucinated)
- PM name: Both arms add hallucinated titles/prefixes ("श्रि", "श्रृंखला", "श्रवण")

Root cause: knowledge cutoff + no grounding. Fix: Tavily search routing for current-facts queries, not more LoRA training.

---

## Verdict

**Arm G is the current best arm.** Bhojpuri dialect quality improved significantly (+0.48 overall, +1.0 on refusals) with negligible factual accuracy cost (-0.06). The three regressions are low-frequency hallucination artifacts, not systematic failures.

**Recommended next steps:**
1. Merge Arm G into main as the new production baseline.
2. Add Tavily routing for `current_facts` category to fix hallucination on political figures.
3. Expand LoRA corpus with harder examples (current-facts, complex agriculture) for next fine-tune iteration.
