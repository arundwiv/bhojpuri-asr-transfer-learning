# Arm G — Blind Eval Analysis (30 fresh questions)

**Branch:** `experiment/arm-g-qlora`  
**Script:** `eval-monitoring/llm_eval_blind.py`  
**Date:** 2026-05-29  
**Purpose:** Validate that quality improvements on the original 50-prompt set generalise
to unseen question phrasings and topics.

---

## Summary

| Metric | Value |
|---|---|
| Total prompts | 30 |
| OK / failed | 30 / 0 |
| Routing correct | 27 / 30 (90.0%) |
| Avg LLM latency | 3.2s |
| Arabic artefact | **NONE** ✅ |
| बाड़न verb form (cf, routed correctly) | **4/4** ✅ |
| हैं (Hindi leakage) | 0 |

---

## Routing Misses (3/30)

| ID | Question | Expected | Got | Impact |
|---|---|---|---|---|
| ncf04 | आजकल बिहार के मुखिया कवन बाड़न? | current_facts | general_qa | **High** — model gave wrong CM (Nitish Kumar instead of Samrat Choudhary via training data) |
| ncf05 | भारत के PM के का नाम बा? | current_facts | general_qa | High — answer severely corrupted ("भ के नरेन्द्र मोदी") |
| ngq10 | गर्मी में लू से बचे के का करीं? | general_qa | weather | Low — answer mixed weather + health tip, partially usable |

**ncf04 root cause:** "मुखिया" (Bhojpuri word for leader/CM) is not in the router keywords.
Fixed: "मुखिया" added to `current_facts` keywords.

**ncf05 root cause:** "PM" (Latin) not matched — router has "पीएम" (Devanagari) but not
the Latin abbreviation. Fixed: "pm" added to keywords (lowercased matching handles case).

**ngq10 root cause:** "गर्मी" in weather keywords fires on a heatstroke-prevention question.
Ambiguous — left open for now.

---

## Quality Observations

**What held on fresh questions:**
- Arabic artefact: zero occurrences across all 30 answers ✅
- Bhojpuri बाड़न verb form: 4/4 correctly-routed current_facts answers ✅
- Out-of-scope refusals: all 4 correctly declined to guess (nos01–nos04)

**New issues surfaced:**
- ngq04 "त के पता खाए से" — severe start corruption (तुलसी → "त"); model dropped the
  subject noun from the first word
- ngq08 dal cooking instructions: completely wrong (described something else)
- ngq07 mustard cultivation: wrong technique (transplanting, not direct seeding)
- nw01 "(īā)" Latin transliteration artefact in Ballia weather answer — from city alias map
- Governor answers (ncf03, ncf06): "जनरल" → "रल" corruption persists (pre-existing)

**Wrong fact when misrouted:**
ncf04 returned "नीतीश कुमार" (stale training data) instead of current CM because it
bypassed Tavily. This confirms routing gaps cause factual regressions — not a model
quality issue, a router coverage issue.

---

## Comparison baseline (Arm G, original 50-prompt set)

| Metric | Original 50 set | Blind 30 set |
|---|---|---|
| Routing accuracy | 100% | 90% (3 misses, 2 fixable) |
| Arabic artefact | 0/10 cf | 0/6 cf ✅ |
| बाड़न (cf, correctly routed) | 9/10 | 4/4 |
| General QA quality | mixed | mixed (new corruptions on fresh topics) |

Routing accuracy drop (100% → 90%) is due to two lexical gaps in the router
("मुखिया", "pm") — not a model generalisation failure. Fixed in same commit.

---

## To run Arm A / Arm D comparison

Switch the Colab notebook to the target branch, restart runtime, then:

```bash
# Arm A (main branch — bnb-nf4 baseline)
python eval-monitoring/llm_eval_blind.py \
  --backend https://hypsometrical-unsnobbishly-queen.ngrok-free.dev \
  --out_dir eval-monitoring/results/arm_a_blind

# Arm D (experiment/llm-int4-marlin — BF16, no QLoRA)
python eval-monitoring/llm_eval_blind.py \
  --backend https://hypsometrical-unsnobbishly-queen.ngrok-free.dev \
  --out_dir eval-monitoring/results/arm_d_blind
```

Results land in the respective `*_blind/` directories with the same CSV/JSONL schema.
