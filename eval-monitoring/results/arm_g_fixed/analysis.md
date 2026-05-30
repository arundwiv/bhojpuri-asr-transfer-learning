# Arm G + Tavily — Post-fix Eval Analysis

**Branch:** `experiment/arm-g-qlora`  
**Commit:** 759a8fe (Arabic fix + Bhojpuri prompt strengthening)  
**Date:** 2026-05-29  
**Eval prompts:** 50 (all categories)

---

## Arabic Artefact Fix — Confirmed

| | Before (arm_g_tavily) | After (arm_g_fixed) |
|---|---|---|
| Answers with Arabic chars | 3/10 (cf03, cf07, cf10) | **0/10** ✅ |

The old artefact `جنرل` (Arabic "General") came from the Tavily search snippet for
Bihar Governor (Lt. Gen. Syed Ata Hasnain). The cache had since refreshed to a clean
Hindi Devanagari answer; `clean_tavily_snippet` and the `clean_llm_output` Arabic strip
now prevent any recurrence regardless of cache state.

---

## Bhojpuri Verb Form — Improved

| Verb | Before | After |
|---|---|---|
| बाड़न (Bhojpuri honorific) | 5/10 | **9/10** ✅ |
| बानी (Bhojpuri 1st-person) | 4/10 | 5/10 |
| हैं (Hindi) | 0/10 | 0/10 ✅ |

Adding explicit verb-form examples (`X बाड़न।`) to the prompt moved बाड़न from 5 to 9
out of 10 current_facts answers.

---

## Latency

| Arm | cf avg LLM | cf avg total |
|---|---|---|
| arm_g_tavily (baseline) | 2.77s | — |
| arm_g_fixed (first run) | 4.0s | — |
| arm_g_fixed (v2, +length cap) | TBD | TBD |

First run introduced +1.2s regression: the length constraint ("1-2 short sentences")
was accidentally dropped from the prompt, causing the model to generate 2-sentence
answers with extra date context. Re-added in follow-up commit.

---

## Remaining Issues

| Issue | Cause | Status |
|---|---|---|
| "जनरल" → "रल" in governor answers | QLoRA model proper noun corruption, pre-existing | Open — not caused by this fix |
| Surname truncation in Bihar CM (cf01, cf04, cf06) | Model drops middle+last name | Open |
| "202४" mixed-script year | Model generates mixed Latin+Devanagari digits | Open |
| "प्रधानमण्ट्री" corruption | Model mangles compound consonant | Open — pre-existing |

The "रल" issue was masked by the Arabic artefact in the previous eval. Both are proper
noun corruptions by the fine-tuned model; root fix requires additional QLoRA training data
covering Lt. Gen. rank names.

---

## Overall picture (all arms)

| Metric | Arm A | Arm D | Arm G | Arm G+Tavily | Arm G+Fixed |
|---|---|---|---|---|---|
| Bhojpuri quality (overall) | 1.70 | 1.84 | 2.18 | 2.18 | ~2.30 (est.) |
| cf Bhojpuri verb form | — | — | — | 5/10 बाड़न | **9/10 बाड़न** |
| cf Arabic artefact | — | — | — | 3/10 | **0/10** |
| Avg LLM latency | 3.45s | 2.47s | 2.62s | 2.54s | TBD (v2) |
