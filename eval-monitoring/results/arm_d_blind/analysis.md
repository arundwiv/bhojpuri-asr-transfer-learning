# Arm D — Blind Eval Analysis (30 fresh questions)

**Branch:** `experiment/llm-int4-marlin`  
**Model:** Llama 3.1 8B BF16 (no QLoRA adapter)  
**Script:** `eval-monitoring/llm_eval_blind.py`  
**Date:** 2026-05-29

---

## Summary

| Metric | Arm D | Arm G |
|---|---|---|
| Routing correct | 27/30 (90%) | 27/30 (90%) |
| Avg LLM latency | **2.9s** | 3.2s |
| Arabic artefact | 0 | 0 |
| बाड़न verb form (cf, correctly routed) | 0/4 | **4/4** |
| Bhojpuri-dominant responses | ~60% | **~80%** |

---

## Routing

Same 3 misses as Arm G — this is router code, identical across arms:
- ncf04 "मुखिया" → general_qa (fixed in arm-g-qlora, not yet in this branch)
- ncf05 "PM" → general_qa (same fix pending)
- ngq10 "गर्मी में लू" → weather (open ambiguity)

---

## Head-to-head answer quality

### Current Facts (correctly routed: ncf01, ncf02, ncf03, ncf06)

| ID | Arm D | Arm G | Winner |
|---|---|---|---|
| ncf01 (India PM) | "नरेन्दर मोदी हवें" — older Bhojpuri हवें form | "नरेन्द्रमोदी बाड़न" — बाड़न but fused name | Tie |
| ncf02 (Bihar CM) | "सम्राट चौंधारी…हें" — **full surname** but Hindi-ish verb | "सम्राट बाड़न" — correct verb but **drops surname** | D (name completeness) |
| ncf03 (Governor) | "जनरल सैयद अता हसनैन" — **"जनरल" intact** | "रल सय्य़ाद" — **"जनरल"→"रल" corruption** | D (proper noun) |
| ncf06 (Governor) | same as ncf03 | same as ncf03 | D |

**Misrouted facts:**
- ncf04 (D): "बिहार का मुख्यमंत्री अब तक बदलल रहलें" — hedged, no wrong name ✅
- ncf04 (G): "नीतीश कुमार बाड़न" — wrong fact ❌
- ncf05 (D): "भारत के प्रधानमंत्री नरेंद्र मोदी बा" — correct and clean ✅
- ncf05 (G): "भ के नरेन्द्र मोदी" — corrupted ❌

### General QA — notable differences

| ID | Topic | Arm D | Arm G | Winner |
|---|---|---|---|---|
| ngq03 | Chhath | Wrong (about buying vessels) | Correct description | G |
| ngq04 | Tulsi | Complete, correct | "त के पता" (corrupted start) | D |
| ngq06 | Bihar rivers | Garbled: "ब pahilee pramukh…" | Ganga/Gandak/Kosi ✅ | G |
| ngq07 | Mustard | General correct advice | Wrong (transplanting instead of direct seeding) | D |
| ngq08 | Arhar dal | Reasonable recipe | Wrong instructions | D |
| ngq09 | Electricity bill | Practical (energy-efficient bulbs) | "Call helpline" — unhelpful | D |

### Language quality

Arm D uses "हवें", "हें", "है" — a mix of older Bhojpuri honorific forms and Hindi leakage.
Arm G consistently uses "बाड़न" (modern standard Bhojpuri) — the QLoRA training effect is clear.

Weather answers: D has more Hindi ("नहीं हो रही", "वर्तमान", "है") vs G which uses
"बा", "नइखे" more consistently.

---

## Key findings

1. **Bhojpuri verb form**: QLoRA clearly shifted G from D's mixed हवें/है to consistent
   बाड़न. This is a real improvement.

2. **Proper noun corruption**: G has "जनरल"→"रल" in governor answers; D does not.
   The QLoRA fine-tuning degraded this specific proper noun. Root fix needs training data
   with Lt. Gen. rank names.

3. **Factual accuracy when misrouted**: D handled ncf04/ncf05 misroutes better than G
   (hedged or gave correct answer from training data vs corrupted output).

4. **Latency**: D is ~0.3s faster on average (no LoRA adapter overhead).

5. **General QA**: Mixed — each arm wins on different topics. No clear overall winner;
   quality depends on topic familiarity in training data.

---

## Awaiting: Arm A blind eval

Run when Colab is switched to `main` branch:
```bash
python eval-monitoring/llm_eval_blind.py \
  --backend https://hypsometrical-unsnobbishly-queen.ngrok-free.dev \
  --out_dir eval-monitoring/results/arm_a_blind
```
