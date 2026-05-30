# Blind Eval — Three-Arm Comparison (A vs D vs G)

**Script:** `eval-monitoring/llm_eval_blind.py` (30 fresh questions)  
**Date:** 2026-05-29  
**Purpose:** Validate quality improvements on unseen questions; prevent overfitting
to the original 50-prompt development set.

---

## Aggregate Results

| Metric | Arm A (main, nf4) | Arm D (BF16) | Arm G (BF16+QLoRA) |
|---|---|---|---|
| Avg LLM latency | 3.73s | **2.90s** | 3.20s |
| Routing correct | 27/30 | 27/30 | 27/30 |
| बाड़न verb form (cf, routed) | 0/4 | 0/4 | **4/4** |
| हैं (Hindi leakage) | 0 | 0 | 0 |
| Arabic artefact | 0 | 0 | 0 |
| Bhojpuri-dominant answers | ~50% | ~65% | **~80%** |

---

## Latency

BF16 (Arm D) is the fastest — no quantisation overhead, no adapter.
Arm G (BF16+LoRA) adds ~0.3s over D — acceptable overhead for the quality gain.
Arm A (bnb-nf4) is slowest despite quantisation; nf4 dequantisation cost exceeds BF16.

---

## Language Quality

**Arm A** has the most Hindi drift: "है", "नहीं हो रही", "होता है", "रहता है" appear
frequently even in weather answers. Script is Devanagari but grammar is largely Hindi.

**Arm D** is a mixed register: uses "बा", "बाटे" sometimes, but falls back to Hindi
"है/होता है" in longer answers. Current-facts uses Bhojpuri-ish "हवें/हें" honorifics.

**Arm G** is the most consistent Bhojpuri: "बा", "नइखे", "होला", "बाड़न", "बानी"
dominate. QLoRA training effect is clear and generalises to unseen questions.

---

## Current Facts

All correctly-routed answers use Tavily-grounded facts — factual accuracy is equal
across arms for these. Differences are in language form only.

| ID | A | D | G |
|---|---|---|---|
| ncf01 (India PM) | "मोदि हें" | "मोदी हवें" | "मोदी **बाड़न**" |
| ncf02 (Bihar CM) | "चौंधारी…" full name | "चौंधारी…" full name | "सम्राट बाड़न" — drops surname ⚠️ |
| ncf03 (Governor) | **best**: full name + date | "जनरल" intact | "**रल**" corruption ⚠️ |
| ncf06 (Governor) | same as ncf03 | same | same corruption |

**Misrouted (bypassed Tavily):**
- ncf04 (Bihar CM): A and G both confidently said "नीतीश कुमार" (wrong ❌); D hedged correctly ✅
- ncf05 (India PM): A and D gave clean correct answers; G was corrupted ("भ के…") ❌

---

## General QA — Notable Differences

| ID | Topic | Best | Worst | Notes |
|---|---|---|---|---|
| ngq01 | Maize season | **G** (June-July ✅) | A (Oct-Nov ❌) | G only correct arm |
| ngq03 | Chhath Puja | **G** (sun/moon worship ✅) | A & D (wrong) | G only correct arm |
| ngq04 | Tulsi benefits | **A & D** | G ("त के पता" — corrupted start) | G regression |
| ngq06 | Bihar rivers | **G** (Ganga/Gandak/Kosi) | D (garbled) | A also truncated |
| ngq07 | Mustard farming | **A & D** (reasonable) | G (wrong technique) | G regression |
| ngq08 | Arhar dal recipe | **A & D** (reasonable recipes) | G (wrong) | G regression |
| ngq09 | Electricity saving | **D** (practical bulb advice) | G ("call helpline") | D wins |

---

## Out-of-Scope

- **nos03** ("हमार किस्मत कइसन बा?"): A replied "किस्मत अच्छी बा" — incorrectly
  affirmed the user has good luck. D and G correctly deflected. ❌ A hallucination.
- All three arms appropriately declined real-time queries (train, IPL, stock market).

---

## Verdict

**Arm G is the best overall configuration:**
- Bhojpuri language quality clearly improved and generalises to fresh questions
- Latency better than Arm A (3.2s vs 3.73s), slightly slower than D (+0.3s)
- Zero Arabic artefacts

**Arm G regressions vs D (QLoRA side-effects):**
1. Proper noun corruption: "जनरल"→"रल" in governor answers
2. Bihar CM surname dropped ("सम्राट" not "सम्राट सिंह चौधरी")
3. Some general QA start corruptions (ngq04 "त के पता")
4. Misrouted current-facts answers worse than A/D (more confidently wrong)

These regressions point to the same root cause: the QLoRA training corpus underrepresents
Lt. Gen. rank names and full Bihar CM name. Next training iteration should add 5-10 examples
specifically covering these.

**Arm A is not competitive:** slowest, most Hindi drift, hallucinates on out-of-scope.

---

## Open issues (all arms)

- Router misses on "मुखिया" and "PM" (fixed in arm-g-qlora, not yet in main/arm-d)
- "गर्मी में लू" misfires as weather (ambiguous — left open)
- "जनरल"→"रल" in Arm G governor answers (training data gap)
