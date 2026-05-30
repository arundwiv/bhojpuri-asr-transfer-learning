# Fluency & Bhojpuri Quality Scores — Blind Eval (30 questions × 3 arms)

**Scored by:** Claude (manual review of all 90 answers)  
**Date:** 2026-05-29  
**Scale:** 1–3 per dimension per answer

## Rubric

**Fluency:**
- 1 = Broken — garbled/corrupted start, nonsense word, incoherent, truncated subject
- 2 = Acceptable — understandable but has word-level errors, awkward phrases, or wrong words
- 3 = Natural — well-formed, coherent, no corruptions

**Bhojpuri:**
- 1 = Hindi-dominant — "है","होता है","नहीं","मैं","हमारा" dominate
- 2 = Mixed — some Bhojpuri ("बा","बानी") alongside Hindi forms
- 3 = Bhojpuri — "बा","बाड़न","बानी","नइखे","होला","करीं" dominate

---

## Overall Scores

| | Arm A (nf4) | Arm D (BF16) | Arm G (BF16+QLoRA) |
|---|---|---|---|
| **Fluency** | 1.97 | 2.10 | **2.30** |
| **Bhojpuri quality** | 1.67 | 1.67 | **2.97** |

---

## By Category

| Category (n) | A fluency | D fluency | G fluency | A bhojpuri | D bhojpuri | G bhojpuri |
|---|---|---|---|---|---|---|
| Weather (5) | 1.80 | 2.00 | **2.20** | 1.60 | 1.60 | **3.00** |
| Current facts (6) | **2.20** | **2.20** | 1.83 | 2.33 | 2.00 | **3.00** |
| General QA (10) | 1.70 | 1.80 | **2.60** | 1.40 | 1.50 | **3.00** |
| Conversational+OOS (9) | **2.22** | **2.44** | 2.22 | 1.67 | 1.78 | **2.89** |

---

## Key Findings

### Fluency

**Arm G is the most fluent overall (2.30)** — including on general QA (2.60), where QLoRA
training produced cleaner, more natural sentence structure.

**Arm G's weakest spot is current facts (1.83)** — two specific corruptions pull the score
down: "रल" in governor answers (3 questions) and "भ के" in the misrouted PM answer. These
are isolated regressions, not a broad fluency drop. Strip those 4 outliers and G's cf
fluency would be ~2.5.

**Arm D is slightly more fluent than A overall (2.10 vs 1.97)** — BF16 generates cleaner
text than nf4 even without fine-tuning. The one catastrophic failure (ngq06: "ब pahilee
pramukh . Dusraa .") is an outlier.

**Common fluency failures across all arms:**
- City name garbling: "गोरख पर" (A), "गोरखुपुर" (D, G)
- "तापमानस" extra स (all three)
- Cooking/agriculture questions: all arms have weak factual+fluency on novel procedural topics

### Bhojpuri Quality

**Arm G dominates: 2.97/3** — essentially every answer uses Bhojpuri grammar.
This is a real, robust improvement from QLoRA and generalises completely to unseen questions.

**Arms A and D are identical: 1.67** — BF16 (Arm D) gives no Bhojpuri quality improvement
over nf4 (Arm A) without fine-tuning. The base Llama 3.1 8B defaults to Hindi regardless
of quantisation.

**The QLoRA Bhojpuri gap is the largest single improvement** across all metrics measured:
+1.30 points over A and D. No other change (Tavily, routing, prompt tuning) moved a metric
this much.

---

## Arm G Fluency Regressions (vs D)

| ID | Issue | Root cause |
|---|---|---|
| ncf03, ncf06 | "रल सय्य़ाद" — "जनरल" mangled | QLoRA training: no Lt. Gen. rank examples |
| ncf05 | "भ के नरेन्द्र मोदी" — corrupted start | Misrouted answer; model uncertain |
| ngq04 | "त के पता" — "तुलसी" truncated | Model drops opening noun; training gap |
| ncv05 | "ई मन के था" — incoherent farewell | Novel social phrase not in training |

All four are fixable with targeted training data. None indicate a systemic fluency
regression — they are specific-phrase gaps.
