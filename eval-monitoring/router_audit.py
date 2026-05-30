"""
Router keyword coverage audit — runs fully locally, no backend needed.

Replicates the deterministic keyword router from app.py and runs every
prompt in the LLM eval set through it, reporting:
  - Routing accuracy vs expected category
  - Which keyword triggered each match
  - Prompts that hit DEFAULT_CATEGORY unexpectedly
  - Ambiguous keywords (broad matches worth human review)

Usage:
    python eval-monitoring/router_audit.py
    python eval-monitoring/router_audit.py --out_dir eval-monitoring/results
"""

import argparse
import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Router — keep in sync with app.py ROUTE_RULES / classify_transcript
# ---------------------------------------------------------------------------
ROUTE_RULES = [
    {
        "category": "weather",
        "needs_tool": True,
        "keywords": ["मौसम", "बारिश", "बरखा", "तापमान", "आंधी",
                     "कोहरा", "धूप", "ठंड", "गर्मी", "बादल", "हवा"],
    },
    {
        "category": "current_facts",
        "needs_tool": True,
        "keywords": [
            "मुख्यमंत्री", "प्रधानमंत्री", "राज्यपाल", "मंत्री",
            "सीएम", "पीएम", "गवर्नर",
            "अभी कौन", "अब कौन", "अभी के", "अब के",
            "आज का",
        ],
    },
]
DEFAULT_CATEGORY = "general_qa"


def route(text: str) -> dict:
    text_lc = (text or "").strip().lower()
    for rule in ROUTE_RULES:
        for kw in rule["keywords"]:
            if kw.lower() in text_lc:
                return {
                    "category": rule["category"],
                    "needs_tool": rule["needs_tool"],
                    "matched_keyword": kw,
                    "reason": f"matched keyword '{kw}'",
                }
    return {
        "category": DEFAULT_CATEGORY,
        "needs_tool": False,
        "matched_keyword": "",
        "reason": "no rule matched",
    }


# ---------------------------------------------------------------------------
# Prompts — imported from llm_eval.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from llm_eval import PROMPTS  # noqa: E402

CSV_FIELDS = [
    "id", "category_expected", "category_actual", "routed_correctly",
    "matched_keyword", "reason", "text",
]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Router keyword coverage audit")
    parser.add_argument("--out_dir", default="eval-monitoring/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "router_coverage.csv"

    rows = []
    mismatches = []
    kw_hit_count: dict[str, int] = {}

    for prompt in PROMPTS:
        result = route(prompt["text"])
        cat_expected = prompt["category_expected"]
        cat_actual   = result["category"]
        correct      = cat_actual == cat_expected
        kw           = result["matched_keyword"]

        if kw:
            kw_hit_count[kw] = kw_hit_count.get(kw, 0) + 1

        row = {
            "id":                prompt["id"],
            "category_expected": cat_expected,
            "category_actual":   cat_actual,
            "routed_correctly":  correct,
            "matched_keyword":   kw,
            "reason":            result["reason"],
            "text":              prompt["text"],
        }
        rows.append(row)
        if not correct:
            mismatches.append(row)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # --- Report ---
    n_total   = len(rows)
    n_correct = sum(1 for r in rows if r["routed_correctly"])
    pct       = round(100 * n_correct / n_total, 1)

    print(f"\n{'='*60}")
    print(f"Router Coverage Audit  —  {n_correct}/{n_total} correct ({pct}%)")
    print(f"{'='*60}")

    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for m in mismatches:
            print(f"  [{m['id']}] expected={m['category_expected']} "
                  f"got={m['category_actual']}  kw='{m['matched_keyword']}'")
            print(f"       \"{m['text']}\"")
    else:
        print("\nNo mismatches — all prompts routed correctly.")

    # Keywords that fired more than once (potential over-broad matches)
    broad = {kw: n for kw, n in kw_hit_count.items() if n > 1}
    if broad:
        print(f"\nKeywords that matched multiple prompts (review for over-breadth):")
        for kw, n in sorted(broad.items(), key=lambda x: -x[1]):
            print(f"  '{kw}'  matched {n} prompts")

    # Keywords that never fired (dead keywords in eval set)
    all_kws = [kw for rule in ROUTE_RULES for kw in rule["keywords"]]
    unused = [kw for kw in all_kws if kw not in kw_hit_count]
    if unused:
        print(f"\nKeywords with zero hits in eval set (may still be valid for real speech):")
        for kw in unused:
            print(f"  '{kw}'")

    print(f"\nFull report: {csv_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
