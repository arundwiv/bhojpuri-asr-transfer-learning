"""
Fluency scoring for blind eval answers using Claude as judge.

Scores each answer 1-3 on two axes:
  fluency   : Is the text well-formed, no corruptions, natural flow?
  bhojpuri  : Is it in Bhojpuri (vs Hindi or garbled mix)?

Reads from arm_*_blind JSONL files, writes a combined CSV.

Usage:
    python eval-monitoring/score_fluency.py
    python eval-monitoring/score_fluency.py --arms a d g
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import anthropic

RUBRIC = """You are evaluating answers from a Bhojpuri voice assistant.
Score the answer on TWO dimensions, each 1–3:

FLUENCY (1–3):
  1 = Broken — garbled words, corrupted start, truncated, or completely off-topic
  2 = Acceptable — understandable but awkward, repetitive, or has minor word errors
  3 = Good — natural, coherent, well-formed sentence(s), no corruptions

BHOJPURI (1–3):
  1 = Hindi or mixed — predominantly Hindi grammar/verbs ("है","हैं","होता है","नहीं")
  2 = Mixed — some Bhojpuri forms ("बा","बानी") but also Hindi forms
  3 = Bhojpuri — predominantly Bhojpuri grammar ("बा","बाड़न","बानी","नइखे","होला")

Question: {question}
Answer: {answer}

Reply with exactly two lines:
fluency: <1|2|3>
bhojpuri: <1|2|3>
reason: <one short phrase>"""


def score_answer(client: anthropic.Anthropic, question: str, answer: str) -> dict:
    prompt = RUBRIC.format(question=question, answer=answer)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    result = {"fluency": None, "bhojpuri": None, "reason": "", "raw": raw}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("fluency:"):
            try:
                result["fluency"] = int(line.split(":")[1].strip()[0])
            except Exception:
                pass
        elif line.startswith("bhojpuri:"):
            try:
                result["bhojpuri"] = int(line.split(":")[1].strip()[0])
            except Exception:
                pass
        elif line.startswith("reason:"):
            result["reason"] = line.split(":", 1)[1].strip()
    return result


def load_arm(arm: str) -> dict:
    path = Path(f"eval-monitoring/results/arm_{arm}_blind/llm_eval_blind_results.jsonl")
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["a", "d", "g"])
    parser.add_argument("--out", default="eval-monitoring/results/fluency_scores.csv")
    parser.add_argument("--pause", type=float, default=0.3)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    arms_data = {arm: load_arm(arm) for arm in args.arms}
    all_ids = sorted(next(iter(arms_data.values())).keys())

    fields = ["id", "category", "question"] + [
        f"{arm}_{metric}"
        for arm in args.arms
        for metric in ["answer", "fluency", "bhojpuri", "reason"]
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    total = len(all_ids) * len(args.arms)
    done = 0

    for pid in all_ids:
        base = next(iter(arms_data.values()))[pid]
        row = {
            "id": pid,
            "category": base.get("category_actual", ""),
            "question": base.get("text", ""),
        }
        for arm in args.arms:
            r = arms_data[arm].get(pid, {})
            answer = r.get("answer", "")
            done += 1
            print(f"[{done}/{total}] {pid} arm={arm} … ", end="", flush=True)
            try:
                scores = score_answer(client, base.get("text", ""), answer)
                print(f"fluency={scores['fluency']} bhojpuri={scores['bhojpuri']} ({scores['reason']})")
            except Exception as e:
                print(f"ERROR: {e}")
                scores = {"fluency": None, "bhojpuri": None, "reason": str(e)}
            row[f"{arm}_answer"]   = answer
            row[f"{arm}_fluency"]  = scores["fluency"]
            row[f"{arm}_bhojpuri"] = scores["bhojpuri"]
            row[f"{arm}_reason"]   = scores["reason"]
            time.sleep(args.pause)
        rows_out.append(row)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nScores written to {out_path}")

    # Summary
    print("\n=== Average scores per arm ===")
    for arm in args.arms:
        f_scores = [r[f"{arm}_fluency"] for r in rows_out if r.get(f"{arm}_fluency")]
        b_scores = [r[f"{arm}_bhojpuri"] for r in rows_out if r.get(f"{arm}_bhojpuri")]
        avg_f = round(sum(f_scores) / len(f_scores), 2) if f_scores else None
        avg_b = round(sum(b_scores) / len(b_scores), 2) if b_scores else None
        print(f"  Arm {arm.upper()}: fluency={avg_f}/3  bhojpuri={avg_b}/3  (n={len(f_scores)})")

    print("\n=== By category ===")
    cats = sorted(set(r["category"] for r in rows_out))
    for cat in cats:
        cat_rows = [r for r in rows_out if r["category"] == cat]
        parts = []
        for arm in args.arms:
            f_scores = [r[f"{arm}_fluency"] for r in cat_rows if r.get(f"{arm}_fluency")]
            avg_f = round(sum(f_scores)/len(f_scores), 2) if f_scores else "-"
            parts.append(f"A{arm.upper()}={avg_f}")
        print(f"  {cat:<16} " + "  ".join(parts))


if __name__ == "__main__":
    main()
