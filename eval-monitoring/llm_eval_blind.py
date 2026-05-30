"""
Blind evaluation harness — 30 questions never seen during development.

Purpose: validate that quality improvements on llm_eval.py generalise and
are not artefacts of overfitting to the original 50-prompt set.

Categories mirror llm_eval.py but use fresh phrasings, cities, and topics:
  nw01–nw05  : weather        (5)  — different cities / question styles
  ncf01–ncf06: current_facts  (6)  — same factual targets, different wordings
  ngq01–ngq10: general_qa    (10)  — topics not in the original set
  ncv01–ncv05: conversational  (5)  — different social phrases
  nos01–nos04: out_of_scope    (4)  — different edge cases

Usage:
    python eval-monitoring/llm_eval_blind.py
    python eval-monitoring/llm_eval_blind.py --backend https://xxx.ngrok-free.app
    python eval-monitoring/llm_eval_blind.py --out_dir eval-monitoring/results/arm_a_blind
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

TIMEOUT = 120

# ---------------------------------------------------------------------------
# Blind prompt set — 30 questions, none from llm_eval.py
# ---------------------------------------------------------------------------
PROMPTS = [
    # --- Weather (5) — cities and angles not in original set ---
    {"id": "nw01", "category_expected": "weather",
     "text": "बलिया में आज मौसम कइसन बा?"},
    {"id": "nw02", "category_expected": "weather",
     "text": "गोरखपुर में अबहीं बारिश हो रहल बा का?"},
    {"id": "nw03", "category_expected": "weather",
     "text": "सीवान में आजु तापमान केतना बा?"},
    {"id": "nw04", "category_expected": "weather",
     "text": "आज बनारस में आंधी आई का?"},
    {"id": "nw05", "category_expected": "weather",
     "text": "बक्सर में ठंड बा कि गर्मी आजु?"},

    # --- Current facts (6) — same targets (Bihar CM / India PM / Bihar Governor)
    #     but different phrasings. Tests whether improvements generalise across
    #     question surface forms.
    {"id": "ncf01", "category_expected": "current_facts",
     "text": "देश के प्रधानमंत्री के नाम बताईं।"},
    {"id": "ncf02", "category_expected": "current_facts",
     "text": "बिहार के सीएम के नाम का बा?"},
    {"id": "ncf03", "category_expected": "current_facts",
     "text": "बिहार राज्य के राज्यपाल कवन बाड़न?"},
    {"id": "ncf04", "category_expected": "current_facts",
     "text": "आजकल बिहार के मुखिया कवन बाड़न?"},
    {"id": "ncf05", "category_expected": "current_facts",
     "text": "भारत के PM के का नाम बा?"},
    {"id": "ncf06", "category_expected": "current_facts",
     "text": "बिहार के गवर्नर साहब के नाम का बा?"},

    # --- General QA (10) — topics not in the original set ---
    {"id": "ngq01", "category_expected": "general_qa",
     "text": "मक्का के खेती कब करीं?"},
    {"id": "ngq02", "category_expected": "general_qa",
     "text": "गाय के दूध के का फायदा बा?"},
    {"id": "ngq03", "category_expected": "general_qa",
     "text": "छठ पूजा में का होला?"},
    {"id": "ngq04", "category_expected": "general_qa",
     "text": "तुलसी के पत्ता के का फायदा बा?"},
    {"id": "ngq05", "category_expected": "general_qa",
     "text": "रोज सुबह योग करे के का फायदा बा?"},
    {"id": "ngq06", "category_expected": "general_qa",
     "text": "बिहार में कवन-कवन प्रमुख नदी बा?"},
    {"id": "ngq07", "category_expected": "general_qa",
     "text": "सरसों के खेती कइसे होला?"},
    {"id": "ngq08", "category_expected": "general_qa",
     "text": "अरहर के दाल कइसे बनाईं?"},
    {"id": "ngq09", "category_expected": "general_qa",
     "text": "बिजली के बिल कम करे के का उपाय बा?"},
    {"id": "ngq10", "category_expected": "general_qa",
     "text": "गर्मी में लू से बचे के का करीं?"},

    # --- Conversational (5) — greetings and social phrases not in original set ---
    {"id": "ncv01", "category_expected": "general_qa",
     "text": "का समाचार बा?"},
    {"id": "ncv02", "category_expected": "general_qa",
     "text": "रउरा कवन भाषा में बात कर सकत बानी?"},
    {"id": "ncv03", "category_expected": "general_qa",
     "text": "जय हो!"},
    {"id": "ncv04", "category_expected": "general_qa",
     "text": "बहुत नीमन बताईं!"},
    {"id": "ncv05", "category_expected": "general_qa",
     "text": "फिर भेंट होई।"},

    # --- Out-of-scope (4) — different edge cases than the original set ---
    {"id": "nos01", "category_expected": "general_qa",
     "text": "कल ट्रेन लेट होई का?"},
    {"id": "nos02", "category_expected": "general_qa",
     "text": "कवन शेयर खरीदीं आजु?"},
    {"id": "nos03", "category_expected": "general_qa",
     "text": "हमार किस्मत कइसन बा?"},
    {"id": "nos04", "category_expected": "general_qa",
     "text": "कल IPL में कवन टीम जीती?"},
]

CSV_FIELDS = [
    "id", "category_expected", "category_actual", "routed_correctly",
    "prompt_hash", "text", "answer",
    "llm_seconds", "total_seconds", "model",
    "timestamp", "quality_score", "notes",
]


def prompt_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def call_eval_text(backend: str, text: str, timeout: int, token: str = "") -> dict:
    url = backend.rstrip("/") + "/eval_text"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.post(url, json={"text": text}, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Blind eval harness — 30 fresh prompts")
    parser.add_argument("--backend",  default="http://localhost:8000")
    parser.add_argument("--out_dir",  default="eval-monitoring/results/arm_g_blind")
    parser.add_argument("--pause",    type=float, default=0.5)
    parser.add_argument("--timeout",  type=int,   default=TIMEOUT)
    parser.add_argument("--token",    default=os.environ.get("BACKEND_AUTH_TOKEN", ""))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path   = out_dir / "llm_eval_blind_results.csv"
    jsonl_path = out_dir / "llm_eval_blind_results.jsonl"

    try:
        r = requests.get(args.backend.rstrip("/") + "/", timeout=10)
        r.raise_for_status()
        print(f"Backend OK  ({args.backend})")
    except Exception as e:
        print(f"Backend health check failed: {e}")
        sys.exit(1)

    print(f"Prompts : {len(PROMPTS)}  (blind set — fresh phrasings/topics)")
    print(f"Out dir : {out_dir}\n")

    rows = []
    n_ok = n_fail = 0
    routing_correct = routing_total = 0

    with open(jsonl_path, "w", encoding="utf-8") as jf, \
         open(csv_path,   "w", newline="", encoding="utf-8-sig") as cf:

        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for i, prompt in enumerate(PROMPTS, 1):
            pid   = prompt["id"]
            text  = prompt["text"]
            cat_x = prompt["category_expected"]
            phash = prompt_hash(text)

            print(f"[{i:02d}/{len(PROMPTS)}] {pid}  {text[:45]}...", end="", flush=True)
            t0 = time.time()

            try:
                data = call_eval_text(args.backend, text, args.timeout, args.token)
                elapsed = round(time.time() - t0, 2)

                cat_actual = data.get("category", "")
                routed_ok  = cat_actual == cat_x
                routing_total += 1
                if routed_ok:
                    routing_correct += 1

                row = {
                    "id":                pid,
                    "category_expected": cat_x,
                    "category_actual":   cat_actual,
                    "routed_correctly":  routed_ok,
                    "prompt_hash":       phash,
                    "text":              text,
                    "answer":            data.get("answer", ""),
                    "llm_seconds":       data.get("llm_seconds", ""),
                    "total_seconds":     data.get("total_seconds", elapsed),
                    "model":             data.get("model", ""),
                    "timestamp":         now_utc(),
                    "quality_score":     "",
                    "notes":             "",
                }
                jf.write(json.dumps({**row, "_full": data}, ensure_ascii=False) + "\n")
                jf.flush()
                writer.writerow(row)
                cf.flush()
                rows.append(row)
                n_ok += 1

                route_flag = "" if routed_ok else f" [ROUTE? expected={cat_x} got={cat_actual}]"
                print(f"  OK  {data.get('total_seconds', elapsed):.1f}s{route_flag}")

            except Exception as e:
                elapsed = round(time.time() - t0, 2)
                row = {f: "" for f in CSV_FIELDS}
                row.update({"id": pid, "text": text, "category_expected": cat_x,
                            "prompt_hash": phash, "timestamp": now_utc(),
                            "notes": str(e)})
                jf.write(json.dumps({"_file": pid, "ok": False, "error": str(e)},
                                    ensure_ascii=False) + "\n")
                jf.flush()
                writer.writerow(row)
                cf.flush()
                rows.append(row)
                n_fail += 1
                print(f"  FAIL ({elapsed}s): {e}")

            if i < len(PROMPTS):
                time.sleep(args.pause)

    def avg(field):
        vals = [float(r[field]) for r in rows if r.get(field) not in ("", None)]
        return round(sum(vals) / len(vals), 2) if vals else None

    route_pct = round(100 * routing_correct / routing_total, 1) if routing_total else 0

    print(f"\n{'='*55}")
    print(f"Results   : {n_ok} OK, {n_fail} failed")
    print(f"Routing   : {routing_correct}/{routing_total} correct ({route_pct}%)")
    print(f"Avg LLM   : {avg('llm_seconds')}s")
    print(f"Avg total : {avg('total_seconds')}s")
    print(f"\nCSV  -> {csv_path}")
    print(f"JSONL-> {jsonl_path}")
    print(f"\nFill in quality_score (1=wrong, 2=ok, 3=good) for manual review.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
