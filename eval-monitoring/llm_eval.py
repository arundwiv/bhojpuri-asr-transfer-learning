"""
LLM answer quality evaluation harness.

Sends a fixed set of Bhojpuri/Hindi text prompts to /eval_text and records
the router decision, LLM answer, and latency. Produces a CSV with a blank
quality_score column for manual human review.

Usage:
    python eval-monitoring/llm_eval.py
    python eval-monitoring/llm_eval.py --backend https://xxx.ngrok-free.app
    python eval-monitoring/llm_eval.py --backend http://localhost:8000 --out_dir eval-monitoring/results
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
# Fixed prompt set — 50 prompts across 5 categories
# ---------------------------------------------------------------------------
PROMPTS = [
    # --- Weather (8) — should route to "weather", trigger tool ---
    {"id": "w01", "category_expected": "weather",       "text": "आज पटना में मौसम कैसा बा?"},
    {"id": "w02", "category_expected": "weather",       "text": "कल बारिश होई?"},
    {"id": "w03", "category_expected": "weather",       "text": "आरा में आज तापमान केतना बा?"},
    {"id": "w04", "category_expected": "weather",       "text": "वाराणसी में धूप बा कि बादल?"},
    {"id": "w05", "category_expected": "weather",       "text": "आज ठंड बा कि गर्मी?"},
    {"id": "w06", "category_expected": "weather",       "text": "मुजफ्फरपुर में मौसम कइसन बा?"},
    {"id": "w07", "category_expected": "weather",       "text": "आज हवा तेज बा?"},
    {"id": "w08", "category_expected": "weather",       "text": "छपरा में बरखा होई आज?"},

    # --- Current facts (10) — should route to "current_facts", trigger tool ---
    {"id": "cf01", "category_expected": "current_facts", "text": "बिहार के मुख्यमंत्री कवन बा?"},
    {"id": "cf02", "category_expected": "current_facts", "text": "भारत के प्रधानमंत्री कौन हैं?"},
    {"id": "cf03", "category_expected": "current_facts", "text": "बिहार के राज्यपाल कौन हैं?"},
    {"id": "cf04", "category_expected": "current_facts", "text": "अभी बिहार के सीएम कौन बा?"},
    {"id": "cf05", "category_expected": "current_facts", "text": "भारत के पीएम कवन बा?"},
    {"id": "cf06", "category_expected": "current_facts", "text": "अभी के मुख्यमंत्री बिहार के कवन बाड़न?"},
    {"id": "cf07", "category_expected": "current_facts", "text": "बिहार के गवर्नर कवन बाड़न?"},
    {"id": "cf08", "category_expected": "current_facts", "text": "आज भारत के प्रधानमंत्री कौन बाड़न?"},
    {"id": "cf09", "category_expected": "current_facts", "text": "अभी कौन सीएम बाड़न बिहार में?"},
    {"id": "cf10", "category_expected": "current_facts", "text": "बिहार के राज्यपाल के नाम बताईं"},

    # --- General QA / Factual (15) — should route to "general_qa" ---
    {"id": "gq01", "category_expected": "general_qa",   "text": "भोजपुरी भाषा कहाँ बोली जाती है?"},
    {"id": "gq02", "category_expected": "general_qa",   "text": "गंगा नदी कहाँ से निकलती है?"},
    {"id": "gq03", "category_expected": "general_qa",   "text": "आम के फायदे का हवे?"},
    {"id": "gq04", "category_expected": "general_qa",   "text": "धान के खेती कब करीं?"},
    {"id": "gq05", "category_expected": "general_qa",   "text": "दूध पीयल के का फायदा बा?"},
    {"id": "gq06", "category_expected": "general_qa",   "text": "बिहार के राजधानी का बा?"},
    {"id": "gq07", "category_expected": "general_qa",   "text": "भारत के राष्ट्रीय पक्षी कवन बा?"},
    {"id": "gq08", "category_expected": "general_qa",   "text": "चाय कइसे बनाईं?"},
    {"id": "gq09", "category_expected": "general_qa",   "text": "बच्चा के बुखार में का करीं?"},
    {"id": "gq10", "category_expected": "general_qa",   "text": "गेहूँ के बुआई कब होला?"},
    {"id": "gq11", "category_expected": "general_qa",   "text": "पानी कतना पीयल चाही रोज?"},
    {"id": "gq12", "category_expected": "general_qa",   "text": "सूरज कहाँ से उगेला?"},
    {"id": "gq13", "category_expected": "general_qa",   "text": "मोबाइल बैटरी जल्दी खतम काहे होला?"},
    {"id": "gq14", "category_expected": "general_qa",   "text": "यूरिया खाद कब डालीं?"},
    {"id": "gq15", "category_expected": "general_qa",   "text": "आलू के खेती कइसे होला?"},

    # --- Conversational (10) — should route to "general_qa", short friendly answer ---
    {"id": "cv01", "category_expected": "general_qa",   "text": "नमस्ते, तू कइसन बाड़s?"},
    {"id": "cv02", "category_expected": "general_qa",   "text": "तोहार नाम का बा?"},
    {"id": "cv03", "category_expected": "general_qa",   "text": "का हाल बा?"},
    {"id": "cv04", "category_expected": "general_qa",   "text": "तू का कर सकेलs?"},
    {"id": "cv05", "category_expected": "general_qa",   "text": "हमरा मदद करs"},
    {"id": "cv06", "category_expected": "general_qa",   "text": "धन्यवाद"},
    {"id": "cv07", "category_expected": "general_qa",   "text": "शुभ प्रभात"},
    {"id": "cv08", "category_expected": "general_qa",   "text": "शुभ रात्रि"},
    {"id": "cv09", "category_expected": "general_qa",   "text": "बहुत बढ़िया बताईं"},
    {"id": "cv10", "category_expected": "general_qa",   "text": "ठीक बा, फिर मिलब"},

    # --- Out-of-scope / tricky (7) — LLM should handle gracefully, not hallucinate ---
    {"id": "os01", "category_expected": "general_qa",   "text": "कल का शेयर मार्केट कइसन रही?"},
    {"id": "os02", "category_expected": "general_qa",   "text": "लॉटरी में कइसे जीतीं?"},
    {"id": "os03", "category_expected": "general_qa",   "text": "हमरा पैसा कहाँ गया?"},
    {"id": "os04", "category_expected": "general_qa",   "text": "तू भगवान बाड़s का?"},
    {"id": "os05", "category_expected": "general_qa",   "text": "आज IPL में कौन जीता?"},
    {"id": "os06", "category_expected": "general_qa",   "text": "हमरा नाम बता"},
    {"id": "os07", "category_expected": "general_qa",   "text": "बिना दवाई के बुखार कइसे उतारीं?"},
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

    parser = argparse.ArgumentParser(description="LLM eval harness for Bhojpuri AI")
    parser.add_argument("--backend",  default="http://localhost:8000")
    parser.add_argument("--out_dir",  default="eval-monitoring/results")
    parser.add_argument("--pause",    type=float, default=0.5,
                        help="Seconds between requests (GPU cooldown)")
    parser.add_argument("--timeout",  type=int,   default=TIMEOUT)
    parser.add_argument("--token",    default=os.environ.get("BACKEND_AUTH_TOKEN", ""),
                        help="Bearer token (or set BACKEND_AUTH_TOKEN env var)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / "llm_eval_results.csv"
    jsonl_path = out_dir / "llm_eval_results.jsonl"

    # Health check
    try:
        r = requests.get(args.backend.rstrip("/") + "/", timeout=10)
        r.raise_for_status()
        print(f"Backend OK  ({args.backend})")
    except Exception as e:
        print(f"Backend health check failed: {e}")
        sys.exit(1)

    print(f"Prompts : {len(PROMPTS)}")
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

            print(f"[{i:02d}/{len(PROMPTS)}] {pid}  {text[:40]}...", end="", flush=True)
            t0 = time.time()

            try:
                data = call_eval_text(args.backend, text, args.timeout, args.token)
                elapsed = round(time.time() - t0, 2)

                cat_actual  = data.get("category", "")
                routed_ok   = cat_actual == cat_x
                routing_total += 1
                if routed_ok:
                    routing_correct += 1

                row = {
                    "id":               pid,
                    "category_expected": cat_x,
                    "category_actual":   cat_actual,
                    "routed_correctly":  routed_ok,
                    "prompt_hash":       phash,
                    "text":             text,
                    "answer":           data.get("answer", ""),
                    "llm_seconds":      data.get("llm_seconds", ""),
                    "total_seconds":    data.get("total_seconds", elapsed),
                    "model":            data.get("model", ""),
                    "timestamp":        now_utc(),
                    "quality_score":    "",   # fill in manually
                    "notes":            "",
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

    # Summary
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
    print(f"\nOpen the CSV, fill in quality_score (1=wrong, 2=ok, 3=good)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
