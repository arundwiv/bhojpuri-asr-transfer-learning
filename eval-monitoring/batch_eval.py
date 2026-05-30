"""
Batch evaluation script for the Bhojpuri AI platform.

Sends each WAV file in a directory to the /chat_debug endpoint and records:
  - ASR transcript
  - LLM answer
  - Per-stage latencies (asr, classifier, tool, llm, tts, total)
  - Category / routing decision
  - Any errors

Results are written incrementally to:
  - results.jsonl  (one JSON object per line, full detail)
  - results.csv    (summary table, easy to open in Excel/Sheets)
  - tts_output/    (decoded WAV files from TTS, optional)

Usage:
    python batch_eval.py --wav_dir /path/to/wavs --backend http://localhost:8000
    python batch_eval.py --wav_dir /path/to/wavs --backend http://<colab-ngrok-url> --save_audio
"""

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

TIMEOUT_SECONDS = 180  # LLM can be slow on first call; generous default

CSV_FIELDS = [
    "file",
    "ok",
    "request_id",
    "transcript",
    "category",
    "needs_tool",
    "answer",
    "asr_seconds",
    "classifier_seconds",
    "tool_seconds",
    "llm_seconds",
    "tts_seconds",
    "audio_duration_seconds",
    "total_seconds",
    "error",
]


def call_chat_debug(backend_url: str, wav_path: Path, timeout: int, token: str = "") -> dict:
    url = backend_url.rstrip("/") + "/chat_debug"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with open(wav_path, "rb") as f:
        files = {"audio": (wav_path.name, f, "audio/wav")}
        resp = requests.post(url, files=files, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def flatten_result(wav_name: str, data: dict) -> dict:
    timings = data.get("timings", {})
    return {
        "file": wav_name,
        "ok": data.get("ok", False),
        "request_id": data.get("request_id", ""),
        "transcript": data.get("transcript", ""),
        "category": data.get("category", ""),
        "needs_tool": data.get("needs_tool", False),
        "answer": data.get("answer", ""),
        "asr_seconds": timings.get("asr_seconds", ""),
        "classifier_seconds": timings.get("classifier_seconds", ""),
        "tool_seconds": timings.get("tool_seconds", ""),
        "llm_seconds": timings.get("llm_seconds", ""),
        "tts_seconds": timings.get("tts_seconds", ""),
        "audio_duration_seconds": timings.get("audio_duration_seconds", ""),
        "total_seconds": timings.get("total_seconds", ""),
        "error": data.get("error", ""),
    }


def save_audio(wav_path: Path, audio_b64: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("tts_" + wav_path.name)
    wav_bytes = base64.b64decode(audio_b64)
    out_path.write_bytes(wav_bytes)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Batch eval for Bhojpuri AI /chat_debug endpoint")
    parser.add_argument("--wav_dir", default="test_wavs",
                        help="Directory containing input WAV files")
    parser.add_argument("--backend", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--out_dir", default="eval-monitoring/results",
                        help="Output directory for results")
    parser.add_argument("--save_audio", action="store_true", help="Decode and save TTS output WAV files")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS, help="Per-request timeout in seconds")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to wait between requests (GPU cooldown)")
    parser.add_argument("--token", default=os.environ.get("BACKEND_AUTH_TOKEN", ""),
                        help="Bearer token (or set BACKEND_AUTH_TOKEN env var)")
    args, _ = parser.parse_known_args()  # ignore Jupyter kernel args passed by Colab

    wav_dir = Path(args.wav_dir)
    if not wav_dir.is_dir():
        print(f"ERROR: --wav_dir '{wav_dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    _seen = set()
    wav_files = []
    for p in sorted(wav_dir.glob("*.wav")) + sorted(wav_dir.glob("*.WAV")):
        key = p.resolve()
        if key not in _seen:
            _seen.add(key)
            wav_files.append(p)
    if not wav_files:
        print(f"No WAV files found in {wav_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    csv_path = out_dir / "results.csv"

    print(f"Backend : {args.backend}")
    print(f"WAV dir : {wav_dir}  ({len(wav_files)} files)")
    print(f"Out dir : {out_dir}")
    print()

    # Verify backend is reachable before starting
    try:
        health = requests.get(args.backend.rstrip("/") + "/", timeout=10)
        health.raise_for_status()
        print("Backend health check OK\n")
    except Exception as e:
        print(f"WARNING: backend health check failed: {e}\nContinuing anyway…\n")

    rows = []
    with open(jsonl_path, "w", encoding="utf-8") as jf, \
         open(csv_path, "w", newline="", encoding="utf-8-sig") as cf:

        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for i, wav_path in enumerate(wav_files, 1):
            print(f"[{i}/{len(wav_files)}] {wav_path.name} … ", end="", flush=True)
            t0 = time.time()
            try:
                data = call_chat_debug(args.backend, wav_path, args.timeout, args.token)
                elapsed = round(time.time() - t0, 2)

                if args.save_audio and data.get("audio_base64"):
                    tts_dir = out_dir / "tts_output"
                    save_audio(wav_path, data["audio_base64"], tts_dir)

                # Strip audio from JSONL to keep file small
                data_for_jsonl = {k: v for k, v in data.items() if k != "audio_base64"}
                data_for_jsonl["_file"] = wav_path.name
                jf.write(json.dumps(data_for_jsonl, ensure_ascii=False) + "\n")
                jf.flush()

                row = flatten_result(wav_path.name, data)
                writer.writerow(row)
                cf.flush()
                rows.append(row)

                status = "OK" if data.get("ok") else f"API_ERROR: {data.get('error', '')}"
                print(f"{status}  total={data.get('timings', {}).get('total_seconds', elapsed)}s")

            except Exception as e:
                elapsed = round(time.time() - t0, 2)
                err_data = {"_file": wav_path.name, "ok": False, "error": str(e)}
                jf.write(json.dumps(err_data, ensure_ascii=False) + "\n")
                jf.flush()
                row = {f: "" for f in CSV_FIELDS}
                row["file"] = wav_path.name
                row["ok"] = False
                row["error"] = str(e)
                writer.writerow(row)
                cf.flush()
                rows.append(row)
                print(f"FAILED ({elapsed}s): {e}")

            if i < len(wav_files):
                time.sleep(args.pause)

    # Summary
    n_ok = sum(1 for r in rows if str(r.get("ok")).lower() in ("true", "1"))
    n_fail = len(rows) - n_ok

    def avg(field):
        vals = [float(r[field]) for r in rows if r.get(field) not in ("", None)]
        return round(sum(vals) / len(vals), 3) if vals else None

    print(f"\n{'='*50}")
    print(f"Results: {n_ok} OK, {n_fail} failed")
    print(f"Average latencies (seconds):")
    for field in ["asr_seconds", "llm_seconds", "tts_seconds", "total_seconds"]:
        print(f"  {field:<25} {avg(field)}")
    print(f"\nFull results saved to:")
    print(f"  {jsonl_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
