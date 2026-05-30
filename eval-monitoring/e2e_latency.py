"""
End-to-end latency evaluation for the Bhojpuri AI pipeline.

Two modes
─────────
Default (--no-streaming):
  Posts WAV to /chat_debug (non-streaming). Waits for full LLM+TTS before
  returning. Gives total pipeline latency per stage. Good for arm comparisons.

Streaming (--streaming):
  Posts WAV to /chat_sse_chunks, parses the SSE event stream in real time,
  and timestamps each event as it arrives. Captures:
    TTFT — time to first text chunk  (LLM started producing)
    TTFA — time to first audio chunk (user hears something)
  These are the actual voice-UX latency numbers, not total response time.

In both modes WAVs are synthesised once with edge-tts + miniaudio and cached.

Dependencies:
    pip install edge-tts miniaudio requests numpy

Usage:
    # Non-streaming baseline
    python eval-monitoring/e2e_latency.py \\
        --backend https://xxx.ngrok-free.app --arm g --no-synth

    # Streaming (TTFA) measurement
    python eval-monitoring/e2e_latency.py \\
        --backend https://xxx.ngrok-free.app --arm g --no-synth --streaming

    # Synthesise WAV cache only
    python eval-monitoring/e2e_latency.py --synth-only
"""

import argparse
import asyncio
import csv
import io
import json
import os
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

try:
    import edge_tts
    import miniaudio
    CAN_SYNTH = True
except ImportError:
    CAN_SYNTH = False

# ---------------------------------------------------------------------------
# Prompts — same 30 as llm_eval_blind.py
# ---------------------------------------------------------------------------
PROMPTS = [
    {"id": "nw01",  "category": "weather",       "text": "बलिया में आज मौसम कइसन बा?"},
    {"id": "nw02",  "category": "weather",       "text": "गोरखपुर में अबहीं बारिश हो रहल बा का?"},
    {"id": "nw03",  "category": "weather",       "text": "सीवान में आजु तापमान केतना बा?"},
    {"id": "nw04",  "category": "weather",       "text": "आज बनारस में आंधी आई का?"},
    {"id": "nw05",  "category": "weather",       "text": "बक्सर में ठंड बा कि गर्मी आजु?"},

    {"id": "ncf01", "category": "current_facts", "text": "देश के प्रधानमंत्री के नाम बताईं।"},
    {"id": "ncf02", "category": "current_facts", "text": "बिहार के सीएम के नाम का बा?"},
    {"id": "ncf03", "category": "current_facts", "text": "बिहार राज्य के राज्यपाल कवन बाड़न?"},
    {"id": "ncf04", "category": "current_facts", "text": "आजकल बिहार के मुखिया कवन बाड़न?"},
    {"id": "ncf05", "category": "current_facts", "text": "भारत के PM के का नाम बा?"},
    {"id": "ncf06", "category": "current_facts", "text": "बिहार के गवर्नर साहब के नाम का बा?"},

    {"id": "ngq01", "category": "general_qa",    "text": "मक्का के खेती कब करीं?"},
    {"id": "ngq02", "category": "general_qa",    "text": "गाय के दूध के का फायदा बा?"},
    {"id": "ngq03", "category": "general_qa",    "text": "छठ पूजा में का होला?"},
    {"id": "ngq04", "category": "general_qa",    "text": "तुलसी के पत्ता के का फायदा बा?"},
    {"id": "ngq05", "category": "general_qa",    "text": "रोज सुबह योग करे के का फायदा बा?"},
    {"id": "ngq06", "category": "general_qa",    "text": "बिहार में कवन-कवन प्रमुख नदी बा?"},
    {"id": "ngq07", "category": "general_qa",    "text": "सरसों के खेती कइसे होला?"},
    {"id": "ngq08", "category": "general_qa",    "text": "अरहर के दाल कइसे बनाईं?"},
    {"id": "ngq09", "category": "general_qa",    "text": "बिजली के बिल कम करे के का उपाय बा?"},
    {"id": "ngq10", "category": "general_qa",    "text": "गर्मी में लू से बचे के का करीं?"},

    {"id": "ncv01", "category": "general_qa",    "text": "का समाचार बा?"},
    {"id": "ncv02", "category": "general_qa",    "text": "रउरा कवन भाषा में बात कर सकत बानी?"},
    {"id": "ncv03", "category": "general_qa",    "text": "जय हो!"},
    {"id": "ncv04", "category": "general_qa",    "text": "बहुत नीमन बताईं!"},
    {"id": "ncv05", "category": "general_qa",    "text": "फिर भेंट होई।"},

    {"id": "nos01", "category": "general_qa",    "text": "कल ट्रेन लेट होई का?"},
    {"id": "nos02", "category": "general_qa",    "text": "कवन शेयर खरीदीं आजु?"},
    {"id": "nos03", "category": "general_qa",    "text": "हमार किस्मत कइसन बा?"},
    {"id": "nos04", "category": "general_qa",    "text": "कल IPL में कवन टीम जीती?"},
]

VOICE     = "hi-IN-SwaraNeural"
TARGET_SR = 16000


# ---------------------------------------------------------------------------
# WAV synthesis
# ---------------------------------------------------------------------------
async def _synth_mp3(text: str) -> bytes:
    comm = edge_tts.Communicate(text, voice=VOICE)
    buf = io.BytesIO()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def _mp3_to_wav(mp3_bytes: bytes) -> bytes:
    decoded = miniaudio.decode(
        mp3_bytes,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=TARGET_SR,
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(decoded.samples)
    return buf.getvalue()


def synthesize_wav(text: str) -> bytes:
    return _mp3_to_wav(asyncio.run(_synth_mp3(text)))


def ensure_wavs(prompts: list, wav_dir: Path, force: bool = False) -> dict:
    wav_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for p in prompts:
        pid = p["id"]
        wp  = wav_dir / f"{pid}.wav"
        paths[pid] = wp
        if not wp.exists() or force:
            print(f"  Synthesising {pid}: {p['text'][:45]}…", end=" ", flush=True)
            try:
                wp.write_bytes(synthesize_wav(p["text"]))
                print(f"{wp.stat().st_size // 1000}KB")
            except Exception as e:
                print(f"FAILED: {e}")
        else:
            print(f"  {pid}: cached ({wp.stat().st_size // 1000}KB)")
    return paths


# ---------------------------------------------------------------------------
# Non-streaming call  (/chat_debug)
# ---------------------------------------------------------------------------
STAGE_KEYS = ["asr_seconds", "classifier_seconds", "tool_seconds",
              "llm_seconds", "tts_seconds", "total_seconds"]


def call_chat_debug(backend: str, wav_path: Path, token: str, timeout: int,
                    category_expected: str) -> dict:
    url     = backend.rstrip("/") + "/chat_debug"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with open(wav_path, "rb") as f:
        resp = requests.post(url, files={"audio": (wav_path.name, f, "audio/wav")},
                             headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    t    = data.get("timings", {})
    cat  = data.get("category", "")
    return {
        "ok":               data.get("ok", False),
        "asr_transcript":   data.get("transcript", ""),
        "category_actual":  cat,
        "routed_correctly": cat == category_expected,
        "answer":           data.get("answer", ""),
        "error":            data.get("error", ""),
        **{k: t.get(k) for k in STAGE_KEYS},
    }


# ---------------------------------------------------------------------------
# Streaming call  (/chat_sse_chunks)
# ---------------------------------------------------------------------------
STREAM_STAGE_KEYS = ["asr_seconds", "tool_seconds", "ttft_seconds",
                     "ttfa_seconds", "ttfa_client_seconds", "total_seconds"]

# Matches server_t inside any SSE data line without decoding the full JSON.
# Used to extract server-side timestamp from audio_chunk (which has a large base64 field).
_SERVER_T_RE = __import__("re").compile(r'"server_t"\s*:\s*([0-9]+(?:\.[0-9]+)?)')


def call_chat_sse_chunks(backend: str, wav_path: Path, token: str, timeout: int,
                         category_expected: str) -> dict:
    """
    POST to /chat_sse_chunks, parse the SSE event stream, and record latency.

    Two reference frames are maintained to keep measurements comparable:

    SERVER-SIDE (from server_t in event payloads, relative to server req_start):
      asr_seconds   — ASR processing time
      tool_seconds  — tool call time (server-reported in tool_result payload)
      ttft_seconds  — server_t when first text_chunk was emitted
      ttfa_seconds  — server_t when first audio_chunk was emitted  ← primary voice UX metric
      total_seconds — server_t when done event was emitted

    CLIENT-SIDE (wall clock from t_start, includes network/ngrok overhead):
      ttfa_client_seconds — client wall time to receive first audio_chunk data line
                            (inflated by base64 audio transmission; ~1–2s over ngrok)

    The server-side metrics are directly comparable to each other.
    The difference (ttfa_client − ttfa) quantifies audio transmission latency.
    """
    url     = backend.rstrip("/") + "/chat_sse_chunks"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    result = {
        "ok":                  False,
        "asr_transcript":      "",
        "category_actual":     "",
        "routed_correctly":    False,
        "answer":              "",
        "error":               "",
        "asr_seconds":         None,
        "tool_seconds":        None,
        "ttft_seconds":        None,   # server-side: first text chunk emitted
        "ttfa_seconds":        None,   # server-side: first audio chunk emitted
        "ttfa_client_seconds": None,   # client-side: first audio line received
        "total_seconds":       None,
        "n_text_chunks":       0,
        "n_audio_chunks":      0,
    }

    t_start    = time.time()
    event_name = None

    with open(wav_path, "rb") as f:
        resp = requests.post(
            url,
            files={"audio": (wav_path.name, f, "audio/wav")},
            headers=headers,
            stream=True,
            timeout=timeout,
        )
    resp.raise_for_status()

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue

        if raw_line.startswith("event:"):
            event_name = raw_line[6:].strip()
            continue

        if not raw_line.startswith("data:") or not event_name:
            continue

        now = time.time()

        # audio_chunk carries a large base64 payload. Skip full JSON parsing —
        # extract only server_t via regex to get the server-side TTFA without
        # the audio transmission overhead that inflates the client wall clock.
        if event_name == "audio_chunk":
            if result["ttfa_client_seconds"] is None:
                result["ttfa_client_seconds"] = round(now - t_start, 3)
            if result["ttfa_seconds"] is None:
                m = _SERVER_T_RE.search(raw_line)
                if m:
                    result["ttfa_seconds"] = float(m.group(1))
            result["n_audio_chunks"] += 1
            event_name = None
            continue

        try:
            payload = json.loads(raw_line[5:].strip())
        except json.JSONDecodeError:
            event_name = None
            continue

        if event_name == "transcript":
            result["asr_seconds"]    = payload.get("asr_seconds") or round(now - t_start, 3)
            result["asr_transcript"] = payload.get("text", "")

        elif event_name == "route":
            cat = payload.get("category", "")
            result["category_actual"]  = cat
            result["routed_correctly"] = (cat == category_expected)

        elif event_name == "tool_result":
            # Use server-reported tool_seconds from the payload (more accurate
            # than client-side timestamp subtraction across the network).
            result["tool_seconds"] = payload.get("tool_seconds")

        elif event_name == "text_chunk":
            if result["ttft_seconds"] is None:
                result["ttft_seconds"] = round(now - t_start, 3)
            result["n_text_chunks"] += 1

        elif event_name == "answer_final":
            result["answer"] = payload.get("text", "")

        elif event_name == "done":
            result["total_seconds"] = payload.get("total_seconds") or round(now - t_start, 3)
            result["ok"]            = payload.get("ok", True)
            break

        elif event_name == "error":
            result["error"] = payload.get("error", "unknown error")
            result["ok"]    = False
            break

        event_name = None

    # Template responses (conversational) skip the LLM streamer entirely and
    # emit one text_chunk immediately followed by one audio_chunk — TTFA ≈ TTFT.
    # If audio never came (e.g. empty TTS), fall back to TTFT.
    if result["ok"] and result["ttfa_seconds"] is None and result["ttft_seconds"] is not None:
        result["ttfa_seconds"] = result["ttft_seconds"]

    return result


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def pct(vals: list, p: int):
    clean = [v for v in vals if v is not None]
    return round(float(np.percentile(clean, p)), 3) if clean else None


def avg(vals: list):
    clean = [v for v in vals if v is not None]
    return round(sum(clean) / len(clean), 3) if clean else None


def stage_vals(rows: list, stage: str, category: str | None = None) -> list:
    subset = rows if category is None else [r for r in rows if r.get("category") == category]
    return [float(r[stage]) for r in subset if r.get(stage) not in ("", None)]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_non_streaming(rows: list, n_ok: int, n_fail: int, arm: str, ts: str,
                          csv_path: Path, jsonl_path: Path):
    ok_rows = [r for r in rows if str(r.get("ok")).lower() in ("true", "1")]
    cats    = ["weather", "current_facts", "general_qa"]

    print(f"\n{'='*62}")
    print(f"Results: {n_ok} OK, {n_fail} failed  |  arm={arm or '?'}  |  {ts}")
    print(f"{'='*62}")
    print(f"\n  {'Stage':<22} {'p50':>7} {'p95':>7} {'mean':>7}   n")
    print("  " + "-"*48)
    for stage in STAGE_KEYS:
        v = stage_vals(ok_rows, stage)
        if not v:
            continue
        print(f"  {stage:<22} {pct(v,50):>7} {pct(v,95):>7} {avg(v):>7}  ({len(v)})")

    print(f"\n  Category breakdown (mean seconds)")
    print("  " + "-"*60)
    hdr = f"  {'Category':<18}" + "".join(f"{s[:6]:>9}" for s in ["asr","tool","llm","tts","total"])
    print(hdr)
    for cat in cats:
        parts = []
        for s in ["asr_seconds","tool_seconds","llm_seconds","tts_seconds","total_seconds"]:
            v = avg(stage_vals(ok_rows, s, cat))
            parts.append(f"{v:>9}" if v is not None else f"{'—':>9}")
        print(f"  {cat:<18}" + "".join(parts))

    sm = {s: avg(stage_vals(ok_rows, s)) for s in STAGE_KEYS}
    total_m = sm.get("total_seconds") or 1
    print(f"\n  Bottleneck (% of total):")
    for s in ["asr_seconds", "tool_seconds", "llm_seconds", "tts_seconds"]:
        v = sm.get(s)
        if v:
            pct_t = round(100 * v / total_m, 1)
            print(f"  {s:<22} {v:>5.2f}s  {pct_t:>5.1f}%  {'█'*int(pct_t/5)}")

    ok   = sum(1 for r in ok_rows if str(r.get("routed_correctly")).lower() == "true")
    tot  = len(ok_rows)
    miss = [(r["id"], r.get("asr_transcript","")[:50])
            for r in ok_rows if str(r.get("routed_correctly")).lower() != "true"]
    print(f"\n  Routing (post-ASR): {ok}/{tot} ({round(100*ok/tot,1) if tot else 0}%)")
    for pid, tr in miss:
        print(f"    MISS {pid}: \"{tr}\"")

    print(f"\n  Results → {csv_path}")
    print(f"  Detail  → {jsonl_path}")
    print(f"{'='*62}")


def report_streaming(rows: list, n_ok: int, n_fail: int, arm: str, ts: str,
                     csv_path: Path, jsonl_path: Path):
    ok_rows = [r for r in rows if str(r.get("ok")).lower() in ("true", "1")]
    cats    = ["weather", "current_facts", "general_qa"]

    print(f"\n{'='*62}")
    print(f"Results: {n_ok} OK, {n_fail} failed  |  arm={arm or '?'}  |  STREAMING  |  {ts}")
    print(f"{'='*62}")

    print(f"\n  {'Stage':<32} {'p50':>7} {'p95':>7} {'mean':>7}   n")
    print("  " + "-"*58)

    # All times are SERVER-SIDE (server_t relative to server req_start),
    # so they are directly comparable to each other.
    stage_labels = [
        ("asr_seconds",         "ASR"),
        ("tool_seconds",        "Tool (when used)"),
        ("ttft_seconds",        "TTFT  (server: 1st text emitted)"),
        ("ttfa_seconds",        "TTFA  (server: 1st audio emitted) ◀ voice UX"),
        ("total_seconds",       "Total (server: pipeline complete)"),
        ("ttfa_client_seconds", "TTFA  (client: audio received, incl. network)"),
    ]
    for key, label in stage_labels:
        v = stage_vals(ok_rows, key)
        if not v:
            continue
        print(f"  {label:<32} {pct(v,50):>7} {pct(v,95):>7} {avg(v):>7}  ({len(v)})")

    # Streaming gain: server-side TTFA vs server-side total (same reference frame)
    ttfa_srv  = stage_vals(ok_rows, "ttfa_seconds")
    total_srv = stage_vals(ok_rows, "total_seconds")
    ttfa_cli  = stage_vals(ok_rows, "ttfa_client_seconds")
    if ttfa_srv and total_srv:
        ttfa_p50   = pct(ttfa_srv, 50)
        total_p50  = pct(total_srv, 50)
        gain       = round(total_p50 - ttfa_p50, 3)
        gain_pct   = round(100 * gain / total_p50, 1) if total_p50 else 0
        net_overhead = round(pct(ttfa_cli, 50) - ttfa_p50, 3) if ttfa_cli else "?"
        print(f"\n  Streaming gain (server-side p50):")
        print(f"    TTFA {ttfa_p50}s  →  Total {total_p50}s  →  gain = {gain}s "
              f"({gain_pct}% of total time)")
        print(f"    Network overhead (client TTFA − server TTFA): ~{net_overhead}s "
              f"(audio base64 transmission over ngrok)")

    print(f"\n  Category breakdown — mean seconds (server-side)")
    print("  " + "-"*66)
    hdr = f"  {'Category':<18}" + "".join(f"{s:>9}" for s in ["ASR","TTFT","TTFA","Total","Chunks"])
    print(hdr)
    for cat in cats:
        asr   = avg(stage_vals(ok_rows, "asr_seconds",   cat))
        ttft  = avg(stage_vals(ok_rows, "ttft_seconds",  cat))
        ttfa  = avg(stage_vals(ok_rows, "ttfa_seconds",  cat))
        total = avg(stage_vals(ok_rows, "total_seconds", cat))
        nchk  = avg([float(r.get("n_text_chunks", 0))
                     for r in ok_rows if r.get("category") == cat])
        parts = [f"{v:>9}" if v is not None else f"{'—':>9}"
                 for v in [asr, ttft, ttfa, total]]
        parts.append(f"{nchk:>9.1f}" if nchk is not None else f"{'—':>9}")
        print(f"  {cat:<18}" + "".join(parts))

    ok   = sum(1 for r in ok_rows if str(r.get("routed_correctly")).lower() == "true")
    tot  = len(ok_rows)
    miss = [(r["id"], r.get("asr_transcript","")[:50])
            for r in ok_rows if str(r.get("routed_correctly")).lower() != "true"]
    print(f"\n  Routing (post-ASR): {ok}/{tot} ({round(100*ok/tot,1) if tot else 0}%)")
    for pid, tr in miss:
        print(f"    MISS {pid}: \"{tr}\"")

    print(f"\n  Results → {csv_path}")
    print(f"  Detail  → {jsonl_path}")
    print(f"{'='*62}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="E2E latency eval for Bhojpuri AI pipeline")
    parser.add_argument("--backend",      default="http://localhost:8000")
    parser.add_argument("--arm",          default="", help="Label (a/d/g) written into filename")
    parser.add_argument("--wav_dir",      default="eval-monitoring/e2e_wavs")
    parser.add_argument("--out_dir",      default="eval-monitoring/results/e2e_latency")
    parser.add_argument("--token",        default=os.environ.get("BACKEND_AUTH_TOKEN", ""))
    parser.add_argument("--timeout",      type=int,   default=120)
    parser.add_argument("--pause",        type=float, default=1.0)
    parser.add_argument("--repeats",      type=int,   default=1,
                        help="Run each prompt N times (>1 builds p95 from warm cache)")
    parser.add_argument("--streaming",    action="store_true",
                        help="Use /chat_sse_chunks; capture TTFT + TTFA (voice UX latency)")
    parser.add_argument("--synth-only",   action="store_true",
                        help="Synthesise WAVs only, skip eval")
    parser.add_argument("--no-synth",     action="store_true",
                        help="Skip synthesis, use pre-existing WAVs")
    parser.add_argument("--force-synth",  action="store_true",
                        help="Re-synthesise even if WAV already cached")
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    out_dir = Path(args.out_dir)

    # --- WAV synthesis ---
    if not args.no_synth:
        if not CAN_SYNTH:
            print("ERROR: edge-tts or miniaudio not installed.\n"
                  "  pip install edge-tts miniaudio", file=sys.stderr)
            sys.exit(1)
        print(f"Synthesising WAVs → {wav_dir}")
        wav_paths = ensure_wavs(PROMPTS, wav_dir, force=args.force_synth)
        print()
    else:
        wav_paths = {p["id"]: wav_dir / f"{p['id']}.wav" for p in PROMPTS}
        missing   = [pid for pid, wp in wav_paths.items() if not wp.exists()]
        if missing:
            print(f"ERROR: missing WAVs (run without --no-synth first): {missing}",
                  file=sys.stderr)
            sys.exit(1)

    if args.synth_only:
        print("--synth-only: done. WAVs ready in", wav_dir)
        return

    # --- Health check ---
    try:
        r = requests.get(args.backend.rstrip("/") + "/", timeout=10)
        r.raise_for_status()
        print(f"Backend OK  ({args.backend})")
    except Exception as e:
        print(f"Backend health check failed: {e}", file=sys.stderr)
        sys.exit(1)

    mode      = "stream" if args.streaming else "batch"
    arm_label = f"_arm{args.arm}" if args.arm else ""
    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path   = out_dir / f"e2e_{mode}{arm_label}_{ts}.csv"
    jsonl_path = out_dir / f"e2e_{mode}{arm_label}_{ts}.jsonl"

    if args.streaming:
        csv_fields = (["run", "id", "category", "text", "ok",
                       "asr_transcript", "category_actual", "routed_correctly", "answer"]
                      + STREAM_STAGE_KEYS
                      + ["n_text_chunks", "n_audio_chunks", "error"])
    else:
        csv_fields = (["run", "id", "category", "text", "ok",
                       "asr_transcript", "category_actual", "routed_correctly", "answer"]
                      + STAGE_KEYS + ["error"])

    endpoint = "/chat_sse_chunks" if args.streaming else "/chat_debug"
    print(f"Endpoint: {endpoint}")
    print(f"Prompts : {len(PROMPTS)} × {args.repeats} repeat(s) = "
          f"{len(PROMPTS) * args.repeats} requests\n")

    rows   = []
    n_ok   = 0
    n_fail = 0

    with open(jsonl_path, "w", encoding="utf-8") as jf, \
         open(csv_path,   "w", newline="", encoding="utf-8-sig") as cf:

        writer = csv.DictWriter(cf, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()

        for repeat in range(1, args.repeats + 1):
            for i, prompt in enumerate(PROMPTS):
                pid  = prompt["id"]
                text = prompt["text"]
                cat  = prompt["category"]
                wp   = wav_paths[pid]

                label = f"[r{repeat} {i+1:02d}/{len(PROMPTS)}]"
                print(f"{label} {pid}  {text[:38]}…", end=" ", flush=True)
                t0 = time.time()

                try:
                    if args.streaming:
                        data = call_chat_sse_chunks(
                            args.backend, wp, args.token, args.timeout, cat)
                        ttfa = data.get("ttft_seconds") or "?"  # fallback label
                        ttfa = data.get("ttfa_seconds") or ttfa
                        asr  = data.get("asr_seconds")  or "?"
                        tot  = data.get("total_seconds") or round(time.time()-t0, 3)
                        route_flag = "" if data.get("routed_correctly") else \
                                     f" [ROUTE→{data.get('category_actual','')}]"
                        print(f"OK  total={tot:.1f}s  "
                              f"(asr={asr}  ttfa={ttfa}){route_flag}")
                    else:
                        data = call_chat_debug(
                            args.backend, wp, args.token, args.timeout, cat)
                        asr  = data.get("asr_seconds")  or "?"
                        llm  = data.get("llm_seconds")  or "?"
                        tts  = data.get("tts_seconds")  or "?"
                        tot  = data.get("total_seconds") or round(time.time()-t0, 3)
                        route_flag = "" if data.get("routed_correctly") else \
                                     f" [ROUTE→{data.get('category_actual','')}]"
                        print(f"OK  {tot:.1f}s  "
                              f"(asr={asr}  llm={llm}  tts={tts}){route_flag}")

                    row = {"run": repeat, "id": pid, "category": cat, "text": text, **data}
                    jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    jf.flush()
                    writer.writerow(row)
                    cf.flush()
                    rows.append(row)
                    n_ok += 1

                except Exception as e:
                    wall = round(time.time() - t0, 3)
                    row  = {f: "" for f in csv_fields}
                    row.update({"run": repeat, "id": pid, "category": cat,
                                "text": text, "ok": False, "error": str(e)})
                    jf.write(json.dumps({"id": pid, "ok": False, "error": str(e)},
                                        ensure_ascii=False) + "\n")
                    jf.flush()
                    writer.writerow(row)
                    cf.flush()
                    rows.append(row)
                    n_fail += 1
                    print(f"FAIL ({wall:.1f}s): {e}")

                if not (repeat == args.repeats and i == len(PROMPTS) - 1):
                    time.sleep(args.pause)

    if args.streaming:
        report_streaming(rows, n_ok, n_fail, args.arm, ts, csv_path, jsonl_path)
    else:
        report_non_streaming(rows, n_ok, n_fail, args.arm, ts, csv_path, jsonl_path)


if __name__ == "__main__":
    main()
