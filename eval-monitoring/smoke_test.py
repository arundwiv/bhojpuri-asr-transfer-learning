"""
Smoke test for the Bhojpuri AI backend.

Sends a single WAV to /chat_sse_chunks and asserts:
  - All required SSE events arrive in the correct order
  - Every event carries a request_id (P1.1)
  - The done event reports ok=true
  - Latencies are logged per stage

Exit code: 0 = PASS, 1 = FAIL

Usage:
    python eval-monitoring/smoke_test.py
    python eval-monitoring/smoke_test.py --backend https://xxx.ngrok-free.app
    python eval-monitoring/smoke_test.py --backend http://localhost:8000 --wav test_wavs/q3.wav
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

# Events that must appear, in this order (others like status/tool_result are optional)
REQUIRED_SEQUENCE = ["transcript", "route", "text_chunk", "audio_chunk", "answer_final", "done"]

TIMEOUT = 120  # seconds — generous for cold LLM start


def parse_sse_events(response):
    """Yield (event_name, payload_dict) from a streaming SSE response."""
    event_name = None
    for raw_line in response.iter_lines():
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8")
        line = raw_line.strip()
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            try:
                payload = json.loads(data_str)
            except json.JSONDecodeError:
                payload = {"_raw": data_str}
            yield event_name, payload
            event_name = None


def check_health(backend: str) -> bool:
    try:
        r = requests.get(backend.rstrip("/") + "/", timeout=10)
        r.raise_for_status()
        print(f"  Health check OK  ({backend})")
        return True
    except Exception as e:
        print(f"  Health check FAILED: {e}")
        return False


def run_smoke(backend: str, wav_path: Path) -> bool:
    checks = []

    def record(name: str, passed: bool, detail: str = ""):
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        checks.append(passed)
        return passed

    print(f"\nSending {wav_path.name} -> {backend}/chat_sse_chunks\n")

    t_start = time.time()
    try:
        with open(wav_path, "rb") as f:
            response = requests.post(
                backend.rstrip("/") + "/chat_sse_chunks",
                files={"audio": (wav_path.name, f, "audio/wav")},
                stream=True,
                timeout=TIMEOUT,
            )
        response.raise_for_status()
    except Exception as e:
        print(f"  [FAIL] Request failed: {e}")
        return False

    events_seen = []       # list of event names in arrival order
    missing_rid = []       # events without request_id
    final_ok = None
    server_t_by_event = {}
    first_seen = {}        # event_name -> server_t of first occurrence

    for event_name, payload in parse_sse_events(response):
        if event_name is None:
            continue
        events_seen.append(event_name)

        # Track request_id presence
        if "request_id" not in payload:
            missing_rid.append(event_name)

        # Track server_t per event type (first occurrence)
        server_t = payload.get("server_t")
        if server_t is not None and event_name not in first_seen:
            first_seen[event_name] = server_t

        if event_name == "done":
            final_ok = payload.get("ok", False)

    wall_time = round(time.time() - t_start, 2)

    # --- Assertions ---

    # 1. Required event sequence
    seq_cursor = 0
    for ev in events_seen:
        if seq_cursor < len(REQUIRED_SEQUENCE) and ev == REQUIRED_SEQUENCE[seq_cursor]:
            seq_cursor += 1
    sequence_ok = seq_cursor == len(REQUIRED_SEQUENCE)
    found = [e for e in REQUIRED_SEQUENCE if e in events_seen]
    missing = [e for e in REQUIRED_SEQUENCE if e not in events_seen]
    record(
        "Required SSE event sequence",
        sequence_ok,
        f"found={found}" + (f" MISSING={missing}" if missing else ""),
    )

    # 2. request_id on every event (P1.1)
    record(
        "request_id present on all events",
        len(missing_rid) == 0,
        f"{len(missing_rid)} events missing it: {missing_rid}" if missing_rid else f"{len(events_seen)} events checked",
    )

    # 3. done.ok == true
    record("done event reports ok=true", final_ok is True, f"ok={final_ok}")

    # 4. Multiple text and audio chunks (pipeline is actually streaming)
    n_text  = events_seen.count("text_chunk")
    n_audio = events_seen.count("audio_chunk")
    record("text_chunk received",  n_text  > 0, f"count={n_text}")
    record("audio_chunk received", n_audio > 0, f"count={n_audio}")

    # --- Latency summary ---
    print(f"\n  Latencies (server_t at first event of each type, seconds):")
    for ev in ["transcript", "route", "text_chunk", "audio_chunk", "answer_final", "done"]:
        t = first_seen.get(ev)
        print(f"    {ev:<20} {f'{t:.3f}s' if t is not None else 'n/a'}")
    print(f"  Wall time: {wall_time}s")

    return all(checks)


def main():
    parser = argparse.ArgumentParser(description="Smoke test for Bhojpuri AI /chat_sse_chunks")
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--wav", default=None, help="WAV file to send (default: first file in test_wavs/)")
    args = parser.parse_args()

    # Resolve WAV
    if args.wav:
        wav_path = Path(args.wav)
    else:
        candidates = sorted(Path("test_wavs").glob("*.wav")) + sorted(Path("test_wavs").glob("*.WAV"))
        seen, unique = set(), []
        for p in candidates:
            k = p.resolve()
            if k not in seen:
                seen.add(k)
                unique.append(p)
        wav_path = unique[0] if unique else None

    if not wav_path or not wav_path.exists():
        print("ERROR: no WAV file found. Use --wav path/to/file.wav or drop a file in test_wavs/")
        sys.exit(1)

    print("=" * 55)
    print("Bhojpuri AI — Backend Smoke Test")
    print("=" * 55)

    if not check_health(args.backend):
        sys.exit(1)

    passed = run_smoke(args.backend, wav_path)

    print()
    print("=" * 55)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print("=" * 55)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
