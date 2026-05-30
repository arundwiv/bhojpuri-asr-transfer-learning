"""
Latency report from latency.csv produced by the backend.

Usage:
    python latency_report.py [path/to/latency.csv]

Prints mean / p50 / p95 for each timing column.
"""
import csv
import statistics
import sys
from pathlib import Path

_FIELDS = [
    "asr_ms",
    "router_ms",
    "tool_ms",
    "llm_first_token_ms",
    "tts_first_chunk_ms",
    "total_ms",
]


def _pct(data: list[float], p: int) -> float:
    data = sorted(data)
    idx = max(0, int(len(data) * p / 100) - 1)
    return data[idx]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("latency.csv")
    if not path.exists():
        print(f"No CSV found at {path}")
        print("Run the backend and make some requests first.")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("CSV is empty.")
        sys.exit(0)

    print(f"Requests analysed: {len(rows)}\n")
    print(f"{'Metric':<25} {'mean':>8} {'p50':>8} {'p95':>8}  (ms)")
    print("-" * 60)

    for field in _FIELDS:
        vals = []
        for r in rows:
            v = r.get(field, "")
            if v not in ("", "None", None):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass

        if not vals:
            print(f"{field:<25} {'n/a':>8} {'n/a':>8} {'n/a':>8}")
            continue

        mean = round(statistics.mean(vals), 1)
        p50 = round(_pct(vals, 50), 1)
        p95 = round(_pct(vals, 95), 1)
        print(f"{field:<25} {mean:>8} {p50:>8} {p95:>8}")


if __name__ == "__main__":
    main()
