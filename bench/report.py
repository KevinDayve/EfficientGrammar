"""Reproducible benchmark report for the prototype writeup.

Runs a fixed matrix (latency, throughput-vs-batch, beam sweep, speller speed)
with median-of-N to damp noise, prints a formatted table, and writes a CSV so
the numbers in the docs are regenerable and plottable.

    python bench/report.py --out docs/assets/bench.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.benchmark import CORPUS, percentiles  # noqa: E402
from emailgrammar.pipeline import build_pipeline  # noqa: E402
from emailgrammar.speller import Speller  # noqa: E402


def corpus(n: int) -> list[str]:
    return [CORPUS[i % len(CORPUS)] for i in range(n)]


def median_throughput(pipe, bs: int, total: int, runs: int = 3) -> float:
    texts = corpus(total)
    pipe.correct_batch(texts[:bs])  # warmup
    rates = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for i in range(0, total, bs):
            pipe.correct_batch(texts[i:i + bs], max_batch_size=bs)
        rates.append(total / (time.perf_counter() - t0))
    return statistics.median(rates)


def latency(pipe, n: int) -> dict:
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        pipe.correct(CORPUS[i % len(CORPUS)])
        times.append((time.perf_counter() - t0) * 1000)
    p = percentiles(times)
    return {"mean": statistics.mean(times), "p50": p[50], "p95": p[95], "p99": p[99]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets/bench.csv")
    ap.add_argument("--total", type=int, default=192)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    ap.add_argument("--models", nargs="+", default=["mini", "tiny"],
                    help="model slots to benchmark; any not yet converted are skipped")
    args = ap.parse_args()

    rows = []
    print(f"System: {platform.platform()}  cores={os.cpu_count()}  py={platform.python_version()}")
    print(f"Matrix: total={args.total} sentences, median of {args.runs} runs\n")

    # Speller alone (throughput ceiling of stage 1)
    sp = Speller()
    for t in CORPUS:
        sp.correct(t)
    t0 = time.perf_counter()
    for i in range(5000):
        sp.correct(CORPUS[i % len(CORPUS)])
    sp_rps = 5000 / (time.perf_counter() - t0)
    print(f"Speller (stage 1) alone: {sp_rps:,.0f} sentences/s\n")
    rows.append({"metric": "speller_only", "model": "-", "beam": "-", "batch": "-", "value": round(sp_rps)})

    # Only benchmark model slots that have actually been converted.
    from emailgrammar.config import CorrectorConfig
    present = [m for m in args.models if CorrectorConfig(model_key=m).ct2_dir.exists()]
    for m in args.models:
        if m not in present:
            print(f"(skip '{m}': {CorrectorConfig(model_key=m).ct2_dir} not found)")
    if not present:
        raise SystemExit("no converted models found -- run the conversion step first")

    # Latency + throughput-vs-batch (full pipeline, beam=1)
    print(f"{'model':6} {'lat p50':>8} {'p95':>7} {'p99':>7} | throughput req/s by batch")
    print("-" * 78)
    for model in present:
        pipe = build_pipeline(model=model, beam_size=1)
        lat = latency(pipe, 48)
        line = f"{model:6} {lat['p50']:7.1f}m {lat['p95']:6.1f}m {lat['p99']:6.1f}m |"
        for bs in args.batch_sizes:
            rps = median_throughput(pipe, bs, args.total, args.runs)
            line += f"  b{bs}={rps:.0f}"
            rows.append({"metric": "throughput", "model": model, "beam": 1, "batch": bs, "value": round(rps, 1)})
        print(line)
        for k in ("p50", "p95", "p99"):
            rows.append({"metric": f"latency_{k}_ms", "model": model, "beam": 1, "batch": 1, "value": round(lat[k], 1)})

    # Beam sweep on the first available model, batch=16 -- quality/throughput trade
    sweep_model = present[0]
    print(f"\nBeam sweep ({sweep_model}, batch=16):")
    for beam in [1, 2, 4, 5]:
        pipe = build_pipeline(model=sweep_model, beam_size=beam)
        rps = median_throughput(pipe, 16, args.total, args.runs)
        print(f"  beam={beam}: {rps:.0f} req/s")
        rows.append({"metric": "beam_throughput", "model": sweep_model, "beam": beam, "batch": 16, "value": round(rps, 1)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "model", "beam", "batch", "value"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
