"""Latency + throughput benchmark for the EmailGrammar pipeline.

Throughput is THE constraint (target: 250+ req/s), so we measure it three ways:
  1. single-request latency percentiles (what one user feels),
  2. batched throughput (one thread, varying batch size),
  3. concurrent throughput (N threads sharing one Translator -- CTranslate2
     releases the GIL, so this is the realistic serving model).

    python bench/benchmark.py --model mini --quantization int8
    python bench/benchmark.py --model tiny --intra-threads 1 --inter-threads 4 --threads 4
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emailgrammar.pipeline import build_pipeline  # noqa: E402

# Noisy email-ish sentences (spelling + grammar errors).
CORPUS = [
    "i has recieve you're emails yesterday and wil responsd son.",
    "Their are alot of things we need too discus before the meating.",
    "Please kindly to find attach the report which you was requesting.",
    "He dont have no time for finishing this projet by tommorow.",
    "We was hoping that you could sended us the invoice agian.",
    "Thankyou for you patients, i apreciate it alot realy.",
    "The team have been working hardly on the new feature's.",
    "Can you confirmed if the payement has been proccessed correctly?",
    "Me and him is going to the conference next weak.",
    "She dont know weather the client will excepts our proposel.",
    "Kindly revert back to me on the earliest as this is urgant.",
    "Their was a issue with the server, we are looking into it currenly.",
]


def percentiles(xs: list[float], ps=(50, 95, 99)) -> dict[int, float]:
    s = sorted(xs)
    out = {}
    for p in ps:
        k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
        out[p] = s[k]
    return out


def bench_latency(pipe, n: int) -> None:
    print("\n=== 1. Single-request latency (batch size 1) ===")
    times = []
    for i in range(n):
        text = CORPUS[i % len(CORPUS)]
        t0 = time.perf_counter()
        pipe.correct(text)
        times.append((time.perf_counter() - t0) * 1000)
    pct = percentiles(times)
    print(f"  n={n}  mean={statistics.mean(times):.1f}ms  "
          f"p50={pct[50]:.1f}ms  p95={pct[95]:.1f}ms  p99={pct[99]:.1f}ms")
    print(f"  -> single-thread serial throughput: {1000 / statistics.mean(times):.1f} req/s")


def bench_batched(pipe, batch_sizes, total: int) -> None:
    print("\n=== 2. Batched throughput (1 thread) ===")
    for bs in batch_sizes:
        texts = [CORPUS[i % len(CORPUS)] for i in range(total)]
        t0 = time.perf_counter()
        for i in range(0, total, bs):
            pipe.correct_batch(texts[i:i + bs], max_batch_size=bs)
        dt = time.perf_counter() - t0
        print(f"  batch_size={bs:<4} {total} sentences in {dt:.2f}s -> {total / dt:.1f} req/s")


def bench_concurrent(pipe, threads: int, per_thread: int) -> None:
    print(f"\n=== 3. Concurrent throughput ({threads} threads, shared Translator) ===")
    barrier = threading.Barrier(threads + 1)
    counts = [0] * threads

    def worker(idx: int) -> None:
        barrier.wait()
        for i in range(per_thread):
            pipe.correct(CORPUS[(idx + i) % len(CORPUS)])
            counts[idx] += 1

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    total = sum(counts)
    print(f"  {total} sentences across {threads} threads in {dt:.2f}s -> {total / dt:.1f} req/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mini", choices=["mini", "tiny"])
    ap.add_argument("--quantization", default="int8")
    ap.add_argument("--no-speller", action="store_true")
    ap.add_argument("--beam-size", type=int, default=1)
    ap.add_argument("--intra-threads", type=int, default=0)
    ap.add_argument("--inter-threads", type=int, default=1)
    ap.add_argument("--latency-n", type=int, default=48)
    ap.add_argument("--batch-total", type=int, default=96)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--per-thread", type=int, default=24)
    args = ap.parse_args()

    print(f"Building pipeline: model={args.model} quant={args.quantization} "
          f"speller={not args.no_speller} beam={args.beam_size} "
          f"intra={args.intra_threads} inter={args.inter_threads}")
    pipe = build_pipeline(
        model=args.model,
        quantization=args.quantization,
        use_speller=not args.no_speller,
        beam_size=args.beam_size,
        intra_threads=args.intra_threads,
        inter_threads=args.inter_threads,
    )

    # Warmup (first call pays lazy init / graph setup costs).
    pipe.correct_batch(CORPUS)
    print("\n=== Sample corrections ===")
    for c in pipe.correct_batch(CORPUS[:4], detailed=True):
        print(f"  in : {c.original}")
        print(f"  out: {c.final}\n")

    bench_latency(pipe, args.latency_n)
    bench_batched(pipe, [1, 4, 8, 16], args.batch_total)
    bench_concurrent(pipe, args.threads, args.per_thread)


if __name__ == "__main__":
    main()
