"""Evaluate the pipeline against the labeled eval CSV.

    python bench/eval_dataset.py --csv data/t5_8bit_fully_trained_check.csv

Reports, against the gold `Correct Sentence` column:
  * exact-match and normalized-match accuracy for OUR pipeline,
  * the same for the dataset's `actual_output` (the fully-trained reference model),
  * average char-similarity to gold (input baseline vs ours vs reference),
  * a per-Use-Case breakdown,
  * latency percentiles (p50/p75/p90/p95/p99) measured on these real sentences.

No LLM judge is used — everything here is string-metric based and reproducible.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emailgrammar.pipeline import build_pipeline  # noqa: E402


# --- Scope per the clarified objective (grammar + spelling only) -------------
# The tool corrects grammar (as suggestions) and spelling (auto). It is NOT a
# formatter (currency/date/ISBN) or a style/rewrite/name-correction tool, so we
# report those buckets separately and steer on CORE.
CORE = {
    "basic-standard-lang", "Punctuation", "Unofficial Spellings & Non-Standard Form",
    "missing commas", "Incorrect word combinations", "superflous commas", "Apostrophe",
    "incorrect word order", "wrong use of tenses", "Casing Errors", "Basic Typos",
    "Basic Commas", "Agreement Errors", "Double Negation", "Word Confusion",
    "Inconsistent Spacing",
}
FORMATTING = {
    "ISBN & IBAN Numbers", "Incorrect Currency Formats",
    "Inconsistent Use of Numbers & Letters", "Incorrect Time and Date Formats",
    "Different Measurement Units", "Incorrect Dates",
}
STYLE_NAMES = {
    "Misspelled Names & Acronyms", "Wrong Name in E-Mail", "Repetition",
    "Overused Words & Phrases", "Lack of Clarity", "Informal Style Detection",
    "Foreign Terms",
}


def scope_of(use_case: str) -> str:
    if use_case in CORE:
        return "CORE (in-scope)"
    if use_case in FORMATTING:
        return "FORMATTING (out of scope)"
    if use_case in STYLE_NAMES:
        return "STYLE/NAMES (out of scope)"
    return "UNCLASSIFIED"


def norm(s: str) -> str:
    """Case/whitespace/trailing-punct-insensitive form for a lenient match."""
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return s.rstrip(".!?").strip()


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()


def pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/t5_8bit_fully_trained_check.csv")
    ap.add_argument("--model", default="mini")
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--correct-capitalized", action="store_true",
                    help="let the speller also fix Capitalized tokens (names)")
    ap.add_argument("--max-batch", type=int, default=32)
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.csv).open(encoding="utf-8", errors="replace")))
    inputs = [r["input"] for r in rows]
    golds = [r["Correct Sentence"] for r in rows]
    refs = [r.get("actual_output", "") for r in rows]      # fully-trained baseline
    cases = [r.get("Use Case", "?") for r in rows]

    print(f"Rows: {len(rows)}  |  model={args.model} beam={args.beam} "
          f"correct_capitalized={args.correct_capitalized}\n")

    # Build pipeline (optionally flip the proper-noun guard)
    overrides = {}
    pipe = build_pipeline(model=args.model, beam_size=args.beam)
    if args.correct_capitalized:
        pipe.speller.cfg.correct_capitalized = True

    # --- quality (batched) ---
    outs = pipe.correct_batch(inputs, max_batch_size=args.max_batch)

    def acc(pred: list[str]) -> tuple[float, float]:
        exact = sum(p == g for p, g in zip(pred, golds)) / len(golds)
        normed = sum(norm(p) == norm(g) for p, g in zip(pred, golds)) / len(golds)
        return exact * 100, normed * 100

    our_exact, our_norm = acc(outs)
    ref_exact, ref_norm = acc(refs)

    sim_in = statistics.mean(sim(norm(i), norm(g)) for i, g in zip(inputs, golds))
    sim_out = statistics.mean(sim(norm(o), norm(g)) for o, g in zip(outs, golds))
    sim_ref = statistics.mean(sim(norm(r), norm(g)) for r, g in zip(refs, golds))

    # improvement direction (did we move closer to gold than the raw input?)
    better = worse = same = 0
    for i, o, g in zip(inputs, outs, golds):
        si, so = sim(norm(i), norm(g)), sim(norm(o), norm(g))
        better += so > si + 1e-9
        worse += so < si - 1e-9
        same += abs(so - si) <= 1e-9

    print("=== Accuracy vs gold `Correct Sentence` ===")
    print(f"                         exact-match   normalized-match")
    print(f"  OURS ({args.model},b{args.beam}):        {our_exact:5.1f}%        {our_norm:5.1f}%")
    print(f"  reference (fully-trained): {ref_exact:5.1f}%        {ref_norm:5.1f}%")
    print(f"\n=== Avg char-similarity to gold (0-1) ===")
    print(f"  raw input : {sim_in:.3f}   (baseline: doing nothing)")
    print(f"  OURS      : {sim_out:.3f}")
    print(f"  reference : {sim_ref:.3f}")
    print(f"\n=== Direction vs input (normalized-sim) ===")
    print(f"  closer to gold: {better}  |  no change: {same}  |  further: {worse}")

    # --- per use-case (normalized match) ---
    by_case_tot = defaultdict(int)
    by_case_our = defaultdict(int)
    by_case_ref = defaultdict(int)
    for o, r, g, c in zip(outs, refs, golds, cases):
        by_case_tot[c] += 1
        by_case_our[c] += norm(o) == norm(g)
        by_case_ref[c] += norm(r) == norm(g)
    print(f"\n=== Normalized-match by Use Case ===")
    print(f"  {'use case':40} {'n':>4}  {'ours':>6}  {'ref':>6}")
    for c in sorted(by_case_tot, key=lambda k: -by_case_tot[k]):
        n = by_case_tot[c]
        print(f"  {c[:40]:40} {n:>4}  {100*by_case_our[c]/n:5.1f}%  {100*by_case_ref[c]/n:5.1f}%")

    # --- by scope bucket (steer on CORE) ---
    sc_tot = defaultdict(int)
    sc_our = defaultdict(int)
    sc_ref = defaultdict(int)
    sc_osim = defaultdict(list)
    sc_isim = defaultdict(list)
    for i, o, r, g, c in zip(inputs, outs, refs, golds, cases):
        b = scope_of(c)
        sc_tot[b] += 1
        sc_our[b] += norm(o) == norm(g)
        sc_ref[b] += norm(r) == norm(g)
        sc_osim[b].append(sim(norm(o), norm(g)))
        sc_isim[b].append(sim(norm(i), norm(g)))
    print(f"\n=== By scope bucket (the number that matters is CORE) ===")
    print(f"  {'bucket':28} {'n':>4}  {'ours':>6}  {'ref':>6}  {'ourSim':>7}  {'inSim':>7}")
    for b in ["CORE (in-scope)", "FORMATTING (out of scope)", "STYLE/NAMES (out of scope)", "UNCLASSIFIED"]:
        if not sc_tot[b]:
            continue
        n = sc_tot[b]
        arrow = "net+" if statistics.mean(sc_osim[b]) >= statistics.mean(sc_isim[b]) else "net-"
        print(f"  {b:28} {n:>4}  {100*sc_our[b]/n:5.1f}%  {100*sc_ref[b]/n:5.1f}%  "
              f"{statistics.mean(sc_osim[b]):.3f}   {statistics.mean(sc_isim[b]):.3f} {arrow}")

    # --- latency (single-request, on real sentences) ---
    lat = []
    for s in inputs:
        t0 = time.perf_counter()
        pipe.correct(s)
        lat.append((time.perf_counter() - t0) * 1000)
    print(f"\n=== Latency on eval sentences (batch=1), ms ===")
    for p in (50, 75, 90, 95, 99):
        print(f"  p{p}: {pct(lat, p):6.1f} ms")
    print(f"  mean: {statistics.mean(lat):.1f} ms")

    # --- throughput (batched over the whole set) ---
    t0 = time.perf_counter()
    pipe.correct_batch(inputs, max_batch_size=args.max_batch)
    dt = time.perf_counter() - t0
    print(f"\n=== Throughput (batched, whole set) ===")
    print(f"  {len(inputs)} sentences in {dt:.2f}s -> {len(inputs)/dt:.0f} req/s")


if __name__ == "__main__":
    main()
