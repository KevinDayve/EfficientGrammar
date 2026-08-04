"""Precision-focused error profile.

Exact-match "accuracy" is the wrong lens for a "don't give wrong suggestions"
mandate. What matters: of the edits the model actually makes, how many HELP (move
toward the gold) vs HURT (move away)? And how often does it correctly leave text
alone? This reports that.

    python bench/error_profile.py --model mini --beam 2
    python bench/error_profile.py --model mini --dump-regressions regressions.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.eval_dataset import norm, scope_of  # noqa: E402
from emailgrammar.pipeline import build_pipeline  # noqa: E402


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mini")
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--csv", default="data/t5_8bit_fully_trained_check.csv")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--dump-regressions", nargs="?", const="regressions.csv", default=None,
                    help="write the away-from-gold edits to a CSV to eyeball (input/output/gold)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8", errors="replace")))
    core = [(r["input"], r["Correct Sentence"]) for r in rows
            if scope_of(r["Use Case"]) == "CORE (in-scope)"]
    pipe = build_pipeline(model=args.model, beam_size=args.beam)
    outs = pipe.correct_batch([i for i, _ in core], max_batch_size=32)

    n = len(core)
    changed = improved = regressed = lateral = exact_edit = 0
    regs: list[dict] = []
    for (inp, gold), out in zip(core, outs):
        ni, no, ng = norm(inp), norm(out), norm(gold)
        if no == ni:
            continue                                  # left unchanged -> safe
        changed += 1
        if no == ng:
            exact_edit += 1
        si, so = sim(ni, ng), sim(no, ng)
        if so > si + args.eps:
            improved += 1
        elif so < si - args.eps:
            regressed += 1
            regs.append({"input": inp, "model_output": out, "gold": gold,
                         "sim_in": round(si, 3), "sim_out": round(so, 3)})
        else:
            lateral += 1

    unchanged = n - changed
    c = max(changed, 1)
    print(f"\n=== Error profile ({args.model}, beam {args.beam}) — CORE n={n} ===")
    print(f"  left UNCHANGED (safe miss)     : {unchanged:3d}  ({100*unchanged/n:4.0f}%)")
    print(f"  EDITED                         : {changed:3d}  ({100*changed/n:4.0f}%)")
    print(f"     - improved (toward gold)    : {improved:3d}  ({100*improved/c:4.0f}% of edits)")
    print(f"     - REGRESSED (away from gold): {regressed:3d}  ({100*regressed/c:4.0f}% of edits)  <- wrong-suggestion UPPER bound")
    print(f"     - lateral                   : {lateral:3d}  ({100*lateral/c:4.0f}% of edits)")
    print(f"     - exact gold among edits    : {exact_edit:3d}  ({100*exact_edit/c:4.0f}% of edits)")
    print(f"\n  EDIT PRECISION (improved / edited) = {100*improved/c:.1f}%  <- the mandate's number")
    print("  note: 'regressed' = moved away from the SINGLE gold string; some are")
    print("  valid alternative corrections, so this OVER-counts true errors.")

    if args.dump_regressions:
        out_path = Path(args.dump_regressions)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["input", "model_output", "gold", "sim_in", "sim_out"])
            w.writeheader()
            w.writerows(regs)
        print(f"\n  wrote {len(regs)} regressions -> {out_path}  (judge these: how many are")
        print("  genuinely wrong vs. valid alternatives? that is your TRUE error rate)")
        for r in regs[:5]:
            print(f"    in : {r['input']}")
            print(f"    out: {r['model_output']}")
            print(f"    ref: {r['gold']}\n")


if __name__ == "__main__":
    main()
