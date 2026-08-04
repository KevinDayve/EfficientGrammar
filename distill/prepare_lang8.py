"""Parse the Lang-8 v1.0 `entries.(train|test)` file into source<TAB>target TSV.

    unzip lang-8-en-1.0.zip                       # -> lang-8-en-1.0/entries.train
    python distill/prepare_lang8.py --entries lang-8-en-1.0/entries.train \
                                    --out distill/data/lang8.tsv \
                                    --max-edits 3 --limit 70000

Lang-8 line format (tab-separated):
    [n_corrections] [serial] [url] [sentence_no] [learner_sentence] [correction_1] ...
We emit (learner_sentence -> first correction) for lines with >=1 correction.

Why the filters matter (learned the hard way): raw Lang-8 corrections are liberal
FLUENCY rewrites by random native speakers, while our objective + eval are
MINIMAL-edit. Training on unfiltered Lang-8 *lowered* CORE (50.6 -> 42.5) because
it taught the model to over-rewrite and swamped the clean BEA signal. So:
  --max-edits N : keep only pairs whose correction changes <= N words (minimal edits)
  --limit K     : cap the count so Lang-8 doesn't drown the BEA pairs
"""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path


def _n_edits(src: str, tgt: str) -> int:
    """Word-level edit count between source and target."""
    a, b = src.split(), tgt.split()
    sm = SequenceMatcher(None, a, b)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def parse(entries_path, keep_identity=False, max_ratio=3.0, max_edits=0, limit=0):
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    with open(entries_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < 6:
                continue
            try:
                n_corr = int(cols[0])
            except ValueError:
                continue
            if n_corr < 1:
                continue
            src, tgt = cols[4].strip(), cols[5].strip()
            if not src or not tgt or (src == tgt and not keep_identity):
                continue
            ls, lt = len(src.split()), len(tgt.split())
            if not ls or not lt or lt > max_ratio * ls or ls > max_ratio * lt:
                continue
            if max_edits and _n_edits(src, tgt) > max_edits:
                continue                          # drop big rewrites -> keep minimal edits
            if (src, tgt) in seen:
                continue
            seen.add((src, tgt))
            out.append((src, tgt))
            if limit and len(out) >= limit:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", required=True, help="path to Lang-8 entries.train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-identity", action="store_true")
    ap.add_argument("--max-ratio", type=float, default=3.0)
    ap.add_argument("--max-edits", type=int, default=0,
                    help="keep only pairs changing <= N words (0 = no limit); 3 is a good minimal-edit filter")
    ap.add_argument("--limit", type=int, default=0, help="cap number of pairs (0 = all)")
    args = ap.parse_args()

    pairs = parse(args.entries, args.keep_identity, args.max_ratio, args.max_edits, args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as w:
        for s, t in pairs:
            w.write(f"{s}\t{t}\n")
    print(f"wrote {len(pairs)} pairs -> {out}  (max_edits={args.max_edits}, limit={args.limit})")


if __name__ == "__main__":
    main()
