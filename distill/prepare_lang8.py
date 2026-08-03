"""Parse the Lang-8 v1.0 `entries.(train|test)` file into source<TAB>target TSV.

    unzip lang-8-en-1.0.zip                       # -> lang-8-en-1.0/entries.train
    python distill/prepare_lang8.py --entries lang-8-en-1.0/entries.train \
                                    --out distill/data/lang8.tsv
    cat distill/data/lang8.tsv >> distill/data/train.tsv     # append to BEA pairs

Lang-8 line format (tab-separated):
    [n_corrections] [serial] [url] [sentence_no] [learner_sentence] [correction_1] [correction_2] ...
We emit (learner_sentence -> first correction) for lines with >=1 correction.

Note: raw Lang-8 is noisier than the cleaned cLang8 (learners correcting each
other, comments, partial edits). We apply light filters (drop no-change pairs and
wild length ratios) but it is not as clean as cLang8. Good for volume; expect some
noise. (Verify the corpus license before a Lang-8-trained model ships.)
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse(entries_path: str, keep_identity: bool = False, max_ratio: float = 3.0):
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    kept_identity = 0
    with open(entries_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue                        # blank line separates entries
            cols = line.split("\t")
            if len(cols) < 6:
                continue
            try:
                n_corr = int(cols[0])
            except ValueError:
                continue
            if n_corr < 1:
                continue                        # no correction offered -> skip
            src, tgt = cols[4].strip(), cols[5].strip()   # learner, first correction
            if not src or not tgt:
                continue
            if src == tgt:
                if not keep_identity:
                    continue
                kept_identity += 1
            # noise guard: a "correction" that is wildly longer/shorter than the
            # source is usually a comment, not an edit.
            ls, lt = len(src.split()), len(tgt.split())
            if not ls or not lt or lt > max_ratio * ls or ls > max_ratio * lt:
                continue
            if (src, tgt) in seen:
                continue
            seen.add((src, tgt))
            out.append((src, tgt))
    return out, kept_identity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", required=True, help="path to Lang-8 entries.train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-identity", action="store_true")
    ap.add_argument("--max-ratio", type=float, default=3.0)
    args = ap.parse_args()

    pairs, n_ident = parse(args.entries, args.keep_identity, args.max_ratio)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as w:
        for s, t in pairs:
            w.write(f"{s}\t{t}\n")
    print(f"wrote {len(pairs)} pairs -> {out}  (identity kept: {n_ident})")


if __name__ == "__main__":
    main()
