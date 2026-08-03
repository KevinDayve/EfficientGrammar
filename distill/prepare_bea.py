"""Convert BEA-2019 (W&I+LOCNESS etc.) M2 files into `source<TAB>target` TSV.

    python distill/prepare_bea.py --m2 path/to/*.m2 --out distill/data/train.tsv
    python distill/prepare_bea.py --m2 wi+locness/m2/*.dev.gold.bea19.m2 \
                                  --out distill/data/dev.tsv

M2 is the standard GEC annotation format. Each block is:
    S <tokenised source sentence>
    A start end|||error_type|||correction|||required|||-NONE-|||annotator_id
    ...
    (blank line)
We apply one annotator's gold edits to the source to reconstruct the corrected
sentence -> a real (erroneous, corrected) pair. No synthetic errors.

Where to get the data (public, real learner text):
  W&I+LOCNESS v2.1 (BEA-2019):
    https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz
  (FCE, NUCLE, Lang-8/cLang8 can be added the same way if you have access.)
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path


def apply_edits(src: list[str], edits: list[tuple[int, int, str]]) -> list[str]:
    """Apply non-overlapping (start, end, replacement) edits to source tokens."""
    out: list[str] = []
    prev = 0
    for start, end, repl in sorted(edits, key=lambda e: (e[0], e[1])):
        out.extend(src[prev:start])
        if repl and repl != "-NONE-":
            out.extend(repl.split())
        prev = end
    out.extend(src[prev:])
    return out


def parse_m2(path: str, annotator: int = 0):
    src = None
    edits: list[tuple[int, int, str]] = []
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("S "):
                src, edits = line[2:].split(), []
            elif line.startswith("A ") and src is not None:
                fields = line[2:].split("|||")
                span = fields[0].split()
                start, end = int(span[0]), int(span[1])
                etype, corr, ann = fields[1], fields[2], int(fields[5])
                if etype == "noop" or start < 0 or ann != annotator:
                    continue
                edits.append((start, end, corr))
            elif line.strip() == "" and src is not None:
                pairs.append((" ".join(src), " ".join(apply_edits(src, edits))))
                src, edits = None, []
    if src is not None:
        pairs.append((" ".join(src), " ".join(apply_edits(src, edits))))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2", nargs="+", required=True, help="M2 file(s) or globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--annotator", type=int, default=0)
    ap.add_argument("--keep-identity", action="store_true",
                    help="keep pairs where source==target (precision signal); "
                         "default drops them so you can mix a controlled ratio later")
    args = ap.parse_args()

    files = [p for pat in args.m2 for p in glob.glob(pat)]
    if not files:
        raise SystemExit(f"no M2 files matched: {args.m2}")

    seen = set()
    n_ident = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for fp in files:
            for src, tgt in parse_m2(fp, args.annotator):
                if not src.strip() or not tgt.strip():
                    continue
                if src == tgt:
                    n_ident += 1
                    if not args.keep_identity:
                        continue
                if (src, tgt) in seen:
                    continue
                seen.add((src, tgt))
                w.write(f"{src}\t{tgt}\n")
    print(f"files={len(files)}  pairs written={len(seen)}  "
          f"(identity pairs {'kept' if args.keep_identity else 'dropped'}: {n_ident})")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
