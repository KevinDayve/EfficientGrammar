"""CLI: correct text from the command line or stdin.

    python -m emailgrammar "i has recieve you're emails yesterday"
    echo "some txt" | python -m emailgrammar --detailed
    python -m emailgrammar --model tiny --no-speller "..."
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(prog="emailgrammar", description="Correct spelling + grammar (CPU).")
    ap.add_argument("text", nargs="*", help="text to correct (else read stdin)")
    ap.add_argument("--model", default="mini", choices=["mini", "tiny"])
    ap.add_argument("--quantization", default="int8")
    ap.add_argument("--no-speller", action="store_true", help="skip SymSpell stage")
    ap.add_argument("--no-protect", action="store_true", help="skip entity masking")
    ap.add_argument("--detailed", action="store_true", help="show each stage's output")
    args = ap.parse_args()

    text = " ".join(args.text) if args.text else sys.stdin.read().strip()
    if not text:
        ap.error("no input text provided")

    from .pipeline import build_pipeline

    pipe = build_pipeline(
        model=args.model,
        quantization=args.quantization,
        use_speller=not args.no_speller,
        use_protector=not args.no_protect,
    )

    if args.detailed:
        c = pipe.correct(text, detailed=True)
        print(f"original : {c.original}")
        print(f"spelled  : {c.spell_corrected}")
        print(f"masked   : {c.masked}")
        print(f"final    : {c.final}" + ("   [fell back: entity unsafe]" if c.fell_back else ""))
    else:
        print(pipe.correct(text))


if __name__ == "__main__":
    main()
