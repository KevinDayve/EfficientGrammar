"""Convert a HuggingFace T5 grammar model to CTranslate2 format.

    python scripts/convert_model.py --model mini --quantization int8
    python scripts/convert_model.py --model tiny --quantization int8

Downloads the checkpoint, quantizes it, and saves the tokenizer alongside so the
resulting directory is fully self-contained for the (torch-free) runtime.

Requires the dev deps (torch + transformers): pip install -r requirements-dev.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/convert_model.py ...)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctranslate2.converters import TransformersConverter  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from emailgrammar.config import HF_MODELS, ct2_dir_for  # noqa: E402


def convert(model_key: str, quantization: str, force: bool = True) -> Path:
    hf_model = HF_MODELS[model_key]
    out_dir = ct2_dir_for(model_key, quantization)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] Converting {hf_model} -> {out_dir}  (quantization={quantization})")
    TransformersConverter(hf_model).convert(
        str(out_dir), quantization=quantization, force=force
    )

    print(f"[2/2] Saving tokenizer into {out_dir}")
    AutoTokenizer.from_pretrained(hf_model).save_pretrained(str(out_dir))

    size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
    print(f"Done. Model dir is {size_mb:.1f} MB")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="HF T5 -> CTranslate2 converter")
    ap.add_argument("--model", default="mini", choices=list(HF_MODELS))
    ap.add_argument(
        "--quantization",
        default="int8",
        choices=["int8", "int8_float32", "int16", "float16", "float32"],
    )
    ap.add_argument("--no-force", action="store_true", help="fail if output exists")
    args = ap.parse_args()
    convert(args.model, args.quantization, force=not args.no_force)


if __name__ == "__main__":
    main()
