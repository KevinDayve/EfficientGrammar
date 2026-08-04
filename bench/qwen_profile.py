"""Benchmark a Qwen3 causal LM as a grammar/spelling corrector on CORE.

Parallels bench/error_profile.py (T5) and gector_hf/gector_error_profile.py, so
the numbers drop straight into docs/RESULTS.md. This runs the RAW model — no
SymSpell, no entity protection, no meaning guard — which is the apples-to-apples
"is a language model less error-prone than T5?" comparison management asked for.
Production would still wrap whichever model ships in the same guards.

Reports, on the in-scope CORE set (n=308) against the gold `Correct Sentence`:
  * normalized-match  (== the scoreboard's "exact-match" column)
  * edit precision, edits%, and the wrong-suggestion upper bound (regressions)
  * optional --dump-regressions CSV of away-from-gold edits (for the spot-check)
  * sentences/sec on the run device (throughput is device-dependent; see note)

Examples
--------
  # Qwen3-0.6B, 4-bit quantized (needs bitsandbytes + a CUDA GPU)
  python bench/qwen_profile.py --model-id Qwen/Qwen3-0.6B --quant 4bit

  # Qwen3-4B-Instruct-2507-FP8 (already FP8; load as-is on a recent GPU)
  python bench/qwen_profile.py --model-id Qwen/Qwen3-4B-Instruct-2507-FP8 --quant none

Notes
-----
* enable_thinking is forced OFF. Qwen3-0.6B is a hybrid that emits <think>…</think>
  by default; that would wreck the output. The 2507-Instruct models are already
  non-thinking, so the flag is a harmless no-op there. We also strip any stray
  <think> block defensively.
* The FP8 checkpoint needs a recent transformers (>=4.54) and a GPU with FP8
  support (Hopper/Ada) or on-the-fly dequant; if transformers struggles with it,
  run that one under vLLM — the metric code here is model-agnostic and you can
  feed it predictions, but the built-in path uses transformers.
* --quant {8bit,4bit} uses bitsandbytes (8bit = INT8 LLM.int8(), 4bit = NF4), so it
  is INT-based, not FP8. Needs: pip install bitsandbytes. It is auto-skipped for a
  checkpoint that is already quantized (e.g. an -FP8 model with FineGrainedFP8Config),
  which loads via its own config — use --quant none there.
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

# --- Scope taxonomy, kept in sync with bench/eval_dataset.py ------------------
# Inlined (rather than imported) on purpose: bench.eval_dataset pulls in the
# CTranslate2 T5 pipeline, which a Qwen benchmark has no reason to require. This
# taxonomy is the frozen eval design, so duplication is safe.
CORE = {
    "basic-standard-lang", "Punctuation", "Unofficial Spellings & Non-Standard Form",
    "missing commas", "Incorrect word combinations", "superflous commas", "Apostrophe",
    "incorrect word order", "wrong use of tenses", "Casing Errors", "Basic Typos",
    "Basic Commas", "Agreement Errors", "Double Negation", "Word Confusion",
    "Inconsistent Spacing",
}


def scope_of(use_case: str) -> str:
    return "CORE (in-scope)" if use_case in CORE else "OUT (out of scope)"


def norm(s: str) -> str:
    """Case/whitespace/trailing-punct-insensitive form for a lenient match."""
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return s.rstrip(".!?").strip()


SYSTEM_PROMPT = (
    "You are a precise writing assistant. Correct only the spelling, grammar, and "
    "punctuation of the user's sentence. Do not rephrase, reword, summarize, or "
    "change the meaning, and preserve the original wording as much as possible. "
    "Output only the corrected sentence, with no quotation marks, no preamble, and "
    "no explanation."
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_PREFIX = re.compile(r"^(corrected sentence|correction|corrected|answer|output)\s*[:\-]\s*",
                     re.IGNORECASE)


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def clean(text: str) -> str:
    """Turn a raw model completion into a single corrected sentence."""
    text = _THINK.sub("", text).strip()
    # take the first non-empty line (models sometimes append commentary)
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    text = _PREFIX.sub("", text).strip()
    if len(text) >= 2 and text[0] in "\"'“‘" and text[-1] in "\"'”’":
        text = text[1:-1].strip()
    return text


def load_core(csv_path: str) -> list[tuple[str, str]]:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8", errors="replace")))
    return [(r["input"], r["Correct Sentence"]) for r in rows
            if scope_of(r["Use Case"]) == "CORE (in-scope)"]


def build_model(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # If the checkpoint is ALREADY quantized (e.g. an -FP8 model carries its own
    # FineGrainedFP8Config), we must not stack a bitsandbytes config on top of it —
    # transformers raises. Load it as-is and ignore --quant.
    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    prequant = getattr(cfg, "quantization_config", None)

    quant_cfg = None
    if prequant is not None:
        method = (prequant.get("quant_method") if isinstance(prequant, dict)
                  else getattr(prequant, "quant_method", "unknown"))
        if args.quant != "none":
            print(f"  note: '{args.model_id}' is already quantized ({method}); "
                  f"ignoring --quant {args.quant} and loading it as-is.")
    elif args.quant in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(       # bitsandbytes LLM.int8() / NF4 (INT-based)
            load_in_8bit=args.quant == "8bit",
            load_in_4bit=args.quant == "4bit",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                      # left-pad for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map=args.device,
        quantization_config=quant_cfg,
        trust_remote_code=True,
    ).eval()
    return model, tok


def generate(model, tok, batch: list[str], max_new_tokens: int) -> list[str]:
    import torch

    msgs = [[{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": s}] for s in batch]
    try:
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for m in msgs]
    except TypeError:                              # older template without the kwarg
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]

    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             num_beams=1, pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return [clean(t) for t in tok.batch_decode(gen, skip_special_tokens=True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--quant", choices=["none", "8bit", "4bit"], default="4bit",
                    help="bitsandbytes quant on load (8bit=INT8 LLM.int8, 4bit=NF4); "
                         "'none' loads as-is. Auto-ignored for already-quantized (FP8) checkpoints.")
    ap.add_argument("--device", default="auto", help="device_map (auto|cpu|cuda:0)")
    ap.add_argument("--csv", default="data/t5_8bit_fully_trained_check.csv")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--dump-regressions", nargs="?", const="qwen_regressions.csv",
                    default=None, help="write away-from-gold edits to CSV to eyeball")
    ap.add_argument("--limit", type=int, default=0, help="cap sentences (0=all; for a smoke test)")
    args = ap.parse_args()

    core = load_core(args.csv)
    if args.limit:
        core = core[:args.limit]
    srcs = [i for i, _ in core]
    golds = [g for _, g in core]

    model, tok = build_model(args)

    preds: list[str] = []
    t0 = time.perf_counter()
    for k in range(0, len(srcs), args.batch_size):
        preds.extend(generate(model, tok, srcs[k:k + args.batch_size], args.max_new_tokens))
        print(f"  ...{min(k + args.batch_size, len(srcs))}/{len(srcs)}", end="\r", flush=True)
    dt = time.perf_counter() - t0

    n = len(core)
    changed = improved = regressed = lateral = exact_edit = 0
    regs: list[dict] = []
    for src, gold, pred in zip(srcs, golds, preds):
        ni, no, ng = norm(src), norm(pred), norm(gold)
        if no == ni:
            continue                                       # left unchanged -> safe
        changed += 1
        if no == ng:
            exact_edit += 1
        si, so = sim(ni, ng), sim(no, ng)
        if so > si + args.eps:
            improved += 1
        elif so < si - args.eps:
            regressed += 1
            regs.append({"input": src, "model_output": pred, "gold": gold,
                         "sim_in": round(si, 3), "sim_out": round(so, 3)})
        else:
            lateral += 1

    unchanged = n - changed
    c = max(changed, 1)
    match = sum(norm(p) == norm(g) for p, g in zip(preds, golds)) / n * 100

    tag = f"{args.model_id} ({args.quant})"
    print(f"\n\n=== Qwen profile — {tag} — CORE n={n} ===")
    print(f"  normalized-match (scoreboard 'exact-match') : {match:5.1f}%")
    print(f"  left UNCHANGED (safe miss)                  : {unchanged:3d}  ({100*unchanged/n:4.0f}%)")
    print(f"  EDITED                                      : {changed:3d}  ({100*changed/n:4.0f}%)")
    print(f"     - improved (toward gold)                 : {improved:3d}  ({100*improved/c:4.0f}% of edits)")
    print(f"     - REGRESSED (away from gold)             : {regressed:3d}  ({100*regressed/c:4.0f}% of edits)  <- wrong-suggestion UPPER bound")
    print(f"     - lateral                                : {lateral:3d}  ({100*lateral/c:4.0f}% of edits)")
    print(f"     - exact gold among edits                 : {exact_edit:3d}  ({100*exact_edit/c:4.0f}% of edits)")
    print(f"\n  EDIT PRECISION (improved / edited) = {100*improved/c:.1f}%   wrong (<=) = {regressed}")
    print(f"  throughput: {n} sentences in {dt:.1f}s -> {n/dt:.1f} sent/s "
          f"(device-dependent; NOT comparable to the CPU T5 rps)")
    print("  note: 'regressed' is vs a single gold string, so it OVER-counts true")
    print("  errors; dump and hand-review to get the real rate (as we did for T5).")

    if match == 0.0 and changed == n and improved == 0:
        print("\n  !! WARNING: 0% match, every sentence changed, none improved.")
        print("  !! This almost always means the model loaded WRONG, not that it is bad")
        print("  !! -- e.g. FP8 scales silently dropped ('weight_scale_inv UNEXPECTED').")
        print("  !! Upgrade transformers (>=4.56) or run the checkpoint under vLLM, or")
        print("  !! benchmark the non-FP8 checkpoint with --quant 8bit. Do NOT trust this run.")

    if args.dump_regressions:
        out_path = Path(args.dump_regressions)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["input", "model_output", "gold", "sim_in", "sim_out"])
            w.writeheader()
            w.writerows(regs)
        print(f"\n  wrote {len(regs)} regressions -> {out_path}")


if __name__ == "__main__":
    main()
