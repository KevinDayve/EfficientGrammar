# Distillation — handoff for the remote GPU VM

> **You are the Claude Code instance on Kevin's GPU VM.** The engine/demo was built
> on a local CPU box (no GPU) and pushed here. `train_distill.py` is **already
> smoke-tested on CPU** (runs clean). Your job is the real training run on GPU, then
> convert + eval. Everything you need is in this repo. Data uses **real BEA-2019**
> (no synthetic errors — Kevin's call).

## Context (30-second version)
- Product: CPU-only, LLM-free **spelling (auto) + grammar (suggestions)** correction
  for email. Must **never change meaning** — precision over recall (management mandate).
- Architecture decided with data: **T5** (GECToR was benchmarked and lost — worse
  quality *and* 25× slower on CPU).
- Goal here: **distill `vennify/t5-base-grammar-correction` (74.4% CORE, too big to
  ship) into a small T5** that we can run int8 on CPU at ≥250 req/s.
- Loss = `alpha·KL(student‖teacher, temperature) + (1-alpha)·CE(student, gold)`.
  Both are T5 → shared vocab → full **logit-level** KL works (not just seq-level).
- Baselines to beat: current **T5-mini + guards = 45.8%** normalized-match on the
  in-scope CORE set (`bench/eval_dataset.py`). Teacher ceiling ≈ 74.4%.

## 0. Environment (on the VM)
```bash
# The eval CSV is already at /home/ubuntu/EmailGrammar/data/. To make that dir the
# repo (git clone refuses a non-empty dir), pull into it instead of cloning:
cd /home/ubuntu/EmailGrammar
git init && git remote add origin git@github.com:KevinDayve/EfficientGrammar.git
git fetch origin && git reset --hard origin/main   # data/ CSV is gitignored -> preserved
# (or just clone elsewhere and cp the CSV into <repo>/data/)

python -m venv .grammar && source .grammar/bin/activate
pip install -r requirements-dev.txt          # transformers, datasets, accelerate, ctranslate2, ...
# IMPORTANT: replace the CPU torch with a CUDA build for your VM's CUDA version:
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA!'; print('GPU:', torch.cuda.get_device_name(0))"
```

## 1. Data — real BEA-2019 (W&I+LOCNESS), no synthetic errors
```bash
wget https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz
tar xzf wi+locness_v2.1.bea19.tar.gz
ls wi+locness/m2/                            # confirm the .m2 filenames, then:
python distill/prepare_bea.py --m2 "wi+locness/m2/*train*.m2" --out distill/data/train.tsv
python distill/prepare_bea.py --m2 "wi+locness/m2/*dev*.m2"   --out distill/data/dev.tsv
```
`prepare_bea.py` applies the gold M2 edits to reconstruct `source<TAB>target` pairs.
Optional: add FCE / NUCLE / cLang8 the same way if you have access (more data helps).
For the precision mandate you can later mix in identity pairs with `--keep-identity`.

## 2. Smoke test on GPU (validates the *real* teacher/student + your env)
```bash
python distill/train_distill.py --smoke --train distill/data/train.tsv \
  --student visheratin/t5-efficient-mini-grammar-correction \
  --teacher vennify/t5-base-grammar-correction
```
(Downloads vennify ~850MB — fast on the VM. Tiny batch, ~seconds. Should print a
train_loss and "Saved…". If the vocab-match assert trips, fall back to sequence-KD.)

## 3. Full run
```bash
python distill/train_distill.py \
  --train distill/data/train.tsv --eval distill/data/dev.tsv \
  --student visheratin/t5-efficient-mini-grammar-correction \
  --teacher vennify/t5-base-grammar-correction \
  --out runs/distill-mini --epochs 3 --alpha 0.7 --temperature 2.0
```
Tuning (precision-first): `--alpha` (KL vs CE), `--temperature`, and student size —
try `--student google/t5-efficient-small` for more capacity if it still holds ≥250 rps.

> **Eval data note:** `bench/eval_dataset.py` reads `data/t5_8bit_fully_trained_check.csv`
> (the 589-row CORE eval set). It is **internal data, kept out of this repo**, but it
> has already been **copied to the VM at `/home/ubuntu/EmailGrammar/data/`**. If you
> clone the repo to a *different* directory, copy it in:
> `cp /home/ubuntu/EmailGrammar/data/t5_8bit_fully_trained_check.csv <repo>/data/`.

## 4. Convert → CPU runtime, then eval (turnkey: overwrite the "mini" slot)
```bash
# quantize the trained student straight into the path the pipeline loads as "mini":
ct2-transformers-converter --model runs/distill-mini \
  --output_dir data/models/t5-efficient-mini-ct2-int8 --quantization int8 --force
python -c "from transformers import AutoTokenizer; \
  AutoTokenizer.from_pretrained('runs/distill-mini').save_pretrained('data/models/t5-efficient-mini-ct2-int8')"

python bench/eval_dataset.py --model mini --beam 2   # CORE normalized-match vs 45.8%
python bench/report.py                                # throughput (must hold >=250 rps int8)
python serve.py                                       # demo now runs the distilled model
```

## 5. Try a different student size (e.g. `small` for more capacity)
Same flow, different `--student`, and its **own model slot** (keeps `mini` for
comparison — no code change; the pipeline loads by directory, not the HF key):
```bash
python distill/train_distill.py --train distill/data/train.tsv --eval distill/data/dev.tsv \
  --student google/t5-efficient-small --teacher vennify/t5-base-grammar-correction \
  --out runs/distill-small --epochs 3 --alpha 0.7 --temperature 2.0        # -> runs/distill-small/

ct2-transformers-converter --model runs/distill-small \
  --output_dir data/models/t5-efficient-small-ct2-int8 --quantization int8 --force
python -c "from transformers import AutoTokenizer; \
  AutoTokenizer.from_pretrained('runs/distill-small').save_pretrained('data/models/t5-efficient-small-ct2-int8')"

python bench/eval_dataset.py --model small --beam 2   # CORE quality (vs mini 50.6%, teacher 74.4%)
python bench/report.py --models small                 # throughput  (must hold >=250 rps)
```
Note: `google/t5-efficient-small` is **not** grammar-pretrained (unlike
visheratin-mini), so on BEA-alone it may lag mini — pair with cLang8 (§1) if so.
And watch throughput: `small` (60M) is ~2× `mini`'s compute.

## Success bar
- CORE normalized-match **well above 45.8%** (target: 70s), char-sim **net-positive**
  (> 0.937 raw-input baseline), **≥250 req/s** int8 on CPU.
- If a size holds quality but not throughput, step the student down; if it holds
  throughput but not quality, step up or raise `--alpha`/epochs.

## Report back to Kevin
CORE normalized-match (distilled vs 45.8% baseline vs 74.4% teacher), char-sim,
throughput, and the student size used. Then we swap it into the demo.

---
### File map (what's already here)
- `emailgrammar/` — engine: `speller.py` (3-tier: autofix/suggest/candidates),
  `corrector.py` (CT2 T5), `protect.py` (entity masking), `meaning.py` (never-change-
  meaning guard), `pipeline.py`.
- `serve.py` — FastAPI demo (auto-fix + clickable spelling suggestions + grammar Apply).
- `bench/` — `eval_dataset.py` (CORE metric), `report.py` (throughput), `plot_report.py`.
- `distill/` — `train_distill.py` (this), `prepare_bea.py`, `data/smoke.tsv`.
- `docs/` — `PROTOTYPE_OVERVIEW.md`, `EVAL_RESULTS.md`, `TRAINING_PLAN.md`.
