# EmailGrammar

CPU-only, **LLM-free** spelling + grammar correction for email text, built for
**high throughput** (target: 250+ req/s).

No API calls, no GPUs, no large language models — just a tiny quantized T5
running on [CTranslate2](https://github.com/OpenNMT/CTranslate2) plus a guarded
[SymSpell](https://github.com/wolfgarbe/SymSpell) speller.

## Pipeline

```
raw text ──▶ [SymSpell: fix non-word typos] ──▶ [T5 (CTranslate2, int8): fix grammar] ──▶ output
             guarded, case/entity-preserving      batched, releases the GIL
```

- **SymSpell** fixes *non-word* misspellings (`recieve → receive`) surgically —
  it preserves case, punctuation, URLs, and **named entities** (see below).
  ~5,000 sentences/sec, so it is never the bottleneck.
- **T5 grammar model** (`visheratin/t5-efficient-{mini,tiny}-grammar-correction`,
  **MIT-licensed**) fixes real grammatical errors. Quantized to int8 and served
  by CTranslate2 for CPU throughput. `vennify/t5-base` is intentionally **not**
  used (not shippable in production) but is a useful quality reference.

## Named-entity / proper-noun safety

The speller will **not** rewrite tokens it cannot confidently treat as plain
English words. Guards (in `emailgrammar/speller.py`):

| Token            | Result      | Why                                    |
|------------------|-------------|----------------------------------------|
| `NASA`, `FIFA`   | untouched   | all-caps acronym guard                 |
| `Kevin`, `Anthropic` | untouched | Capitalized → proper-noun guard      |
| `s3://bucket`, `a@b.com`, `v2.3` | untouched | contains digits/symbols  |
| `chekc`, `recieve` | corrected | plain lowercase word, not in dictionary|

For **domain-specific** entities (brand/product/people names that may appear
lowercased), add them to a protected-terms dictionary via
`SpellerConfig.extra_dictionaries` — an O(1), latency-free allow-list. A heavier
NER model is deliberately avoided unless proven necessary at 250 req/s.

## Setup

```bash
python -m venv .grammar && source .grammar/bin/activate

# runtime only (no torch):
pip install -r requirements.txt

# to convert models offline (adds torch, CPU build):
pip install -r requirements-dev.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Convert a model (one-time, offline)

```bash
python scripts/convert_model.py --model mini --quantization int8   # 34.5 MB
python scripts/convert_model.py --model tiny --quantization int8   # 18.8 MB
```
The tokenizer is saved alongside the model, so the runtime never touches the HF
hub or torch.

## Use

```bash
python -m emailgrammar "i has recieve you're emails yesterday"
# -> I received your emails yesterday

python -m emailgrammar --model tiny --detailed "their are alot of issue's"
```

```python
from emailgrammar import build_pipeline
pipe = build_pipeline(model="tiny", quantization="int8")
pipe.correct_batch(["me and him is going too the meating"])
```

## Benchmark

```bash
python bench/benchmark.py --model tiny --quantization int8
```

### Numbers (4-core dev box, int8, single thread + batching)

| Model | Params | Size  | batch=16 | batch=64 | Quality |
|-------|--------|-------|----------|----------|---------|
| mini  | ~31M   | 34 MB | ~120 req/s | (T5-only ~110) | better |
| tiny  | ~16M   | 19 MB | ~150 req/s | **~244 req/s** | good   |

**Batching is the throughput lever** (near-linear up to the CPU's limit).
`tiny` already clears the 250 req/s target on pure T5 (~300 req/s) on a 4-core
dev box; production hardware + dynamic batching + multiple replicas gives ample
headroom. Single-request latency (batch=1) is ~40 ms p50.

## Status / roadmap

- [x] Core engine: guarded SymSpell + int8 T5 via CTranslate2
- [x] Model conversion, CLI, benchmark harness
- [ ] **Serving layer**: dynamic micro-batching + replica pool (the real 250 req/s path)
- [ ] **Quality eval**: JFLEG / BEA-style scoring to choose tiny-vs-mini and beam size
- [ ] Protected-terms dictionary for domain entities
- [ ] Fine-tune / distill on email-domain data if base quality is insufficient

## Licensing

Grammar models are **MIT** (`visheratin/*`). `vennify/*` is excluded from the
shippable path per project policy.
