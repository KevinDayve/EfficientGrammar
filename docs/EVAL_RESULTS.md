# EmailGrammar — Evaluation Results

Results of our pipeline on the labeled eval set
`data/t5_8bit_fully_trained_check.csv`. All numbers are string-metric based and
reproducible — **no LLM judge** — via:

```bash
python bench/eval_dataset.py --model mini --beam 2
```

## The dataset

- **589 rows**, ~20 each across **29 "Use Case" categories**.
- Columns: `input` (noisy) → **`Correct Sentence`** (gold reference) · `Use Case`
  · **`actual_output`** (output of the *fully fine-tuned* 8-bit T5 — our
  comparison baseline) · an LLM "Correct/Incorrect" judgment.
- So we can compare **our base-model pipeline** directly against a **fine-tuned
  model** on the same inputs.

Two match metrics: **exact** (string-identical to gold) and **normalized**
(case-, whitespace-, and trailing-punctuation-insensitive — the fairer number).

## Headline

| System | exact-match | normalized-match | avg char-sim to gold |
|---|---|---|---|
| raw input (do nothing) | — | — | 0.893 |
| **OURS** (mini, beam 2) | 23.9% | **29.2%** | 0.887 |
| reference (fine-tuned) | 42.6% | **50.6%** | 0.909 |

**Two hard truths:**

1. **The base models are far behind a fine-tuned model** — 29% vs 51% normalized.
   Fine-tuning roughly *doubles* accuracy on this set.
2. **On average we are net-neutral-to-slightly-negative.** Our char-similarity to
   gold (0.887) is *below* doing nothing (0.893). Direction of our edits vs the
   raw input:

   | closer to gold | no change | **further from gold** |
   |---|---|---|
   | 180 | 200 | **209** |

   We move *away* from the gold slightly more often than toward it. That is the
   number to take seriously.

## Per-Use-Case breakdown (normalized-match)

| Use case | n | ours | reference |
|---|---:|---:|---:|
| Basic Typos | 20 | **90.0%** | 80.0% |
| wrong use of tenses | 21 | 85.7% | 95.2% |
| Inconsistent Spacing | 20 | 80.0% | 85.0% |
| missing commas | 21 | 71.4% | 95.2% |
| basic-standard-lang | 21 | 61.9% | 100.0% |
| Basic Commas | 20 | 55.0% | 80.0% |
| Agreement Errors | 20 | 50.0% | 85.0% |
| Incorrect word combinations | 21 | 47.6% | 76.2% |
| Word Confusion | 20 | 40.0% | 65.0% |
| Misspelled Names & Acronyms | 20 | 35.0% | 45.0% |
| Casing Errors | 20 | 35.0% | 95.0% |
| Unofficial Spellings & Non-Standard | 21 | 33.3% | 38.1% |
| Double Negation | 20 | 30.0% | 45.0% |
| Wrong Name in E-Mail | 20 | 30.0% | 70.0% |
| Apostrophe | 21 | 28.6% | 95.2% |
| superflous commas | 21 | 28.6% | 52.4% |
| Punctuation | 21 | 14.3% | 38.1% |
| incorrect word order | 21 | 9.5% | 28.6% |
| ISBN & IBAN Numbers | 20 | 5.0% | 5.0% |
| Different Measurement Units | 20 | 5.0% | 40.0% |
| Foreign Terms | 20 | 5.0% | 60.0% |
| Incorrect Currency Formats | 20 | 0.0% | 5.0% |
| Incorrect Time and Date Formats | 20 | 0.0% | 45.0% |
| Incorrect Dates | 20 | 0.0% | 30.0% |
| Inconsistent Use of Numbers & Letters | 20 | 0.0% | 0.0% |
| Repetition | 20 | 0.0% | 5.0% |
| Overused Words & Phrases | 20 | 0.0% | 0.0% |
| Lack of Clarity | 20 | 0.0% | 0.0% |
| Informal Style Detection | 20 | 0.0% | 0.0% |

### Reading the breakdown

- **Where the current design works** (competitive or ahead): **Basic Typos (90%,
  beats the fine-tuned model)**, tenses, spacing, missing commas. This validates
  the speller + T5 core on plain spelling/grammar.
- **Where our entity-protection *design* costs us:** Names & Acronyms, Casing,
  Currency, Dates, Numbers, ISBN/IBAN, Measurement units. **We deliberately
  protect capitalized words, acronyms, and anything with a digit — but this eval
  wants those corrected/reformatted.** These categories score low *by design*, not
  by accident (see the lever below).
- **Categories that aren't really "grammar":** Overused Words, Lack of Clarity,
  Informal Style, Repetition (0% for us *and* the fine-tuned model gets ~0% too) —
  these are **style/rewrite** tasks. Currency/Date/Number/ISBN are **format
  normalization** — rule-based work, not a grammar model's job.

## The design tension, quantified

Our proper-noun guard (`correct_capitalized=False`) is the direct cause of the low
"names" score. Flipping it (`--correct-capitalized`):

| | default | correct_capitalized=True |
|---|---|---|
| Misspelled Names & Acronyms | 35.0% | **60.0%** (now beats reference's 45%) |
| overall normalized-match | 29.2% | 30.1% (barely moves) |

So the guard is a real, tunable trade-off: flipping it **helps names a lot** but
risks corrupting genuine names in production, and barely moves the aggregate. The
right answer is probably *not* a blanket flag but a **names gazetteer / protected-
terms list** — decide post-meeting.

## Latency & throughput on the real eval sentences

Single-request (batch = 1, mini, beam 2) over all 589 sentences:

| p50 | p75 | p90 | p95 | p99 | mean |
|---|---|---|---|---|---|
| 53.7 ms | 68.9 ms | 88.0 ms | 106.6 ms | 157.8 ms | 59.7 ms |

Batched over the whole set: **118 req/s**. (Higher latency than the synthetic
§6 numbers because these sentences are longer and this runs beam 2, not greedy.)

> **What p50 / p75 / p99 mean.** These are **percentiles of the per-request
> latency distribution**, not averages. **p50** (the median) = half of requests
> finish faster than this — the *typical* experience. **p75** = three-quarters
> finish faster; a quarter are slower. **p99** = 99% finish faster; it's the
> *tail* — the near-worst case 1% of users hit. We track them because an average
> hides spikes: our mean (59.7 ms) looks fine, but the p99 (157.8 ms) shows some
> requests take ~3× longer, which is what an SLA must be sized against.

## Implications (for the post-meeting call)

1. **Fine-tuning looks necessary to hit this bar.** Base models ≈ half the
   fine-tuned model's accuracy. Fine-tune/distill `visheratin` (or the fully-
   trained checkpoint) on this data distribution.
2. **Reconcile the protection philosophy with the requirements.** "Never touch
   names/acronyms/numbers" (your earlier ask) directly conflicts with the eval's
   "Misspelled Names", "Casing", "Currency", "Dates" categories. Pick per-category
   behaviour; a gazetteer is the likely middle path.
3. **Scope the target.** Style (clarity, overused words, informal) and formatting
   (currency, dates, ISBN) are not grammar-model tasks — either exclude them from
   the target metric or handle with dedicated rule-based components.
4. **Keep what works.** Typos, tenses, spacing, agreement are solid — the
   speller + T5 core is sound; the gap is coverage, which fine-tuning closes.

*Reproduce:* `python bench/eval_dataset.py --model mini --beam 2`
(add `--correct-capitalized` for the names lever).
