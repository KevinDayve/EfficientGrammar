# EfficientGrammar — Results & Approach

*A results report for the email spelling/grammar correction engine. Written for a
technical audience and safe to share across teams. Every number is reproducible
from the harnesses in this repo (see §7).*

## 1. Objective

EfficientGrammar corrects email text on CPU, at high throughput, without calling a
large hosted model. Spelling mistakes are auto-corrected when we're confident;
grammar issues are offered as suggestions the user can accept or ignore.

The product objective is deliberately lopsided: **precision over recall — avoid
wrong suggestions even if we miss a few.** Accuracy in the usual sense is explicitly
secondary. A wrong suggestion erodes a user's trust in the feature; a missed one
costs almost nothing. That single priority shapes both how we measure the system and
what we optimize.

## 2. How we measure

The obvious metric — the fraction of sentences corrected to *exactly* the reference —
is a poor fit here. It compares against a single "correct" phrasing, so a valid
correction worded differently counts as a failure, and it rewards catching more
errors (recall), which is orthogonal to not being wrong. We report it for continuity,
but we steer by two numbers that speak to the actual objective:

- **Edit precision** — of the edits the model makes, the share that move the
  sentence *toward* a correct version. This is the headline number.
- **Wrong-suggestion count** — the edits that move *away* from correct. This is what
  we drive toward zero.

One caveat we carry everywhere: "moved away" is judged against a single reference, so
it over-counts — some of those edits are valid alternatives. It is an **upper bound**
on wrong suggestions; the true rate comes from a quick human review of the flagged
cases, which the tooling can dump on demand.

## 3. Experiments

### 3.1 Base model

We start from `visheratin/t5-efficient-mini`, a 31M-parameter T5 that is
MIT-licensed, pretrained on general English (C4) and tuned for grammar correction —
small enough to run comfortably on CPU. On our in-scope evaluation set (308 real
grammar-and-spelling sentences) it reaches 45.8% exact-match. The precision view is
the real story: it edits 72% of sentences and its edit precision is only ~56% — it
corrects too eagerly, and more than a third of its edits regress. For a
don't-be-wrong product, over-editing is the core problem, not raw accuracy.

### 3.2 Sequence model vs. edit-tagging

Before committing to an approach, we benchmarked an edit-tagging model (GECToR,
`roberta-base`) against the sequence-to-sequence T5. The tagger was expected to be
the more conservative, minimal-edit option, but on our data it was worse on every
axis: 36.7% exact-match, ~43% edit precision (68 wrong suggestions — the most of any
model we tried), and roughly 25× slower on CPU, because the only off-the-shelf
versions use large encoders. The minimal-edit intuition did not translate into
higher precision. We concluded T5 is the right base and did not pursue tagging.

### 3.3 Distillation from vennify

A larger, fully fine-tuned model — `vennify/t5-base-grammar-correction` (~220M) —
scores 74.4% on our set. We cannot ship vennify itself, but the reason is
**licensing, not size**: its licence does not permit our production use. (Size is not
the blocker — a 0.6B model, Qwen3-0.6B, is now cleared for production, so a 220M model
sits well within budget.) So instead of shipping vennify, we distil its ability into
the small, MIT-licensed `visheratin` model that we *can* ship: the student is trained
to match vennify's output distribution — a soft "learn how the expert thinks" signal
— alongside the reference corrections themselves.

The data choice mattered as much as the method. We trained on **BEA-2019
(W&I+LOCNESS)**, a corpus of *minimal edits* — corrections that change as little as
possible — which is exactly the behaviour our mandate wants. The result was
encouraging, and most so on the metric we care about. Exact-match rose from 45.8% to
50.6%, but the precision improved far more: edit precision climbed from ~56% to ~70%,
the wrong-suggestion upper bound was roughly halved (82 → 34), and the model became
more conservative (editing 58% of sentences rather than 72%). Distilling on
minimal-edit data made the model both better and safer.

### 3.4 Data style vs. volume

The obvious next move — adding a much larger corpus (Lang-8, an order of magnitude
more data) — made the model *worse* (exact-match 50.6% → 42.5%). The reason is
instructive. Lang-8's corrections are liberal fluency rewrites by many different
people — the opposite of minimal edits — and, being far larger, they swamped the
clean BEA signal. The resulting model was even more cautious (editing only 48% of
sentences) but no more precise (~67% edit precision, below the BEA model's 70%) and
caught noticeably less. The takeaway is durable: for a precision-first tool, the
*style* of the training data matters more than the quantity; more data of the wrong
style is a liability.

### 3.5 Qwen3-0.6B

Because the priority is making no mistakes, and a 0.6B general-purpose model is now
cleared for production, we are benchmarking Qwen3-0.6B on the hypothesis that a larger,
better-calibrated model errs less. Two questions decide it, and we measure both:
**throughput** — a 0.6B model decodes token by token and is a real risk against the
CPU throughput target — and **behaviour** — language models tend to over-rewrite,
which could *increase* wrong suggestions unless tightly constrained. We judge it on
the same precision-and-throughput lens as everything else, not on accuracy alone.

## 4. Reducing wrong suggestions

The robust way to raise precision — without an ever-growing rulebook — is
**confidence-gated abstention**. The model already produces a probability for each
correction; we surface an edit only when that confidence clears a calibrated
threshold, and otherwise leave the text untouched. Because low-confidence edits are
disproportionately the wrong ones, gating removes wrong suggestions first: precision
goes up, coverage goes down — exactly the trade the mandate asks for. It is a single
tunable dial, not a list to maintain, and it works on whichever model we ship.

Two things reinforce it, neither a hand-written rule. In training, we use minimal-edit
data and include a portion of already-correct examples, so the model learns when to
stay silent. In the pipeline, we keep only a small, stable set of genuine safeguards
— never altering emails, URLs, or IDs — and deliberately avoid a sprawling set of
special cases.

## 5. Status

Our best model today is the BEA-distilled `visheratin` model: 50.6% exact-match, ~70%
edit precision, an upper bound of 34 wrong suggestions over 308 sentences,
appropriately conservative, and within the CPU throughput budget. It is a shippable
v1 behind the suggestion interface.

The number we optimize from here is the wrong-suggestion count, driven down with
confidence gating (with coverage as the thing we trade away). The highest-value next
steps are: turn on confidence gating; run a precision spot-check (human review of the
flagged regressions) to convert the upper bound into a true error rate; finish the
Qwen benchmark; and — the best data we can get — collect real user accept/reject
signal once the feature is live, which is perfectly in-domain and in-style, and will
move the model further than any public corpus.

## 6. Results

All figures are on the in-scope grammar+spelling evaluation set (n=308). "Edits" is
the share of sentences the model changes; "edit precision" is the share of those
edits that improve the sentence; "wrong (≤)" is the upper bound on wrong suggestions.

| Model | Exact-match | Edit precision | Edits | Wrong (≤) | Notes |
|---|---|---|---|---|---|
| Base — visheratin-mini (31M) | 45.8% | ~56% | 72% | 82 | over-eager |
| **+ BEA distillation (visheratin)** | **50.6%** | **~70%** | **58%** | **34** | **best; shippable** |
| + raw Lang-8 | 42.5% | ~67% | 48% | 34 | wrong data style |
| GECToR (roberta-base) | 36.7% | ~43% | 64% | 68 | worse + ~25× slower |
| Teacher — vennify/t5-base (~220M) | 74.4% | — | — | — | can't ship: **licensing** |
| Qwen3-0.6B (600M) | — | — | — | — | under evaluation |

## 7. Reproducing these numbers

- `bench/eval_dataset.py --model <slot>` — accuracy, split by in-scope vs. out-of-scope.
- `bench/error_profile.py --model <slot> [--dump-regressions out.csv]` — edit
  precision, and the list of away-from-gold edits to review.
- `bench/report.py --models <slot>` — CPU throughput.
- Training/distillation and data prep: `distill/`. GECToR profile: `gector_hf/`.

## 8. Caveats

- The precision figures use a single reference, so the wrong-suggestion count is an
  upper bound; the true rate needs the human spot-check.
- Licensing must be cleared before shipping: the teacher (vennify) is excluded from
  production for licence reasons, and training corpora (BEA, Lang-8) obtained via a
  form are not automatically cleared for commercial use — distilling from vennify and
  training on these corpora should both be reviewed before release.
- Throughput figures depend on the hardware they were measured on; production numbers
  should be re-measured on the target fleet.
