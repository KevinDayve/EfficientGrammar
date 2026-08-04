# EfficientGrammar — Results & Approach

*A results report for the email spelling/grammar correction engine. Written for a
technical audience and safe to share across teams. Every number here is
reproducible from the harnesses in this repo (see §7).*

## 1. What we're building, and the objective that governs everything

EfficientGrammar corrects email text on CPU, at high throughput, without calling a
large hosted model. Spelling mistakes are auto-corrected when we're confident;
grammar issues are offered as **suggestions** the user can accept or ignore.

The product objective is the important part, and it is deliberately lopsided:
**precision over recall — "avoid wrong suggestions even if we miss a few."**
Accuracy, in the usual sense, is explicitly secondary. The reasoning is simple: a
wrong suggestion erodes a user's trust in the feature, whereas a missed one costs
almost nothing. That single priority shapes both *how we measure* the system and
*what we optimize*.

## 2. Why plain "accuracy" is the wrong scoreboard

The obvious metric — the fraction of sentences the model corrects to *exactly* the
reference — is a poor fit here for two reasons. First, it compares against a single
"correct" phrasing, so a perfectly valid correction worded differently is counted
as a failure. Second, and more importantly, it rewards *catching more errors*
(recall), which is orthogonal to our actual goal of *not being wrong*.

So we measure the thing the mandate cares about directly. For every sentence the
model touches, we ask whether its edit moved the text **toward** a correct version
or **away** from it. That yields two numbers we actually steer by:

- **Edit precision** — of the edits the model makes, the share that improve the
  sentence. This is our headline number.
- **Wrong-suggestion rate** — the edits that move *away* from correct. This is what
  we drive toward zero.

One honest caveat we carry everywhere: "moved away" is measured against a single
reference, so it *over-counts* — some of those edits are valid alternatives. It is
an **upper bound** on wrong suggestions; the true rate is obtained by a quick human
review of the flagged cases (which the tooling can dump on demand).

## 3. The experiments

### 3.1 Starting point — the base model

We start from `visheratin/t5-efficient-mini`, a 31M-parameter T5 that is
MIT-licensed, pretrained on general English (C4) and tuned for grammar correction —
small enough to run comfortably on CPU. On our in-scope evaluation set (308 real
grammar-and-spelling sentences), it reaches **45.8%** exact-match. But the
precision view tells the real story: it **edits 72% of sentences** and its **edit
precision is only ~56%** — it corrects too eagerly, and more than a third of its
edits regress. For a "don't be wrong" product, over-editing is the core problem, not
raw accuracy.

### 3.2 Architecture check — sequence model vs. edit-tagging

Before investing in one approach, we benchmarked an edit-tagging model (GECToR
style) as an alternative to the sequence-to-sequence T5. On the same data the T5
was both **more accurate and far faster on CPU** (the tagging model scored ~37% and
ran roughly 25× slower, because the only off-the-shelf versions use large
encoders). We concluded T5 is the right base and did not pursue tagging further.

### 3.3 Making it more precise — distillation on minimal-edit data

A larger, fully fine-tuned model (a T5-base grammar model) scores ~74% on our set,
but it is too big to serve on CPU within our throughput budget. So rather than ship
it, we **distil its ability into the small model**: the student is trained to match
the large model's output distribution (a soft, "learn-how-the-expert-thinks" signal)
alongside the reference corrections themselves.

The *data* choice was as important as the method. We trained on **BEA-2019
(W&I+LOCNESS)**, a corpus of **minimal edits** — corrections that change as little as
possible. That aligns precisely with our mandate: we want a model that makes small,
safe fixes, not one that rewrites.

The result was encouraging, and most so on the metric we care about. Exact-match
rose from 45.8% to **50.6%**, but the precision improved much more: **edit precision
climbed from ~56% to ~70%**, the wrong-suggestion upper bound was roughly **halved
(82 → 34)**, and the model grew **more conservative** (editing 58% of sentences
rather than 72%). In short, distillation on minimal-edit data made the model both
better *and* safer — exactly the direction the product wants.

### 3.4 A lesson worth keeping — data *style* beats data *volume*

We then tried the obvious next move: add a much larger corpus (Lang-8, an order of
magnitude more data). It made the model **worse** (50.6% → 42.5%). The reason is
instructive: Lang-8's corrections are liberal *fluency rewrites* by many different
people — the opposite of minimal edits — and, being far larger, they swamped the
clean BEA signal and taught the model to over-rewrite. The durable takeaway is that
for a precision-first correction tool, **the style of the training data matters
more than the quantity**. More data of the wrong style is a liability, not an asset.

### 3.5 Under evaluation — a small language model

Because the priority is "make no mistakes," we are also evaluating a small
general-purpose language model (Qwen3-0.6B, ~20× the parameters of our base) on the
hypothesis that a larger, better-calibrated model errs less. Two questions decide
it, and we're measuring both: **throughput** — a 0.6B model decodes token by token
and is a real risk against our CPU throughput target — and **behaviour** — language
models tend to *over-rewrite*, which could *increase* wrong suggestions unless
tightly constrained. We judge it on the same precision-and-throughput lens as
everything else, not on accuracy alone.

## 4. How we make "don't be wrong" a mechanism, not a wish

The robust way to raise precision — and the one that avoids an ever-growing
rulebook — is **confidence-gated abstention**. The model already produces a
probability for each correction; we surface an edit only when that confidence
clears a calibrated threshold, and otherwise leave the text untouched. Because
low-confidence edits are disproportionately the wrong ones, gating removes wrong
suggestions first: precision goes up, coverage goes down — exactly the trade the
mandate asks for. It is a **single tunable dial**, not a list to maintain, and it
works on whichever model we ship.

Two things reinforce it, and neither is a hand-written rule. In training, we use
minimal-edit data and include a portion of *already-correct* examples, so the model
learns *when to stay silent*. In the pipeline, we keep only a small, stable set of
genuine safeguards — never altering emails, URLs, or IDs — and deliberately avoid a
sprawling set of special cases.

## 5. Where we stand

Our best model today is the **BEA-distilled small model**: 50.6% exact-match, ~70%
edit precision, an upper bound of ~34 wrong suggestions over 308 sentences,
appropriately conservative, and within the CPU throughput budget. It is a
**shippable v1** behind the suggestion interface.

The number we optimize from here is the **wrong-suggestion rate**, driven down with
confidence gating (with coverage as the thing we trade away). The highest-value next
steps are: turn on confidence gating; run a **precision spot-check** (human review of
the flagged regressions) to convert the upper bound into a true error rate; finish
the Qwen benchmark; and — the best data we can get — collect **real user
accept/reject signal** once the feature is live, which is perfectly in-domain and
in-style, and is what will move the model further than any public corpus.

## 6. Scoreboard

| Model | Exact-match | Edit precision | Edits made | Wrong-suggestion upper bound |
|---|---|---|---|---|
| Base (visheratin-mini, 31M) | 45.8% | ~56% | 72% | 82 |
| **+ BEA distillation** | **50.6%** | **~70%** | **58%** | **34** |
| + raw Lang-8 | 42.5% | — | — | (regressed) |
| Teacher (t5-base grammar, ~220M) | 74.4% | — | — | too big to ship on CPU |
| Edit-tagging (GECToR, off-the-shelf) | 36.7% | — | — | ~25× slower on CPU |

*(Numbers are on the in-scope grammar+spelling eval, n=308.)*

## 7. Reproducibility

- `bench/eval_dataset.py --model <slot>` — accuracy, split by in-scope vs. out-of-scope.
- `bench/error_profile.py --model <slot> [--dump-regressions out.csv]` — edit
  precision, and the list of away-from-gold edits to review.
- `bench/report.py --models <slot>` — CPU throughput.
- Training/distillation and data prep: `distill/`.

## 8. Caveats we're carrying honestly

- The precision numbers use a single reference, so the wrong-suggestion figure is an
  **upper bound**; the true rate needs the human spot-check.
- **Training-data licensing** (BEA, Lang-8) must be cleared before any model trained
  on them ships commercially — access via a form is not the same as a commercial
  licence.
- Throughput figures depend on the hardware they were measured on; production
  numbers should be re-measured on the target fleet.
