# EmailGrammar — Model Training / Fine-Tuning Plan

*Goal: if/when we decide to train, take the grammar model from **45.8%** to beyond
the fine-tuned reference (**74.4%** normalized-match on the in-scope CORE set),
while staying CPU-only, int8, and ≥250 req/s.*

> **Scope update (current):** the product is now **grammar suggestions + spelling**
> only — not a formatter (currency/date/ISBN) or style/name-corrector, and it must
> **never change meaning** (suggestion-based, precision over recall). So the model's
> only job to train is **grammar** (spelling = SymSpell + the suggestion tier).
> Training is currently **deferred** (Kevin's call: the C4_200M base may suffice);
> this plan is the "if we need it" path. Everything below still holds — just read
> "quality" as the **CORE** metric and weight the recipe hard toward **precision**
> (identity data, minimal-edit targets, the meaning-guard as a training signal).

This plan is grounded in the standard high-performing GEC recipe (C4_200M synthetic
pretraining → real fine-tuning, tagged corruption, edit-tagging, distillation) and
targeted at the two concrete failure modes our eval exposed. Citations are to the
canonical works by name; confirm exact numbers/links when the web tools are back up.

---

## 0. Thesis (what actually moves the needle)

GEC quality on a *small* model is **data-bound, not architecture-bound**. The wins,
in priority order:

1. **Scale + distribution-matched synthetic data** (the single biggest lever).
2. **Kill over-correction** — our model is *net-negative on average* (char-sim
   0.887 < 0.893 for doing nothing). Fixable with identity data + minimal-edit
   training + edit-gating.
3. **Right-size the model** to the throughput ceiling (likely T5-efficient-**small**,
   not mini) — or **distill** a big teacher into a tiny student.
4. **Don't make the model do non-grammar work** — formatting (currency/date/ISBN)
   and style are handled by rules/scope decisions, freeing capacity for grammar.

We should expect a well-executed version of this to **roughly match or beat the
fine-tuned reference** and, with distillation + domain data, exceed it — because we
can *shape the training distribution to our eval's categories*, which a generic
fine-tune does not.

---

## 1. Reframe the task: a hybrid system, not one model

Our eval bundles 29 categories that are **not the same kind of problem**. Forcing a
30M model to learn all of them spreads its capacity thin and *causes* over-correction.
Split them:

| Sub-task | Owner | Why |
|---|---|---|
| Spelling (non-word) | **SymSpell** (have it) | already 90% on Basic Typos; free |
| Grammar: agreement, tense, articles, prepositions, word order, word choice | **the trained model** | this is where a seq2seq/edit model earns its keep |
| Punctuation & casing | **the trained model** (in-distribution) | learnable, common |
| Currency / date / number / ISBN / measurement formatting | **deterministic rule module** | regex/`babel`/`price-parser`; a model should *never* learn ISBN formatting |
| Named-entity spelling (Names) | **gazetteer + policy** | reconcile the protect-vs-correct tension per §6 |
| Style: clarity, overused words, informal tone | **out of scope** (defer) | even the fine-tuned reference scores ~0%; not a small-model job |

**Decision to lock before training:** the model's target is
**grammar + punctuation + casing + real-word errors**; formatting → rules; style →
deferred. This alone reclaims ~6–8 eval categories from the model's burden.

---

## 2. The two failure modes we are training against

From [EVAL_RESULTS.md](EVAL_RESULTS.md):

- **Coverage gap** — base models miss the long tail (29% vs 51%). Fix: more, better,
  distribution-matched data.
- **Over-correction** — 209 sentences moved *further* from gold than doing nothing.
  Fix (all three): (a) inject a large fraction of **identity pairs** (correct→correct)
  so the model learns to leave good text alone; (b) train on **minimal-edit** targets
  (BEA-style) rather than fluency rewrites; (c) **edit-gating at inference** — reject a
  rewrite whose model score / edit-confidence is low, or that changes protected spans.

Precision matters more than recall here (F0.5), and for an email product a wrong
"fix" is worse than a miss — the whole recipe is tilted toward precision.

---

## 3. Model track — pick one (or run 1 + 3 in parallel)

| Track | What | Pro | Con | Fit |
|---|---|---|---|---|
| **A. Fine-tune T5-efficient** (seq2seq) | continue-train `visheratin` tiny/mini/**small** | drop-in with our CT2 int8 stack; flexible | autoregressive decode = slower; can hallucinate | **default / lowest-risk** |
| **B. Edit-tagging** (GECToR / seq2edit) | small encoder (ELECTRA/BERT-small) predicts per-token edit tags | **non-autoregressive → far higher throughput**; inherently minimal-edit (less over-correction) | fixed edit vocab; iterative refinement; not CT2 (use ONNX Runtime) | **high-upside for throughput; evaluate** |
| **C. Distillation** | big teacher (the fine-tuned reference, or a larger T5) labels lots of email text; train tiny student on teacher outputs | transfers a strong model into a tiny fast one; no LLM needed (teacher already exists) | needs the teacher + unlabeled email at scale | **strong accelerator; combine with A** |

**Recommendation:** run **A as the backbone** and **layer C (distillation) on top**
(teacher = the existing fine-tuned checkpoint and/or a larger T5 we train first).
**Spike B in parallel** — GECToR-style tagging is the single biggest *throughput*
lever (one forward pass vs token-by-token) and is naturally minimal-edit, which
directly attacks our over-correction problem. If B matches A's quality, it wins on
latency by a wide margin.

**Base-size sweep (Track A):** benchmark `tiny (16M) / mini (31M) / small (60M)` under
int8 + batching and pick the **largest that holds ≥250 req/s** with our batching +
replica plan. Expectation: `small` lands ~100–160 req/s at batch 64 on 4 cores →
viable with more cores/replicas, and buys real quality over mini. Confirm with
`bench/report.py` after converting `small`.

---

## 4. Data strategy (the crux — spend the most effort here)

### 4a. Stage-1 corpus (scale → general GEC ability)
- **C4_200M** (Stahlberg & Kumar 2021) — ~200M synthetic error/correct pairs from
  corrupting clean C4. The workhorse pretraining set for small GEC models. Use a
  few **M** pairs (a small model saturates well before 200M).
- Alternative/supplement: **PIE synthetic**, **WikEd** (Wikipedia revision edits).

### 4b. Stage-2 corpus (real errors → precision)
- **cLang-8** (Rothe et al. 2021, gT5) — cleaned Lang-8, the standard high-quality
  fine-tune set.
- **BEA-2019 train** = **W&I+LOCNESS + FCE + NUCLE + Lang-8** — real learner errors,
  **minimal-edit** (helps over-correction).

### 4c. Domain adaptation (email — the differentiator)
- **Enron Email Corpus** as *clean* targets (real email register, unlike learner
  essays). Corrupt them with the tagged-corruption model (below) → in-domain pairs.
- Any **real Rediff correction logs / accept-reject telemetry** if obtainable — the
  most valuable data of all; even a few thousand pairs for a final polish fine-tune.

### 4d. Tagged corruption — match the error distribution to OUR eval
This is the highest-leverage data technique. Using a **tagged corruption model**
(Stahlberg & Kumar 2021) or **confusion-set/spellcheck noising** (Grundkiewicz et al.
2019), we generate synthetic errors **by ERRANT type** and *control the mix* so it
matches the distribution we care about:
1. Run **ERRANT** on the fine-tuned reference's known errors + any real error sample
   to estimate the target per-type error distribution.
2. Generate synthetic corruptions of clean email/C4 text **weighted to that
   distribution**, over-sampling our weak categories (agreement, articles, word
   order, punctuation, apostrophes, casing).
3. This is how you turn a 29% category into a 70% category — the model sees far more
   of exactly the errors it currently misses.

### 4e. Identity (no-error) data — the over-correction fix
Include **~25–40% identity pairs** (correct→correct), drawn from clean Enron/C4.
Without this, seq2seq models learn to "always change something." This is the most
direct lever on our 209 net-negative sentences.

### 4f. Optional LLM-assisted data (offline only — flag the ethos)
An LLM can (a) generate hard pairs for sparse categories, or (b) corrupt clean email
realistically. This is **training-time only** and does **not** touch the no-LLM
inference constraint. *But* it's a policy call — the fully non-LLM paths above
(public corpora + tagged corruption + distillation from the existing reference model)
are sufficient, so LLM data is an accelerator, not a dependency. **Decision needed.**

### 4g. Hygiene
Dedup (against dev/test!), length-filter to ≤128 tokens (our inference cap),
normalize whitespace, and **hold the 589-row eval set out entirely** — use it as a
diagnostic dev set, never training. Build a **larger held-out in-domain test set**
too; 589 rows is too small to steer on alone.

---

## 5. Training recipe (Track A, concrete starting points)

- **Init:** pretrained `t5-efficient-small` (or mini) — never from scratch.
- **Objective:** seq2seq cross-entropy, teacher forcing; no task prefix (matches
  visheratin); **label smoothing 0.1**.
- **Optimizer:** Adafactor (T5 standard, memory-light), **LR 3e-4–5e-4**, linear or
  constant-with-warmup (~1–2k steps), dropout 0.1.
- **Seq len:** 128 in/out (matches inference; sentence-level units after segmentation).
- **Two stages:**
  1. **Stage 1** — a few M synthetic pairs, 1–3 epochs, large effective batch
     (gradient accumulation). Goal: broad GEC ability.
  2. **Stage 2** — cLang8 + BEA + in-domain (Enron-corrupted) + **identity pairs**,
     1–3 epochs, lower LR (~1e-4). Goal: precision + domain.
  3. **(Optional Stage 3)** — tiny high-quality real Rediff set, 1 epoch, very low LR.
- **Distillation (Track C) overlay:** teacher labels a large batch of unlabeled email;
  add teacher outputs to Stage-2 data (sequence-level KD). Cheap, high transfer.
- **Checkpoint selection:** by **ERRANT F0.5 on BEA-dev** + our in-house per-category
  metric, not by training loss.

**Compute:** a single modern GPU (A100 / L4 / A10 / even T4). Stage 1 on a few M
pairs ≈ hours to ~1 day; Stage 2 ≈ 1–3 hours. Total cloud cost: low tens of dollars.
(This 4-core box can't train — rent a GPU for the training phase only; inference
stays CPU.)

---

## 6. Reconcile the entity/name policy (the design tension)

The eval *wants* `Jonh→John`, `Amazoon→Amazon` fixed; our guards protect them. Don't
solve this with a blanket `correct_capitalized=True` (risks corrupting real names).
Instead:
- **Names gazetteer** (protected-terms dictionary) of known-good names/brands/product
  terms → never corrected; capitalized tokens *not* in the gazetteer become eligible
  for correction (so genuine misspellings like `Amazoon` get fixed, `Rediff` doesn't).
- **Keep hard-entity masking** (emails/URLs/ids) exactly as is.
- Let the **trained model** learn casing/name-spelling from data where it's safe.
This is a per-category policy, decided with the product owner, not a global switch.

---

## 7. Evaluation & iteration loop (already have the harness)

- **In-house:** `bench/eval_dataset.py` — normalized-match + **per-category** table +
  the "further-from-gold" over-correction counter. Our steering signal.
- **Standard:** add **ERRANT F0.5** (BEA-dev) and **GLEU** (JFLEG) harnesses for
  comparability and to watch precision/recall + meaning-changing edits.
- **Loop:** train → score per-category → find the weakest categories → **re-weight the
  synthetic distribution** toward them → retrain. 2–4 iterations typically capture most
  of the gain.
- **Guardrail metric:** track precision and over-correction rate explicitly; a model
  that gains recall by rewriting more is a regression for us.

---

## 8. Quantize & ship (closing the loop with our stack)

1. Convert the HF checkpoint → CTranslate2 **int8** with our `scripts/convert_model.py`.
2. **Measure quantization loss** (int8 can cost ~0.5–1 pt F0.5) — if material, try
   `int8_float32` or `int16`, or quantization-aware fine-tuning.
3. Re-run `bench/report.py` (throughput) + `bench/eval_dataset.py` (quality) on the
   quantized model — ship only if it holds ≥250 req/s *and* the quality target.
4. Regression-gate with `scripts/gen_examples.py`.

---

## 9. Phased execution plan

| Phase | Work | Output | ~Effort |
|---|---|---|---|
| **P0** Scope & harness | lock target categories; add ERRANT+GLEU to the eval harness; freeze baselines | agreed scope, metric suite | 0.5 day |
| **P1** Data pipeline | assemble C4_200M/cLang8/BEA + Enron; build tagged-corruption generator matched to our distribution; add identity pairs; dedup/split | reproducible dataset builder | 2–4 days |
| **P2** Model sweep | convert+bench tiny/mini/small for throughput; pick base | throughput-viable base choice | 0.5 day |
| **P3** Train (Track A) | Stage-1 synthetic → Stage-2 real+domain+identity; checkpoint on F0.5 | trained HF checkpoint | 2–3 days + GPU |
| **P4** Distill (Track C) | teacher-label email; sequence-KD into student | improved student | 1–2 days |
| **P5** Spike (Track B) | GECToR/seq2edit prototype; compare quality+latency to A | go/no-go on edit-tagging | 2–3 days (parallel) |
| **P6** Quantize+eval loop | CT2 int8; per-category error analysis; re-weight; iterate | shippable model + numbers | 2–3 days |
| **P7** Hybrid glue | rule-based formatter (currency/date/ISBN); names gazetteer; edit-gating | full system | 2 days |

Critical path P0→P1→P3→P6 is ~1.5–2 weeks to a first strong model; P4/P5/P7 layer in.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Small model can't reach the bar | distillation from a stronger teacher; step up to `small`; edit-tagging (Track B) |
| Over-correction persists | more identity data; minimal-edit targets; inference edit-gating; F0.5-select |
| Domain mismatch (learner vs email) | Enron-corrupted domain data; final Rediff-log polish |
| int8 quality drop | int8_float32 / int16 / QAT; measure before shipping |
| Overfitting to the 589-set | keep it dev-only; build a larger in-domain test set |
| Data licensing (C4/Lang-8/Enron) | verify licenses before shipping model weights; Enron is public, C4_200M is released for research |

---

## 11. Success criteria

- **Primary:** in-house normalized-match **> 51%** (beat the fine-tuned reference),
  with **avg char-sim to gold > raw input** (i.e., net-positive — the current
  regression eliminated) and over-correction count sharply down.
- **Per-category:** no core-grammar category (agreement, tense, articles, punctuation,
  apostrophe, word order) below ~60%.
- **Throughput:** hold **≥250 req/s** int8 on the target hardware.
- **Standard-benchmark sanity:** report ERRANT F0.5 (BEA) + GLEU (JFLEG) so quality
  generalizes beyond the in-house set.

---

### Key references (confirm links when web is back)
Stahlberg & Kumar 2021 (*Synthetic Data with Tagged Corruption Models*, C4_200M) ·
Rothe et al. 2021 (*A Simple Recipe for Multilingual GEC*, cLang8/gT5) ·
Omelianchuk et al. 2020 (*GECToR — Tag, Not Rewrite*) ·
Grundkiewicz et al. 2019 (*Neural GEC with synthetic/spellcheck noise*) ·
Kiyono et al. 2019 (*pretraining with pseudo data*) · BEA-2019 shared task ·
Bryant et al. 2017 (ERRANT) · Napoles et al. 2015 (GLEU).
