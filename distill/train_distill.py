"""T5 -> T5 knowledge distillation for EmailGrammar.

    Teacher : vennify/t5-base-grammar-correction  (~220M, too big to ship)
    Student : visheratin/t5-efficient-mini (~31M, warm-started at ~45.8% CORE)
              (swap to google/t5-efficient-small for more capacity if throughput allows)

Loss = alpha * KL(student || teacher, temperature)   <- the "dark knowledge"
     + (1 - alpha) * CrossEntropy(student, gold)      <- the hard target

Both models are T5 and SHARE the SentencePiece vocab (32k), so per-position logits
align under teacher forcing -> full logit-level KL works (not just sequence-level).

>>> STATUS: written on a CPU box, NOT yet run on GPU. Monday, before the full run:
    1. `python train_distill.py --smoke` (tiny batch) to shake out shapes/devices.
    2. confirm the assert below (teacher/student vocab identical) passes.
    3. then launch the real run.

Run:
    python train_distill.py \
        --train data/train.tsv --eval data/dev.tsv \
        --student visheratin/t5-efficient-mini-grammar-correction \
        --teacher vennify/t5-base-grammar-correction \
        --out runs/distill-mini --epochs 3 --alpha 0.7 --temperature 2.0

Data format: a TSV per line -> `source<TAB>target` (erroneous \t corrected).
Get it from BEA-2019 / cLang8 (public), or generate with the teacher. See README.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

MAX_LEN = 128


def read_tsv(path: str) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in csv.reader(f, delimiter="\t"):
            if len(line) >= 2 and line[0].strip() and line[1].strip():
                rows.append({"src": line[0], "tgt": line[1]})
    return Dataset.from_list(rows)


def tokenize_fn(tok):
    def _f(batch):
        model_in = tok(batch["src"], max_length=MAX_LEN, truncation=True)
        labels = tok(text_target=batch["tgt"], max_length=MAX_LEN, truncation=True)
        model_in["labels"] = labels["input_ids"]
        return model_in
    return _f


class DistillTrainer(Trainer):
    def __init__(self, teacher, alpha, temperature, **kw):
        super().__init__(**kw)
        self.teacher = teacher.to(self.args.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.alpha = alpha
        self.T = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs["labels"]
        out = model(**inputs)                       # student: CE loss + logits
        ce = out.loss
        with torch.no_grad():
            t_logits = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,                      # same teacher forcing -> aligned
            ).logits
        # KL over vocab, temperature-scaled, masked to non-pad target positions.
        s_logp = F.log_softmax(out.logits / self.T, dim=-1)
        t_prob = F.softmax(t_logits / self.T, dim=-1)
        kl = F.kl_div(s_logp, t_prob, reduction="none").sum(-1)   # [B, L]
        mask = (labels != -100).float()
        kl = (kl * mask).sum() / mask.sum().clamp(min=1) * (self.T ** 2)
        loss = self.alpha * kl + (1 - self.alpha) * ce
        return (loss, out) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=False, default="data/train.tsv")
    ap.add_argument("--eval", default=None)
    ap.add_argument("--student", default="visheratin/t5-efficient-mini-grammar-correction")
    ap.add_argument("--teacher", default="vennify/t5-base-grammar-correction")
    ap.add_argument("--out", default="runs/distill-mini")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.7, help="weight on KL vs CE")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check shapes")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.student)
    student = T5ForConditionalGeneration.from_pretrained(args.student)
    teacher = T5ForConditionalGeneration.from_pretrained(args.teacher)

    # Logit-KL only valid if the two share a vocab. If this fails, distill
    # sequence-level instead (train on teacher-generated targets + CE only).
    assert student.config.vocab_size == teacher.config.vocab_size, (
        f"vocab mismatch: student {student.config.vocab_size} vs teacher "
        f"{teacher.config.vocab_size} -> use sequence-level KD, not logit-KL"
    )

    train_ds = read_tsv(args.train).map(tokenize_fn(tok), batched=True,
                                        remove_columns=["src", "tgt"])
    eval_ds = (read_tsv(args.eval).map(tokenize_fn(tok), batched=True,
                                       remove_columns=["src", "tgt"])
               if args.eval else None)
    if args.smoke:
        train_ds = train_ds.select(range(min(64, len(train_ds))))

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=1 if args.smoke else args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch",
        logging_steps=50,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = DistillTrainer(
        teacher=teacher, alpha=args.alpha, temperature=args.temperature,
        model=student, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=student),
    )
    trainer.train()
    student.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved distilled student -> {args.out}")
    print("Next: convert to CTranslate2 int8 via EmailGrammar/scripts/convert_model.py, "
          "then benchmark with bench/eval_dataset.py (CORE) and bench/report.py (throughput).")


if __name__ == "__main__":
    main()
