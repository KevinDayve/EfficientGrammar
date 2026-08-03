"""T5 grammar correction via CTranslate2 (CPU, int8).

The heavy lifting of throughput lives here:
  * weights are int8-quantized -> ~2-4x faster on CPU, ~4x smaller,
  * ``translate_batch`` runs the encoder/decoder for many inputs at once,
  * CTranslate2 releases the GIL, so a single Translator can be driven from
    multiple Python threads (``inter_threads`` replicas) for concurrency.

Torch is deliberately NOT imported here -- runtime needs only ctranslate2 +
the sentencepiece tokenizer.
"""
from __future__ import annotations

from transformers import AutoTokenizer

import ctranslate2

from .config import CorrectorConfig


class T5Corrector:
    def __init__(self, cfg: CorrectorConfig | None = None) -> None:
        self.cfg = cfg or CorrectorConfig()
        model_dir = str(self.cfg.ct2_dir)
        # Tokenizer is saved alongside the converted model so runtime never
        # needs the HuggingFace hub.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.translator = ctranslate2.Translator(
            model_dir,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
            inter_threads=self.cfg.inter_threads,
            intra_threads=self.cfg.intra_threads,
        )

    def _encode(self, text: str) -> list[str]:
        ids = self.tokenizer.encode(
            text, truncation=True, max_length=self.cfg.max_input_length
        )
        return self.tokenizer.convert_ids_to_tokens(ids)

    def _decode(self, tokens: list[str]) -> str:
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def correct_batch(self, texts: list[str], max_batch_size: int = 0) -> list[str]:
        if not texts:
            return []
        source = [self._encode(t) for t in texts]
        results = self.translator.translate_batch(
            source,
            beam_size=self.cfg.beam_size,
            max_decoding_length=self.cfg.max_decoding_length,
            max_batch_size=max_batch_size,
        )
        return [self._decode(r.hypotheses[0]) for r in results]

    def correct(self, text: str) -> str:
        return self.correct_batch([text])[0]
