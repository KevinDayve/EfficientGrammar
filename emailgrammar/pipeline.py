"""End-to-end grammar correction pipeline.

    raw text
      --> [Speller: fix non-word typos]        (guarded; entities untouched)
      --> [Protector: mask emails/URLs/ids]     (-> __0__ sentinels)
      --> [T5-CT2: fix grammar]                 (batched; the throughput stage)
      --> [Protector: restore, or fall back]    (never emit a corrupted entity)
      --> output

Every stage is optional/swappable. The speller + protector run per-text
(microseconds); the corrector batches, which is where throughput comes from.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import PipelineConfig
from .corrector import T5Corrector
from .meaning import preserves_meaning
from .protect import Protector
from .speller import Speller


@dataclass
class Correction:
    original: str
    spell_corrected: str
    masked: str
    final: str
    fell_back: bool = False       # a sentinel didn't survive -> spell-only output
    meaning_guarded: bool = False  # grammar edit rejected -> spell-only output

    @property
    def changed(self) -> bool:
        return self.final != self.original


class GrammarPipeline:
    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg = cfg or PipelineConfig()
        self.speller = Speller(self.cfg.speller) if self.cfg.use_speller else None
        self.protector = Protector(enabled=self.cfg.use_protector)
        self.corrector = T5Corrector(self.cfg.corrector)

    def _spell(self, texts: list[str]) -> list[str]:
        if self.speller is None:
            return list(texts)
        return self.speller.correct_batch(texts)

    def correct_batch(
        self, texts: list[str], *, detailed: bool = False, max_batch_size: int = 0
    ):
        spelled = self._spell(texts)

        masked_spans = [self.protector.mask(s) for s in spelled]
        masked = [m for m, _ in masked_spans]

        corrected = self.corrector.correct_batch(masked, max_batch_size=max_batch_size)

        final: list[str] = []
        fell_back: list[bool] = []
        guarded: list[bool] = []
        for corr, (_, spans), spell in zip(corrected, masked_spans, spelled):
            restored, ok = self.protector.unmask(corr, spans)
            # Safety net 1: a lost sentinel means an entity could be corrupted, so
            # emit the (entity-safe) spell-only text instead of a mangled result.
            cand = restored if ok else spell
            # Safety net 2: never change meaning. Drop the grammar edit (keep
            # spelling only) if it (a) deletes/alters an unrecognized word -- a
            # likely name, e.g. demra -> me -- or (b) flips a negation / rewrites
            # wholesale.
            was_guarded = False
            if self.cfg.meaning_guard and cand != spell:
                dropped_name = (
                    self.speller is not None
                    and bool(self.speller.unknown_words(spell) - self.speller.word_cores(cand))
                )
                if dropped_name or not preserves_meaning(spell, cand, self.cfg.min_meaning_sim):
                    cand, was_guarded = spell, True
            final.append(cand)
            fell_back.append(not ok)
            guarded.append(was_guarded)

        if not detailed:
            return final
        return [
            Correction(original=o, spell_corrected=s, masked=m, final=f,
                       fell_back=fb, meaning_guarded=mg)
            for o, s, m, f, fb, mg in zip(texts, spelled, masked, final, fell_back, guarded)
        ]

    def correct(self, text: str, *, detailed: bool = False):
        return self.correct_batch([text], detailed=detailed)[0]


def build_pipeline(
    model: str = "mini",
    *,
    quantization: str = "int8",
    use_speller: bool = True,
    use_protector: bool = True,
    **corrector_overrides,
) -> GrammarPipeline:
    """Convenience factory used by the CLI, benchmark and (later) the server."""
    from .config import CorrectorConfig, SpellerConfig

    corrector = CorrectorConfig(
        model_key=model, quantization=quantization, **corrector_overrides
    )
    return GrammarPipeline(
        PipelineConfig(
            use_speller=use_speller,
            use_protector=use_protector,
            corrector=corrector,
            speller=SpellerConfig(),
        )
    )
