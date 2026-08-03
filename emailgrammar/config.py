"""Central configuration for EmailGrammar.

All tunables live here so the engine, conversion scripts, and benchmarks stay in
sync. Paths can be overridden with the ``EG_DATA_DIR`` environment variable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EG_DATA_DIR", ROOT / "data"))
MODELS_DIR = DATA_DIR / "models"
DICT_DIR = DATA_DIR / "dictionaries"

# --- Models (MIT-licensed, production-safe) --------------------------------
# visheratin/* are T5-efficient checkpoints: tiny (~16M) and mini (~31M).
# They were trained on a single task, so NO task prefix is prepended.
HF_MODELS = {
    "mini": "visheratin/t5-efficient-mini-grammar-correction",
    "tiny": "visheratin/t5-efficient-tiny-grammar-correction",
}
DEFAULT_MODEL = "mini"
GRAMMAR_PREFIX = ""  # these checkpoints need no "grammar:" style prefix


def ct2_dir_for(model_key: str, quantization: str) -> Path:
    """Canonical on-disk location for a converted CTranslate2 model."""
    return MODELS_DIR / f"t5-efficient-{model_key}-ct2-{quantization}"


@dataclass
class CorrectorConfig:
    """Settings for the CTranslate2 T5 grammar corrector."""

    model_key: str = DEFAULT_MODEL
    quantization: str = "int8"          # int8 | int8_float32 | float32
    device: str = "cpu"
    compute_type: str = "int8"          # how weights are held at runtime
    intra_threads: int = 0              # 0 => CTranslate2 picks (uses all cores)
    inter_threads: int = 1              # parallel replicas for concurrent throughput
    beam_size: int = 1                  # 1 = greedy (fastest); >1 trades speed for quality
    max_input_length: int = 128
    max_decoding_length: int = 128
    ct2_dir: Path | None = None         # defaults to ct2_dir_for(model_key, quantization)

    def __post_init__(self) -> None:
        if self.ct2_dir is None:
            self.ct2_dir = ct2_dir_for(self.model_key, self.quantization)
        self.ct2_dir = Path(self.ct2_dir)

    @property
    def hf_model(self) -> str:
        return HF_MODELS[self.model_key]


@dataclass
class SpellerConfig:
    """Settings for the guarded SymSpell speller.

    Defaults are intentionally conservative: emails contain names, product
    terms, addresses and URLs we must never "correct". Auto-correction only fires
    when it is *high-confidence* (common target word + one clear winner).
    """

    max_edit_distance: int = 2           # max single-char edits from typo to word
    prefix_length: int = 7
    min_token_len: int = 3               # skip very short tokens ("hi", "ok")
    # Correct Capitalized tokens too (Rkght->Right). Safe now: the confidence
    # gate below already protects real names/brands (Kevin, Anthropic, Raghu).
    correct_capitalized: bool = True
    # --- high-confidence gate (keeps demra/cip/raghu/names untouched) ---
    autocorrect_max_distance: int = 1        # only fix 1-edit typos (raghu->right is 2)
    min_suggestion_count: int = 10_000_000   # only auto-fix TO a common word
    # Auto-fix only when the top answer UTTERLY dominates (teh->the). Ambiguous
    # words with several plausible neighbours (nite->site/note/night) fall through
    # to the suggestion tier instead of being silently mis-corrected.
    dominance_ratio: float = 20.0
    allcaps_min_len: int = 6                 # leave short ALL-CAPS (NASA/FIFA/RMA)
    dictionary_path: Path | None = None      # None => symspellpy's bundled 82k dict
    # Any words to protect (domain terms, product names, shorthand) live in a
    # data file here -- not hardcoded in the library.
    extra_dictionaries: list[Path] = field(default_factory=list)


@dataclass
class PipelineConfig:
    use_speller: bool = True
    use_protector: bool = True          # mask emails/URLs/ids from T5, restore after
    meaning_guard: bool = True          # reject grammar edits that change meaning
    min_meaning_sim: float = 0.6        # below this similarity = a rewrite, reject
    corrector: CorrectorConfig = field(default_factory=CorrectorConfig)
    speller: SpellerConfig = field(default_factory=SpellerConfig)
