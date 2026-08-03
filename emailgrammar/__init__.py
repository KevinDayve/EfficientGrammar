"""EmailGrammar: CPU-only, LLM-free spelling + grammar correction."""
from .config import CorrectorConfig, PipelineConfig, SpellerConfig
from .corrector import T5Corrector
from .pipeline import Correction, GrammarPipeline, build_pipeline
from .protect import Protector
from .speller import Speller

__all__ = [
    "CorrectorConfig",
    "PipelineConfig",
    "SpellerConfig",
    "T5Corrector",
    "Speller",
    "Protector",
    "GrammarPipeline",
    "Correction",
    "build_pipeline",
]
