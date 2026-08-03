"""Guarded, case- and punctuation-preserving spell correction via SymSpell.

Why not ``lookup_compound``?
    SymSpell's compound lookup lowercases everything, drops most punctuation and
    happily rewrites tokens it does not recognise. For email text that is
    destructive -- it mangles names, ``ACME``, ``s3://bucket``, ``don't`` and so
    on. Instead we do surgical, per-token correction with explicit guard rails
    and only touch tokens we are confident are plain misspelled words.

The speller fixes *non-word* errors (``recieve`` -> ``receive``); real-word and
grammatical errors are left to the downstream T5 model.
"""
from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

from symspellpy import SymSpell, Verbosity

from .config import SpellerConfig


_TOKEN_SPLIT = re.compile(r"(\s+)") # This is a simple whitespace split
_PEEL = re.compile(r"^(?P<pre>[^\w]*)(?P<core>.*?)(?P<post>[^\w]*)$", re.UNICODE) # This regex is used to peel off leading and trailing punctuations, preserving the original word for spell-checking.
_ALPHA_APOS_HYPHEN = re.compile(r"^[A-Za-z]+(?:['\-][A-Za-z]+)*$") # Used to match words that are composed of letters, apostrophes, and hyphens. This regex ensures that the core word is a valid English word without any digits or special characters.


def _bundled_dictionary() -> Path:
    """Path to symspellpy's bundled 82k-word frequency dictionary."""
    return Path(str(files("symspellpy") / "frequency_dictionary_en_82_765.txt"))


def _match_case(original: str, replacement: str) -> str:
    """Project the capitalisation pattern of ``original`` onto ``replacement``."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


class Speller:
    def __init__(self, cfg: SpellerConfig | None = None) -> None:
        self.cfg = cfg or SpellerConfig()
        self.sym = SymSpell(
            max_dictionary_edit_distance=self.cfg.max_edit_distance,
            prefix_length=self.cfg.prefix_length,
        )
        dict_path = self.cfg.dictionary_path or _bundled_dictionary()
        if not self.sym.load_dictionary(str(dict_path), term_index=0, count_index=1):
            raise FileNotFoundError(f"Could not load SymSpell dictionary: {dict_path}")
        for extra in self.cfg.extra_dictionaries:
            self.sym.load_dictionary(str(extra), term_index=0, count_index=1)

    # -- guard rails ---------------------------------------------------------
    def _is_correctable(self, core: str) -> bool:
        """True only for tokens that are safe to treat as plain English words."""
        if len(core) < self.cfg.min_token_len:
            return False
        if not _ALPHA_APOS_HYPHEN.match(core):
            return False  # has digits / @ / _ / other symbols -> leave alone
        if core.isupper() and len(core) > 1:
            # ALL-CAPS is almost always an acronym (NASA/FIFA/RMA). Only a LONG
            # all-caps token can be a shouted misspelling (RECIEVE); short ones
            # stay untouched. The confidence gate still has final say.
            return len(core) >= self.cfg.allcaps_min_len
        if not self.cfg.correct_capitalized and core[:1].isupper():
            return False  # Kevin, Anthropic -> likely proper noun
        return True

    def _high_confidence(self, closest) -> bool:
        """True if the top candidate is safe to AUTO-apply (else: suggest only)."""
        top = closest[0]
        if top.distance > self.cfg.autocorrect_max_distance:
            return False  # raghu->right (2 edits) -> leave, likely a name
        if top.count < self.cfg.min_suggestion_count:
            return False  # target too rare (demra->debra) -> leave
        if len(closest) > 1 and top.count < self.cfg.dominance_ratio * closest[1].count:
            return False  # coin-flip (cip->zip vs cup) -> leave
        return True

    def _closest(self, lower: str):
        return self.sym.lookup(
            lower, Verbosity.CLOSEST,
            max_edit_distance=self.cfg.max_edit_distance, include_unknown=False,
        )

    def _correct_core(self, core: str) -> str:
        lower = core.lower()
        # Already a known word? Do nothing (this also protects real words).
        if self.sym.lookup(lower, Verbosity.TOP, max_edit_distance=0, include_unknown=False):
            return core
        closest = self._closest(lower)
        if not closest or not self._high_confidence(closest):
            return core
        return _match_case(core, closest[0].term)

    # -- public API ----------------------------------------------------------
    def correct(self, text: str) -> str:
        out: list[str] = []
        for chunk in _TOKEN_SPLIT.split(text):
            if not chunk or chunk.isspace():
                out.append(chunk)
                continue
            m = _PEEL.match(chunk)
            pre, core, post = m.group("pre"), m.group("core"), m.group("post")
            if core and self._is_correctable(core):
                core = self._correct_core(core)
            out.append(pre + core + post)
        return "".join(out)

    def correct_batch(self, texts: list[str]) -> list[str]:
        return [self.correct(t) for t in texts]

    # -- dictionary helpers (used by the meaning guard) ----------------------
    def _cores(self, text: str) -> set[str]:
        """Lowercased plain-word tokens in the text."""
        cores = set()
        for chunk in _TOKEN_SPLIT.split(text):
            core = _PEEL.match(chunk).group("core")
            if core and _ALPHA_APOS_HYPHEN.match(core):
                cores.add(core.lower())
        return cores

    def word_cores(self, text: str) -> set[str]:
        return self._cores(text)

    def unknown_words(self, text: str) -> set[str]:
        """Plain words the dictionary doesn't know -- likely names/jargon."""
        return {
            w for w in self._cores(text)
            if len(w) >= self.cfg.min_token_len
            and not self.sym.lookup(w, Verbosity.TOP, max_edit_distance=0, include_unknown=False)
        }

    # -- per-word analysis (for the UI: auto-fix vs. clickable suggestions) ---
    def _analyze_core(self, core: str, n: int = 6) -> dict:
        lower = core.lower()
        if self.sym.lookup(lower, Verbosity.TOP, max_edit_distance=0, include_unknown=False):
            return {"status": "ok"}
        closest = self._closest(lower)
        if not closest:
            return {"status": "unknown"}  # misspelled, but no candidate (rare)
        if self._high_confidence(closest):
            return {"status": "autofix", "fix": _match_case(core, closest[0].term)}
        # Suggestion: widen to ALL edits so useful farther matches (nite -> night,
        # 2 edits) appear, ranked by closeness then frequency.
        allc = self.sym.lookup(
            lower, Verbosity.ALL,
            max_edit_distance=self.cfg.max_edit_distance, include_unknown=False,
        )
        return {"status": "suggest", "candidates": [_match_case(core, c.term) for c in allc[:n]]}

    def analyze(self, text: str) -> list[dict]:
        """Flat token stream that reconstructs ``text`` exactly, each word tagged
        ok | autofix (auto-apply) | suggest (offer candidates) | unknown."""
        tokens: list[dict] = []
        for chunk in _TOKEN_SPLIT.split(text):
            if chunk == "":
                continue
            if chunk.isspace():
                tokens.append({"kind": "space", "raw": chunk})
                continue
            pre, core, post = (_PEEL.match(chunk).group(g) for g in ("pre", "core", "post"))
            tok = {"kind": "word", "pre": pre, "core": core, "post": post, "status": "ok"}
            if core and self._is_correctable(core):
                tok.update(self._analyze_core(core))
            tokens.append(tok)
        return tokens
