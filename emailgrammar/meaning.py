"""Cheap "did the meaning change?" checks for the grammar model's output.

The grammar model occasionally flips a negation ("don't know" -> "does know") or
rewrites a sentence wholesale ("give demra my best" -> "give me a break"). Neither
is acceptable -- the tool fixes *form*, never *meaning*. These are plain string
checks (no model), used to REJECT an unsafe grammar edit so the pipeline falls
back to leaving the sentence as-is (spelling only).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

# Words that carry negation. Contractions are matched with and without the
# apostrophe, so "don't" and "dont" both count.
_NEG_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing", "nowhere", "neither",
    "nor", "without", "cannot",
}
_NEG_CONTRACTIONS = {
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "wont",
    "cant", "couldnt", "shouldnt", "wouldnt", "hasnt", "havent", "hadnt",
    "aint", "mustnt", "neednt", "wouldve",  # (wouldve harmless; keeps set simple)
}


def _negation_count(text: str) -> int:
    t = re.sub(r"['’`]", "", text.lower())          # don't -> dont
    return sum(
        1 for w in re.findall(r"[a-z]+", t)
        if w in _NEG_WORDS or w in _NEG_CONTRACTIONS
    )


def preserves_meaning(before: str, after: str, min_sim: float = 0.6) -> bool:
    """False if the grammar edit looks like it changed meaning."""
    # 1) A negation added/removed flips meaning.
    if _negation_count(before) != _negation_count(after):
        return False
    # 2) A wholesale rewrite (little of the original left) is not a grammar fix.
    if SequenceMatcher(None, before.lower(), after.lower()).ratio() < min_sim:
        return False
    return True
