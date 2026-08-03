"""Entity protection via placeholder masking.

Why masking (and not constrained decoding)?
    CTranslate2 has no positive "must-include" lexical constraint, and constrained
    decoding -- which *forces* terms to appear -- is both absent from CT2 and slow
    (a non-starter at 250 req/s). Placeholder masking (Crego et al. 2016; Post,
    ACL W19-6618) is the production-standard alternative: model-agnostic, zero
    throughput cost, no retraining. Swap each protectable span for a sentinel the
    model copies verbatim, run T5, swap the originals back.

Sentinel choice:
    Empirically the visheratin T5 reliably copies only ``__N__`` style tokens;
    alphabetic/bracket/unicode sentinels get mangled (``ENT0`` -> ``ENT 1`` etc).
    See the sentinel round-trip experiment. We therefore use ``__0__``, ``__1__``.

Safety net:
    Even ``__N__`` can theoretically be altered in some context, so we VERIFY every
    sentinel survived. If one did not, ``unmask`` reports failure and the pipeline
    falls back to the un-corrected (spell-only) text for that item -- we would
    rather skip grammar correction than emit a corrupted email address or order id.

Scope:
    This protects *structured*, regex-detectable spans (emails, URLs, @handles,
    #tags, domains, and any token containing a digit: ids, times, amounts, phone
    numbers, versions). It does NOT protect free-text lowercase names (e.g. a
    person called "demra") -- those are a named-entity problem handled elsewhere
    (capitalisation guard / protected-terms dictionary / optional NER).
"""
from __future__ import annotations

import re

_WS = re.compile(r"(\s+)")
_LEAD = set("([{\"'“‘")            # opening brackets/quotes
_TRAIL = set(")]}\"'”’.,;:!?")     # closing brackets/quotes + sentence punct

_SENTINEL = "__{}__"
_SENTINEL_RE = re.compile(r"__(\d+)__")
# A "placeholder-ish" run for positional recovery: the model sometimes keeps the
# sentinel's position but eats its digit (__0__ -> ____). Any maximal [_\d] run
# with >=2 underscores counts; a bare number (no underscore) never does, so real
# numbers left in the text are not mistaken for sentinels.
_RUN_RE = re.compile(r"[_\d]+")
# A bare domain like "rediff.com" / "team.io" / "node.js": >=2 labels of >=2
# chars each, TLD-ish final label. The >=2 length rule excludes "e.g." / "a.m.".
_DOMAIN_RE = re.compile(r"^(?:[A-Za-z0-9-]{2,}\.)+[A-Za-z]{2,6}$")


def _looks_structured(core: str) -> bool:
    """True for tokens we must hand to T5 unchanged."""
    if not core:
        return False
    if "@" in core or core[0] == "#":
        return True                                   # email / mention / hashtag
    low = core.lower()
    if low.startswith("http") or low.startswith("www.") or "://" in core:
        return True                                   # URL
    if "/" in core:
        return True                                   # path / fraction / s3://-style
    if any(ch.isdigit() for ch in core):
        return True                                   # id / time / amount / version
    if _DOMAIN_RE.match(core):
        return True                                   # bare domain
    return False


def _peel(token: str) -> tuple[str, str, str]:
    """Split surrounding brackets/punctuation off a token: (pre, core, post)."""
    i, j = 0, len(token)
    while i < j and token[i] in _LEAD:
        i += 1
    while j > i and token[j - 1] in _TRAIL:
        j -= 1
    return token[:i], token[i:j], token[j:]


class Protector:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def mask(self, text: str) -> tuple[str, list[str]]:
        """Return (masked_text, spans). spans[i] is the original for ``__i__``."""
        if not self.enabled:
            return text, []
        spans: list[str] = []
        out: list[str] = []
        for chunk in _WS.split(text):
            if not chunk or chunk.isspace():
                out.append(chunk)
                continue
            pre, core, post = _peel(chunk)
            if _looks_structured(core):
                out.append(pre + _SENTINEL.format(len(spans)) + post)
                spans.append(core)
            else:
                out.append(chunk)
        return "".join(out), spans

    def unmask(self, text: str, spans: list[str]) -> tuple[str, bool]:
        """Restore originals. Returns (restored_text, ok).

        Three tiers, safest-first:
          1. every sentinel survived verbatim -> restore by index;
          2. positional recovery -> the model kept the sentinels' *positions* but
             mangled them (e.g. ate the digit: ``__0__`` -> ``____``); if the count
             of placeholder-ish runs still matches, restore by order;
          3. give up (``ok=False``) so the caller falls back to entity-safe text
             rather than emit a corrupted address/id.
        """
        if not spans:
            return text, True

        # Tier 1: exact.
        if all(_SENTINEL.format(i) in text for i in range(len(spans))):
            def repl(m: re.Match) -> str:
                idx = int(m.group(1))
                return spans[idx] if 0 <= idx < len(spans) else m.group(0)

            return _SENTINEL_RE.sub(repl, text), True

        # Tier 2: positional recovery.
        runs = [m for m in _RUN_RE.finditer(text) if m.group().count("_") >= 2]
        if len(runs) == len(spans):
            pieces: list[str] = []
            last = 0
            for span, m in zip(spans, runs):
                pieces.append(text[last:m.start()])
                pieces.append(span)
                last = m.end()
            pieces.append(text[last:])
            return "".join(pieces), True

        # Tier 3: unsafe -> caller falls back.
        return text, False
