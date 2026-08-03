"""Generate docs/EXAMPLES.md from live pipeline output (nothing hand-faked).

    python scripts/gen_examples.py

Each example is run through the recommended config (mini, beam=2, protector on)
and grouped by category. "Limitation" rows are included on purpose -- honesty in
a demo beats cherry-picking.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emailgrammar.pipeline import build_pipeline  # noqa: E402

# (category, note, input)
EXAMPLES = [
    ("Spelling (non-word typos)", "classic misspellings the speller fixes", "i recieved you're seperate messege yesterdey"),
    ("Spelling (non-word typos)", "transpositions + doubled letters", "teh committe will discus it tommorow"),
    ("Grammar (agreement / tense)", "subject-verb agreement", "the team have been working hardly on it"),
    ("Grammar (agreement / tense)", "articles + preposition", "please find attach report which you was requesting"),
    ("Grammar (agreement / tense)", "double negative + tense", "she dont know if the client will excepts our proposal"),
    ("Spelling + grammar together", "the common email case", "i has recieve you're emails yesterday and wil responsd son"),
    ("Spelling + grammar together", "run-on-ish", "their are alot of thing we need too discus before the meating"),
    ("Entity protection: email", "address must stay byte-exact", "pls send the file to john.doe@rediff.com by eod"),
    ("Entity protection: URL", "link preserved", "the details is on https://rediff.com/offers/2026 kindly chek"),
    ("Entity protection: order id", "hyphenated id preserved", "my order id ORD-99213 hasnt arrive yet"),
    ("Entity protection: amount + time", "money and time preserved", "the amt is Rs.4500 and meeting is at 3pm sharp"),
    ("Entity protection: handle + domain", "mention + bare domain preserved", "cc rahul@team.io and visit team.io for detail"),
    ("Entity protection: multi-entity", "3 entities recovered by position, grammar still applied", "email me at a.b@x.com or visit https://x.com/h before 5pm"),
    ("Casing / position", "stray full-stop merge, case kept", "That. Works."),
    ("Limitation: model capacity", "31M model gets the meaning wrong", "he dont have no time for finishing this projet by tommorow"),
    ("Limitation: model capacity", "'am gone' is locally valid -> no error signal (see overview 7.1)", "I am gone insane"),
    ("Limitation: speller is context-blind", "'cup' becomes 'zip' (freq-ranked, no context)", "cant wait for the wirld cip finsl next weak"),
    ("Limitation: lowercase name", "not regex-detectable -> can be mangled", "please give demra and raghu my best regard"),
    ("Limitation: shouted typo", "all-caps guarded -> speller leaves it for T5", "this is REALY URGENT pls RESPOND"),
    ("Limitation: chat abbreviation", "'pls' wrongly -> 'plus' (needs abbrev dict)", "pls revert asap thx"),
]


def main() -> None:
    pipe = build_pipeline(model="mini", beam_size=2)  # protector on by default
    lines = [
        "# EmailGrammar — Illustrative Examples",
        "",
        "Live output from the recommended config: **model=mini, beam=2, speller on, "
        "entity-protection on**. Regenerate with `python scripts/gen_examples.py`.",
        "",
        "Rows tagged **Limitation** are included deliberately — they map to the "
        "roadmap items in the overview doc.",
        "",
    ]
    current = None
    n = 0
    for cat, note, inp in EXAMPLES:
        if cat != current:
            lines += ["", f"## {cat}", ""]
            current = cat
        d = pipe.correct(inp, detailed=True)
        n += 1
        flag = "  _(fell back — entity kept safe, grammar skipped)_" if d.fell_back else ""
        lines += [
            f"**{n}. {note}**",
            "",
            f"- in : `{d.original}`",
            f"- out: `{d.final}`{flag}",
            "",
        ]
    out = Path(__file__).resolve().parent.parent / "docs" / "EXAMPLES.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {n} examples -> {out}")


if __name__ == "__main__":
    main()
