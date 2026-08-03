"""Render slide-ready charts from bench/report.py's CSV.

    python bench/plot_report.py                      # reads docs/assets/bench.csv
    -> docs/assets/throughput_vs_batch.png
    -> docs/assets/beam_tradeoff.png

Colours are the validated CVD-safe categorical slots (blue/green); text stays in
ink tokens, grid/axes are recessive, ≥2 series carry a legend + direct labels.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- validated palette (see dataviz skill: palette.md) ---
MINI, TINY = "#2a78d6", "#008300"      # categorical slots 1 & 2 (blue, green)
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
TARGET = "#d03b3b"                      # status: the line to beat


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def load(csv_path: Path):
    rows = list(csv.DictReader(csv_path.open()))
    thr = defaultdict(dict)   # thr[model][batch] = value
    beam = {}                 # beam[beam] = value
    for r in rows:
        if r["metric"] == "throughput":
            thr[r["model"]][int(r["batch"])] = float(r["value"])
        elif r["metric"] == "beam_throughput":
            beam[int(r["beam"])] = float(r["value"])
    return thr, beam


def plot_throughput(thr, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    batches = sorted(next(iter(thr.values())).keys())
    x = list(range(len(batches)))
    for model, color in (("mini", MINI), ("tiny", TINY)):
        y = [thr[model][b] for b in batches]
        ax.plot(x, y, color=color, linewidth=2.4, marker="o", markersize=7,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
                zorder=3, label=model)
        ax.annotate(f"{y[-1]:.0f}", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(8, 0), va="center", color=SECONDARY, fontsize=10, fontweight="bold")

    # 250 req/s target line
    ax.axhline(250, color=TARGET, linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    ax.annotate("250 req/s target", (0, 250), textcoords="offset points",
                xytext=(2, 6), color=TARGET, fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlim(-0.3, len(batches) - 0.3 + 0.6)
    ax.set_ylim(0, 340)
    ax.set_xlabel("batch size", color=SECONDARY, fontsize=11)
    ax.set_ylabel("requests / second", color=SECONDARY, fontsize=11)
    ax.set_title("Throughput scales with batch size  —  4-core CPU, int8",
                 color=INK, fontsize=13.5, fontweight="bold", loc="left", pad=12)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=11,
                    labelcolor=SECONDARY, handlelength=1.4)
    leg.set_title(None)
    fig.text(0.995, 0.01, "reproduce: python bench/report.py && python bench/plot_report.py",
             ha="right", color=MUTED, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def plot_beam(beam, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    beams = sorted(beam.keys())
    x = list(range(len(beams)))
    y = [beam[b] for b in beams]
    knee = beams.index(2)
    ax.bar(x, y, width=0.6, color=MINI, zorder=3)
    for xi, yi in zip(x, y):
        if xi == knee:
            # label inside the bar so the "knee" annotation above has clear space
            ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points", xytext=(0, -18),
                        ha="center", color=SURFACE, fontsize=10.5, fontweight="bold")
        else:
            ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points", xytext=(0, 4),
                        ha="center", color=SECONDARY, fontsize=10.5, fontweight="bold")

    ax.annotate("recommended\n(knee)", (knee, y[knee]), textcoords="offset points",
                xytext=(0, 34), ha="center", color=INK, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))

    ax.set_xticks(x)
    ax.set_xticklabels([f"beam {b}" for b in beams])
    ax.set_ylim(0, max(y) * 1.28)
    ax.set_ylabel("requests / second", color=SECONDARY, fontsize=11)
    ax.set_title("Beam size: quality vs. throughput  —  mini, batch=16",
                 color=INK, fontsize=13.5, fontweight="bold", loc="left", pad=12)
    fig.text(0.995, 0.01, "higher beam = better quality, lower throughput",
             ha="right", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="docs/assets/bench.csv")
    args = ap.parse_args()
    csv_path = Path(args.csv)
    thr, beam = load(csv_path)
    out_dir = csv_path.parent
    plot_throughput(thr, out_dir / "throughput_vs_batch.png")
    plot_beam(beam, out_dir / "beam_tradeoff.png")


if __name__ == "__main__":
    main()
