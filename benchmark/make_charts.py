"""Render the two benchmark charts as PNGs into benchmark/results/.

Chart 1 - setups_accuracy_tokens.png : A/B/C, accuracy and tokens (two panels,
          never a dual axis).
Chart 2 - crossover.png              : latency vs tool count, no-gateway vs
          gateway, with an accuracy panel beneath.

Reads the JSON summaries written by run_benchmark.py, so it can be re-run
standalone to regenerate charts without re-running the benchmark.

Palette + mark choices follow the data-viz method: fixed-order categorical hues
(validated colorblind-safe), thin marks, recessive axes/grid, a legend for >=2
series, selective direct labels, one y-axis per panel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402

RESULTS_DIR = config.RESULTS_DIR

# --- validated categorical hues (reference palette, light surface) ----------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e4e1"
BLUE = "#2a78d6"     # slot 1  -> "all tools" / "no gateway"
ORANGE = "#eb6834"   # slot 2  -> "gateway (top-k)" / "with gateway"
XLINE = "#8a8a86"    # neutral crossover marker

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 11,
    "font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
})


def _load(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text())


def _sim_suffix() -> str:
    try:
        if _load("run_meta.json").get("is_simulated"):
            return "   — SIMULATED provider, accuracy illustrative"
    except Exception:  # noqa: BLE001
        pass
    return ""


def _provider_note() -> str:
    try:
        m = _load("run_meta.json")
        return f"provider: {m.get('provider')} · {m.get('embedder', '')} · {m.get('queries')} queries"
    except Exception:  # noqa: BLE001
        return ""


def _despine(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


# --------------------------------------------------------------------------- #
# Chart 1: A / B / C
# --------------------------------------------------------------------------- #


def chart_setups() -> Path:
    s = _load("main_summary.json")
    keys = ["A", "B", "C"]
    short = {"A": "A\nall tools\nno cache",
             "B": "B\nall tools\ncached",
             "C": "C\ncached +\nFAISS (gateway)"}
    xs = [short[k] for k in keys]
    acc = [s[k]["accuracy"] * 100 for k in keys]
    tok = [s[k]["avg_total_tokens"] for k in keys]
    e2e = [s[k]["avg_e2e_latency_s"] for k in keys]
    # A & B share the "all tools" identity; C is the gateway.
    colors = [BLUE, BLUE, ORANGE]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7))
    panels = [
        (axes[0], acc, "tool-selection accuracy (%)", "{:.1f}%", (0, 108)),
        (axes[1], tok, "tokens sent to the LLM  (avg / query)", "{:,.0f}", None),
        (axes[2], e2e, "end-to-end latency  (s / query)", "{:.2f}s", None),
    ]
    for ax, vals, ylabel, fmt, ylim in panels:
        bars = ax.bar(xs, vals, color=colors, width=0.62, zorder=3)
        ax.set_ylabel(ylabel, fontsize=10)
        top = ylim[1] if ylim else max(vals) * 1.18
        ax.set_ylim(ylim or (0, top))
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + top * 0.015,
                    fmt.format(v), ha="center", va="bottom", fontsize=9.5, color=INK)
        _despine(ax)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, labelsize=8.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["all tools (A, B)", "gateway top-k (C)"],
               loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("A/B/C — caching (B) only speeds fetching; retrieval (C) cuts tokens & latency"
                 + _sim_suffix(), fontsize=12.5)
    fig.text(0.5, 0.925, _provider_note(), ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    out = RESULTS_DIR / "setups_accuracy_tokens.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Chart 2: crossover
# --------------------------------------------------------------------------- #


def chart_crossover() -> Path:
    data = _load("crossover_summary.json")
    pts = data.get("points") or []
    xp = data.get("crossover_tool_count")
    if len(pts) < 2:
        fig, ax = plt.subplots(figsize=(8.8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "crossover sweep not run\n(run: python benchmark/run_benchmark.py)",
                ha="center", va="center", fontsize=12, color=INK2)
        out = RESULTS_DIR / "crossover.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return out
    xs = [p["tool_count"] for p in pts]

    lat_no = [p["no_gateway_latency_s"] for p in pts]
    lat_gw = [p["with_gateway_latency_s"] for p in pts]
    acc_no = [p["no_gateway_accuracy"] * 100 for p in pts]
    acc_gw = [p["with_gateway_accuracy"] * 100 for p in pts]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.1, 1]})

    def _line(ax, ys, color, label):
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=7, zorder=3,
                label=label, clip_on=False)

    _line(ax1, lat_no, BLUE, "no gateway  (all tools → LLM)")
    _line(ax1, lat_gw, ORANGE, "with gateway  (FAISS top-k → LLM)")
    ax1.set_ylabel("end-to-end latency (s)")
    ax1.set_ylim(0, max(lat_no + lat_gw) * 1.32)
    ax1.legend(frameon=False, fontsize=9, loc="upper left", ncol=1)

    _line(ax2, acc_no, BLUE, "no gateway")
    _line(ax2, acc_gw, ORANGE, "with gateway")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_xlabel("number of tools available to the agent")
    ax2.set_ylim(0, 118)
    ax2.set_yticks(range(0, 101, 25))

    # endpoint labels for the latency lines only (accuracy is flat — noted in text)
    for ys, color in ((lat_no, BLUE), (lat_gw, ORANGE)):
        ax1.annotate(f"{ys[-1]:.2f}s", (xs[-1], ys[-1]),
                     textcoords="offset points", xytext=(6, 0),
                     va="center", fontsize=9, color=color)
    if acc_no == acc_gw:
        ax2.text(0.5, 0.5, f"both conditions: {acc_no[0]:.0f}% at every tool count\n"
                 "(this model is not confused by extra tools — no accuracy headroom)",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=9, color=INK2)
    else:
        for ys, color in ((acc_no, BLUE), (acc_gw, ORANGE)):
            ax2.annotate(f"{ys[-1]:.0f}%", (xs[-1], ys[-1]),
                         textcoords="offset points", xytext=(6, 0),
                         va="center", fontsize=9, color=color)

    for ax in (ax1, ax2):
        _despine(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(xs)
        if xp:
            ax.axvline(xp, color=XLINE, ls=(0, (4, 3)), lw=1.1, zorder=1)
    if xp:
        ax1.annotate(f"crossover ≈ {xp} tools", (xp, ax1.get_ylim()[1] * 0.52),
                     textcoords="offset points", xytext=(7, 0),
                     fontsize=9, color=XLINE, va="center", rotation=90)

    ax1.set_title("Crossover: end-to-end latency vs number of tools available"
                  + _sim_suffix(), fontsize=12.5)
    fig.text(0.5, 0.02, _provider_note(), ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    out = RESULTS_DIR / "crossover.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def build_all_charts() -> list[Path]:
    config.ensure_dir(RESULTS_DIR)
    return [chart_setups(), chart_crossover()]


if __name__ == "__main__":
    for p in build_all_charts():
        print("wrote", p)
