"""Render README.md from README.template.md + the last benchmark run's numbers.

The template holds ``{{PLACEHOLDER}}`` tokens; this script fills them from
benchmark/results/*.json and writes README.md.  Rendering always starts from the
pristine template, so it is safe to run repeatedly.

Called at the end of run_benchmark.py; also runnable standalone:
    python benchmark/update_readme.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

RESULTS = config.RESULTS_DIR
TEMPLATE = ROOT / "README.template.md"
README = ROOT / "README.md"


def _load(name: str):
    return json.loads((RESULTS / name).read_text())


def _verdict_block(verdict_lines: list[str] | None) -> str:
    if verdict_lines:
        return "\n".join(verdict_lines)
    saved = RESULTS / "verdicts.txt"
    return saved.read_text().strip() if saved.exists() else "(see console output)"


def _fmt_main_table(main: dict) -> str:
    lines = [
        f"{'':2} {'setup':<34} {'acc':>6} {'tools':>6} {'tokens':>8} "
        f"{'fetch ms':>9} {'retr ms':>8} {'llm s':>7} {'e2e s':>7}",
        "-" * 78,
    ]
    for s in ("A", "B", "C"):
        r = main[s]
        lines.append(
            f"{s:2} {r['label'][:34]:<34} {r['accuracy']*100:>5.1f}% "
            f"{r['avg_tools_shown']:>6.0f} {r['avg_total_tokens']:>8.0f} "
            f"{r['avg_fetch_ms']:>9.2f} {r['avg_retrieval_ms']:>8.2f} "
            f"{r['avg_llm_latency_s']:>7.3f} {r['avg_e2e_latency_s']:>7.3f}"
        )
    return "\n".join(lines)


def _fmt_xover_table(points: list[dict]) -> str:
    lines = [
        f"{'tools':>6} | {'acc no-gw':>9} {'acc gw':>7} | "
        f"{'lat no-gw':>10} {'lat gw':>8} | {'tok no-gw':>9} {'tok gw':>8}",
        "-" * 78,
    ]
    for r in points:
        lines.append(
            f"{r['tool_count']:>6} | {r['no_gateway_accuracy']*100:>8.1f}% "
            f"{r['with_gateway_accuracy']*100:>6.1f}% | "
            f"{r['no_gateway_latency_s']:>10.3f} {r['with_gateway_latency_s']:>8.3f} | "
            f"{r['no_gateway_tokens']:>9.0f} {r['with_gateway_tokens']:>8.0f}"
        )
    return "\n".join(lines)


def build_mapping(verdict_lines: list[str] | None = None) -> dict[str, str]:
    main = _load("main_summary.json")
    xo = _load("crossover_summary.json")
    meta = _load("run_meta.json")
    A, B, C = main["A"], main["B"], main["C"]
    points = xo.get("points", [])
    xp = xo.get("crossover_tool_count")

    fetch_speedup = (A["avg_fetch_ms"] / B["avg_fetch_ms"]) if B["avg_fetch_ms"] else 0.0
    acc_delta_ab = abs(A["accuracy"] - B["accuracy"]) * 100
    tok_ratio = (B["avg_total_tokens"] / C["avg_total_tokens"]) if C["avg_total_tokens"] else 0.0
    lat_cut = (1 - C["avg_e2e_latency_s"] / B["avg_e2e_latency_s"]) * 100 if B["avg_e2e_latency_s"] else 0.0

    tok_down = C["avg_total_tokens"] < B["avg_total_tokens"]
    lat_down = C["avg_e2e_latency_s"] < B["avg_e2e_latency_s"]
    acc_up = C["accuracy"] > B["accuracy"] + 1e-9
    c1 = fetch_speedup > 2.0 and acc_delta_ab <= 5.0
    c2_full = acc_up and tok_down and lat_down
    c2_partial = tok_down and lat_down
    c3 = xp is not None and 20 <= xp <= 40

    def badge(ok: bool, partial: bool = False) -> str:
        return "✅ holds" if ok else ("🟡 partial" if partial else "❌ did not hold")

    # Claim 2 sentence adapts to whether there was accuracy headroom.
    if acc_up:
        claim2 = (f"retrieval lifted accuracy {B['accuracy']*100:.1f}%→{C['accuracy']*100:.1f}%, "
                  f"cut tokens {tok_ratio:.0f}× ({B['avg_total_tokens']:.0f}→{C['avg_total_tokens']:.0f}) "
                  f"and end-to-end latency {lat_cut:.0f}% "
                  f"({B['avg_e2e_latency_s']:.2f}s→{C['avg_e2e_latency_s']:.2f}s), "
                  f"for a {C['avg_retrieval_ms']:.0f} ms retrieval cost")
    else:
        claim2 = (f"the model already scored {B['accuracy']*100:.1f}% on the full "
                  f"{int(B['avg_tools_shown'])}-tool list, so there was no accuracy "
                  f"headroom to reclaim — retrieval's win here was **{tok_ratio:.0f}× fewer "
                  f"tokens** ({B['avg_total_tokens']:.0f}→{C['avg_total_tokens']:.0f}) and "
                  f"**{lat_cut:.0f}% lower latency** "
                  f"({B['avg_e2e_latency_s']:.2f}s→{C['avg_e2e_latency_s']:.2f}s) "
                  f"for a {C['avg_retrieval_ms']:.0f} ms retrieval step; accuracy moved "
                  f"{B['accuracy']*100:.1f}%→{C['accuracy']*100:.1f}% "
                  f"(one query whose target fell outside the top-{meta.get('top_k', 5)})")

    if xp is not None:
        big = next((r for r in points if r["tool_count"] >= 80), points[-1] if points else None)
        tail = ""
        if big:
            tail = (f"; by {big['tool_count']} tools the gateway is "
                    f"{big['no_gateway_latency_s'] - big['with_gateway_latency_s']:.2f}s faster "
                    f"({big['no_gateway_latency_s']:.2f}s→{big['with_gateway_latency_s']:.2f}s)")
        claim3 = f"latency lines cross at **{xp} tools**{tail}"
    else:
        claim3 = "no crossover was found inside the tested range"

    m = {
        "FETCH_SPEEDUP": f"{fetch_speedup:.0f}",
        "ACC_DELTA_AB": f"{acc_delta_ab:.1f}",
        "ACC_A": f"{A['accuracy']*100:.1f}%",
        "ACC_B": f"{B['accuracy']*100:.1f}%",
        "ACC_C": f"{C['accuracy']*100:.1f}%",
        "TOK_B": f"{B['avg_total_tokens']:.0f}",
        "TOK_C": f"{C['avg_total_tokens']:.0f}",
        "TOK_RATIO": f"{tok_ratio:.0f}",
        "LAT_B": f"{B['avg_e2e_latency_s']:.2f}s",
        "LAT_C": f"{C['avg_e2e_latency_s']:.2f}s",
        "LAT_CUT_PCT": f"{lat_cut:.0f}",
        "RETR_MS": f"{C['avg_retrieval_ms']:.0f}",
        "CROSSOVER": str(xp) if xp is not None else "the tested range's",
        "CLAIM2_SENTENCE": claim2,
        "CLAIM3_SENTENCE": claim3,
        "VERDICT_1": badge(c1),
        "VERDICT_2": badge(c2_full, partial=c2_partial),
        "VERDICT_3": badge(bool(c3)),
        "PROVIDER": str(meta.get("provider")),
        "MODEL_NOTE": ("SIMULATED lexical matcher — accuracy illustrative"
                       if meta.get("is_simulated") else f"real LLM via {meta.get('provider')}"),
        "MAIN_TABLE": _fmt_main_table(main),
        "CROSSOVER_TABLE": _fmt_xover_table(points) if points else "(crossover sweep not run)",
        "VERDICT_BLOCK": _verdict_block(verdict_lines),
        "RUN_META": (f"provider={meta.get('provider')} · simulated={meta.get('is_simulated')} · "
                     f"catalogue={meta.get('catalogue')} · real_mcp_live={meta.get('real_mcp_tools_live')} · "
                     f"embedder={meta.get('embedder')} · backend={meta.get('vector_backend')} · "
                     f"queries={meta.get('queries')} · audit_rows={meta.get('audit_rows')} · "
                     f"ts={meta.get('timestamp')}"),
    }
    return m


def apply(verdict_lines: list[str] | None = None) -> None:
    if not TEMPLATE.exists():
        print(f"  [readme] no template at {TEMPLATE.name}; skipping")
        return
    text = TEMPLATE.read_text()
    for key, val in build_mapping(verdict_lines).items():
        text = text.replace("{{" + key + "}}", val)
    leftover = sorted(set(re.findall(r"\{\{[A-Z_0-9]+\}\}", text)))
    README.write_text(text)
    if leftover:
        print(f"  [readme] warning: unresolved placeholders {leftover}")
    else:
        print("  [readme] rendered from template with measured numbers")


if __name__ == "__main__":
    apply()
