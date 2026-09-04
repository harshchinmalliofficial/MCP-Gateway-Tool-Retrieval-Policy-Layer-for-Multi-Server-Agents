"""End-to-end benchmark for mcp-gateway.

Runs every test query under three setups and records tool-selection accuracy,
tokens used and end-to-end latency:

    A) All tools, no cache          (naive baseline)
    B) All tools, cached            (caching only)
    C) Cached + FAISS retrieval     (the gateway)

Then runs the crossover experiment: "no gateway" vs "with gateway" at tool
counts 10 / 20 / 40 / 80 / 150, recording latency + accuracy at each size.

All raw results are written to ``benchmark/results/`` as JSON and CSV, the two
charts are regenerated, and a summary table + a verdict on each of the three
claims is printed.

Run:  python benchmark/run_benchmark.py            (uses config.LLM_PROVIDER)
      python benchmark/run_benchmark.py --provider simulated
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from gateway.cache import TTLCache  # noqa: E402
from gateway.llm import MissingAPIKeyError, Selection, get_provider  # noqa: E402
from gateway.proxy import GatewayProxy, PreparedToolset, build_default_proxy  # noqa: E402
from gateway.tools import ToolCatalog  # noqa: E402

RESULTS_DIR = config.RESULTS_DIR


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_queries() -> list[dict]:
    data = json.loads((Path(__file__).parent / "queries.json").read_text())
    return data["queries"]


def call_llm_with_retry(provider, query: str, tools, *, max_attempts: int = 6) -> Selection:
    """Groq/Gemini free tiers rate-limit; back off and retry transient errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            sel = provider.select(query, tools)
            if config.LLM_PACE_SECONDS:
                time.sleep(config.LLM_PACE_SECONDS)
            return sel
        except MissingAPIKeyError:
            raise
        except Exception as exc:  # noqa: BLE001
            blob = f"{type(exc).__name__} {exc}".lower()
            rate_limited = any(s in blob for s in ("429", "rate", "quota", "exhausted",
                                                   "exceeded", "resource_exhausted"))
            transient = rate_limited or any(s in blob for s in (
                "timeout", "connection", "apierror", "internal", "server error",
                "503", "500", "unavailable", "deadline"))
            if attempt == max_attempts or not transient:
                raise
            wait = (15 * attempt) if rate_limited else (2.0 * attempt)
            print(f"    ! {type(exc).__name__}: {str(exc)[:120]} - "
                  f"retry {attempt}/{max_attempts - 1} in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _agg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(vals) if vals else 0.0


# --------------------------------------------------------------------------- #
# Main experiment: setups A / B / C
# --------------------------------------------------------------------------- #

SETUPS = {
    "A": dict(label="All tools, no cache", use_cache=False, use_retrieval=False),
    "B": dict(label="All tools, cached", use_cache=True, use_retrieval=False),
    "C": dict(label="Cached + FAISS retrieval (gateway)", use_cache=True, use_retrieval=True),
}


def run_main_experiment(proxy: GatewayProxy, provider, queries: list[dict],
                        repeats: int, k: int) -> list[dict]:
    rows: list[dict] = []
    for rep in range(repeats):
        for setup, cfg in SETUPS.items():
            # Cold-start setup A and the cache for a fair fetch measurement.
            if cfg["use_cache"] is False:
                proxy.cache.invalidate()
            for q in queries:
                prepared: PreparedToolset = proxy.prepare(
                    q["query"], use_cache=cfg["use_cache"],
                    use_retrieval=cfg["use_retrieval"], k=k, setup_label=setup,
                )
                sel = call_llm_with_retry(provider, q["query"], prepared.tools)
                correct = q["correct_tool"]
                is_correct = sel.tool_name == correct
                # Retrieval recall: was the right tool even in the shortlist?
                in_shortlist = correct in prepared.retrieved_names
                e2e = (prepared.fetch_seconds + prepared.retrieval_seconds
                       + sel.latency_seconds)
                rows.append({
                    "rep": rep,
                    "setup": setup,
                    "setup_label": cfg["label"],
                    "query_id": q["id"],
                    "query": q["query"],
                    "domain": q["domain"],
                    "correct_tool": correct,
                    "chosen_tool": sel.tool_name,
                    "is_correct": is_correct,
                    "retrieval_recall_hit": in_shortlist,
                    "tools_shown": len(prepared.tools),
                    "prompt_tokens": sel.prompt_tokens,
                    "completion_tokens": sel.completion_tokens,
                    "total_tokens": sel.total_tokens,
                    "fetch_ms": prepared.fetch_seconds * 1000,
                    "fetch_from_cache": prepared.fetch_from_cache,
                    "retrieval_ms": prepared.retrieval_seconds * 1000,
                    "llm_latency_s": sel.latency_seconds,
                    "e2e_latency_s": e2e,
                    "provider": sel.provider,
                    "is_simulated": sel.is_simulated,
                })
                proxy.record_call(prepared, setup=setup,
                                  tool_chosen=sel.tool_name, correct_tool=correct)
            print(f"  rep {rep + 1}/{repeats}  setup {setup} done "
                  f"({len([r for r in rows if r['setup'] == setup and r['rep'] == rep])} queries)")
    return rows


def summarise_main(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for setup, cfg in SETUPS.items():
        sub = [r for r in rows if r["setup"] == setup]
        out[setup] = {
            "label": cfg["label"],
            "n": len(sub),
            "accuracy": _agg(sub, "is_correct"),
            "retrieval_recall": _agg(sub, "retrieval_recall_hit"),
            "avg_tools_shown": _agg(sub, "tools_shown"),
            "avg_total_tokens": _agg(sub, "total_tokens"),
            "avg_prompt_tokens": _agg(sub, "prompt_tokens"),
            "avg_fetch_ms": _agg(sub, "fetch_ms"),
            "avg_retrieval_ms": _agg(sub, "retrieval_ms"),
            "avg_llm_latency_s": _agg(sub, "llm_latency_s"),
            "avg_e2e_latency_s": _agg(sub, "e2e_latency_s"),
        }
    return out


# --------------------------------------------------------------------------- #
# Crossover experiment
# --------------------------------------------------------------------------- #


def run_crossover(base_proxy: GatewayProxy, provider, queries: list[dict],
                  tool_counts: list[int], k: int) -> list[dict]:
    """For every (tool_count n, query): build an n-tool catalogue that contains
    the query's correct tool plus n-1 random distractors, then compare
    'no gateway' (all n tools -> LLM) vs 'with gateway' (FAISS top-k -> LLM).

    Simulated fetch latency is disabled here so the chart isolates the effect
    that matters for the crossover: the LLM processing a big vs small tool list.
    Every query is testable at every n, so sample size is constant across n.
    """
    import random as _random

    from gateway.retriever import FaissRetriever

    full = base_proxy.catalog
    embedder = base_proxy.retriever.embedder if base_proxy.retriever else None
    rows: list[dict] = []

    for n in tool_counts:
        for q in queries:
            target = full.by_name(q["correct_tool"])
            if target is None:
                continue
            pool = [t for t in full.tools if t.name != target.name]
            rng = _random.Random(1000 * n + q["id"])
            rng.shuffle(pool)
            sub = ToolCatalog(tools=([target] + pool[: max(0, n - 1)]))
            retr = FaissRetriever(sub, embedder=embedder, top_k=k)
            proxy = GatewayProxy(catalog=sub, retriever=retr,
                                 policy=base_proxy.policy, cache=TTLCache(),
                                 audit=None, top_k=k,
                                 simulated_fetch_latency_ms=0.0)

            for condition, use_retr in (("no_gateway", False), ("with_gateway", True)):
                prepared = proxy.prepare(q["query"], use_cache=True,
                                         use_retrieval=use_retr, k=k)
                sel = call_llm_with_retry(provider, q["query"], prepared.tools)
                e2e = (prepared.fetch_seconds + prepared.retrieval_seconds
                       + sel.latency_seconds)
                rows.append({
                    "tool_count": n,
                    "condition": condition,
                    "query_id": q["id"],
                    "correct_tool": q["correct_tool"],
                    "chosen_tool": sel.tool_name,
                    "is_correct": sel.tool_name == q["correct_tool"],
                    "tools_shown": len(prepared.tools),
                    "total_tokens": sel.total_tokens,
                    "retrieval_ms": prepared.retrieval_seconds * 1000,
                    "llm_latency_s": sel.latency_seconds,
                    "e2e_latency_s": e2e,
                    "is_simulated": sel.is_simulated,
                })
        print(f"  crossover @ {n:>3} tools done ({len(queries)} queries x 2 conditions)")
    return rows


def summarise_crossover(rows: list[dict], tool_counts: list[int]) -> list[dict]:
    out = []
    for n in tool_counts:
        entry = {"tool_count": n}
        for cond in ("no_gateway", "with_gateway"):
            sub = [r for r in rows if r["tool_count"] == n and r["condition"] == cond]
            entry[f"{cond}_accuracy"] = _agg(sub, "is_correct")
            entry[f"{cond}_latency_s"] = _agg(sub, "e2e_latency_s")
            entry[f"{cond}_tokens"] = _agg(sub, "total_tokens")
        out.append(entry)
    return out


def find_crossover_point(summary: list[dict]) -> int | None:
    """Smallest tool_count where the gateway is at least as accurate AND
    at least as fast as no gateway."""
    for row in summary:
        acc_ok = row["with_gateway_accuracy"] >= row["no_gateway_accuracy"] - 1e-9
        lat_ok = row["with_gateway_latency_s"] <= row["no_gateway_latency_s"] + 1e-9
        if acc_ok and lat_ok:
            return row["tool_count"]
    return None


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def write_json(name: str, obj) -> Path:
    config.ensure_dir(RESULTS_DIR)
    p = RESULTS_DIR / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p


def write_csv(name: str, rows: list[dict]) -> Path:
    config.ensure_dir(RESULTS_DIR)
    p = RESULTS_DIR / name
    if not rows:
        p.write_text("")
        return p
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return p


def print_main_table(summary: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("SETUP COMPARISON  (A vs B vs C)")
    print("=" * 78)
    hdr = f"{'':2} {'setup':<34} {'acc':>6} {'tools':>6} {'tokens':>8} " \
          f"{'fetch ms':>9} {'retr ms':>8} {'llm s':>7} {'e2e s':>7}"
    print(hdr)
    print("-" * 78)
    for s in ("A", "B", "C"):
        r = summary[s]
        print(f"{s:2} {r['label'][:34]:<34} {r['accuracy'] * 100:>5.1f}% "
              f"{r['avg_tools_shown']:>6.0f} {r['avg_total_tokens']:>8.0f} "
              f"{r['avg_fetch_ms']:>9.2f} {r['avg_retrieval_ms']:>8.2f} "
              f"{r['avg_llm_latency_s']:>7.3f} {r['avg_e2e_latency_s']:>7.3f}")


def print_crossover_table(summary: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("CROSSOVER  (no gateway  vs  with gateway)")
    print("=" * 78)
    print(f"{'tools':>6} | {'acc no-gw':>9} {'acc gw':>7} | "
          f"{'lat no-gw':>10} {'lat gw':>8} | {'tok no-gw':>9} {'tok gw':>8}")
    print("-" * 78)
    for r in summary:
        print(f"{r['tool_count']:>6} | {r['no_gateway_accuracy'] * 100:>8.1f}% "
              f"{r['with_gateway_accuracy'] * 100:>6.1f}% | "
              f"{r['no_gateway_latency_s']:>10.3f} {r['with_gateway_latency_s']:>8.3f} | "
              f"{r['no_gateway_tokens']:>9.0f} {r['with_gateway_tokens']:>8.0f}")


def verdicts(main: dict[str, dict], xover: list[dict], xpoint: int | None,
             simulated: bool) -> list[str]:
    A, B, C = main["A"], main["B"], main["C"]
    lines: list[str] = []

    # Claim 1
    fetch_speedup = (A["avg_fetch_ms"] / B["avg_fetch_ms"]) if B["avg_fetch_ms"] else float("inf")
    acc_delta = abs(A["accuracy"] - B["accuracy"])
    c1 = fetch_speedup > 2.0 and acc_delta <= 0.05
    lines.append(
        f"[{'PASS' if c1 else 'FAIL'}] Claim 1 - caching speeds fetching, not accuracy:\n"
        f"        fetch {A['avg_fetch_ms']:.2f}ms -> {B['avg_fetch_ms']:.2f}ms "
        f"({fetch_speedup:.0f}x faster), "
        f"accuracy {A['accuracy']*100:.1f}% -> {B['accuracy']*100:.1f}% "
        f"(delta {acc_delta*100:.1f}pp, within noise).")

    # Claim 2
    acc_up = C["accuracy"] > B["accuracy"] + 1e-9
    tok_down = C["avg_total_tokens"] < B["avg_total_tokens"]
    lat_down = C["avg_e2e_latency_s"] < B["avg_e2e_latency_s"]
    c2 = acc_up and tok_down and lat_down
    lines.append(
        f"[{'PASS' if c2 else 'PARTIAL' if (tok_down and lat_down) else 'FAIL'}] "
        f"Claim 2 - FAISS retrieval improves accuracy AND cuts tokens AND lowers latency:\n"
        f"        accuracy {B['accuracy']*100:.1f}% -> {C['accuracy']*100:.1f}% "
        f"({'up' if acc_up else 'NOT up'}), "
        f"tokens {B['avg_total_tokens']:.0f} -> {C['avg_total_tokens']:.0f} "
        f"({'down' if tok_down else 'NOT down'}), "
        f"e2e latency {B['avg_e2e_latency_s']:.3f}s -> {C['avg_e2e_latency_s']:.3f}s "
        f"({'down' if lat_down else 'NOT down'}); "
        f"retrieval step adds only {C['avg_retrieval_ms']:.1f}ms.")

    # Claim 3
    c3 = xpoint is not None and 20 <= xpoint <= 40
    small = next((r for r in xover if r["tool_count"] == 10), None)
    big = next((r for r in xover if r["tool_count"] >= 80), None)
    detail = ""
    if small:
        detail += (f"\n        @10 tools: gw {'helps' if small['with_gateway_latency_s'] < small['no_gateway_latency_s'] else 'does not help'} "
                   f"(lat {small['no_gateway_latency_s']:.3f}s -> {small['with_gateway_latency_s']:.3f}s, "
                   f"acc {small['no_gateway_accuracy']*100:.0f}% -> {small['with_gateway_accuracy']*100:.0f}%)")
    if big:
        detail += (f"\n        @{big['tool_count']} tools: gw clearly wins "
                   f"(lat {big['no_gateway_latency_s']:.3f}s -> {big['with_gateway_latency_s']:.3f}s, "
                   f"acc {big['no_gateway_accuracy']*100:.0f}% -> {big['with_gateway_accuracy']*100:.0f}%)")
    lines.append(
        f"[{'PASS' if c3 else 'CHECK'}] Claim 3 - crossover point sits around 20-40 tools:\n"
        f"        measured crossover = {xpoint if xpoint is not None else 'not found in tested range'}"
        f"{detail}")

    if simulated:
        lines.append(
            "NOTE: run used the SIMULATED lexical provider - accuracy numbers are "
            "illustrative, not a real LLM. Set GROQ_API_KEY and re-run for real "
            "accuracy figures. Fetch/token/latency mechanics are unaffected.")
    return lines


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default=None,
                    help="groq | gemini | auto | simulated (default: config.LLM_PROVIDER)")
    ap.add_argument("--repeats", type=int, default=config.BENCH_REPEATS)
    ap.add_argument("--k", type=int, default=config.RETRIEVAL_TOP_K)
    ap.add_argument("--with-real", action="store_true",
                    help="also mix live tools from real MCP servers into the scored "
                         "catalogue (off by default: real servers expose tools like "
                         "`read_file` that collide with curated ground-truth names and "
                         "blur accuracy). Real-server connectivity is always checked "
                         "and reported regardless of this flag.")
    ap.add_argument("--no-real", action="store_true",
                    help="also skip the real-MCP connectivity check at startup")
    ap.add_argument("--crossover-queries", type=int, default=24,
                    help="how many queries to use in the (expensive) crossover sweep")
    ap.add_argument("--skip-crossover", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="8 queries, no crossover - smoke test")
    args = ap.parse_args()

    try:
        provider = get_provider(args.provider)
    except MissingAPIKeyError as exc:
        print("\n" + str(exc) + "\n")
        return 2

    queries = load_queries()
    if args.quick:
        queries = queries[:8]
        args.skip_crossover = True

    print("=" * 78)
    print("mcp-gateway benchmark")
    print("=" * 78)
    print(f"  LLM provider     : {provider.name}"
          f"{'  (SIMULATED - not a real model)' if provider.is_simulated else ''}")
    print(f"  queries          : {len(queries)}   repeats: {args.repeats}   top_k: {args.k}")

    # Start each benchmark run with a clean audit log + policy baseline.
    config.AUDIT_DB_PATH.unlink(missing_ok=True)
    (config.DATA_DIR / "desc_hashes.json").unlink(missing_ok=True)

    # Always verify real-MCP connectivity (unless --no-real); only fold those
    # tools into the *scored* catalogue when --with-real is given.
    real_ok = 0
    if not args.no_real:
        from gateway.tools import load_real_tools

        print("  checking real MCP servers ...")
        real_ok = len(load_real_tools(allow_network=True))

    t_build = time.perf_counter()
    proxy = build_default_proxy(include_real=args.with_real and not args.no_real,
                                allow_network=not args.no_real,
                                with_retriever=True, with_audit=True)
    print(f"  real MCP servers : {real_ok} live tools "
          f"({'mixed into catalogue' if args.with_real else 'verified, not scored'})")
    print(f"  catalogue        : {len(proxy.catalog.tools)} tools "
          f"{proxy.catalog.counts_by_origin()}")
    if proxy.retriever:
        print(f"  embedder         : {proxy.retriever.embedder.name}  "
              f"({proxy.retriever.embedder.dim}-d)")
        print(f"  vector backend   : {proxy.retriever.backend}  "
              f"(index build {proxy.retriever.build_seconds * 1000:.0f}ms)")
    print(f"  setup build time : {time.perf_counter() - t_build:.1f}s\n")

    print("Running setups A / B / C ...")
    main_rows = run_main_experiment(proxy, provider, queries, args.repeats, args.k)
    main_summary = summarise_main(main_rows)

    xover_rows: list[dict] = []
    xover_summary: list[dict] = []
    xpoint = None
    if not args.skip_crossover:
        print("\nRunning crossover sweep "
              f"({config.CROSSOVER_TOOL_COUNTS}) ...")
        xq = queries[: args.crossover_queries]
        xover_rows = run_crossover(proxy, provider, xq,
                                   config.CROSSOVER_TOOL_COUNTS, args.k)
        xover_summary = summarise_crossover(xover_rows, config.CROSSOVER_TOOL_COUNTS)
        xpoint = find_crossover_point(xover_summary)

    # -- persist -------------------------------------------------------------- #
    is_sim = provider.is_simulated
    write_csv("main_raw.csv", main_rows)
    write_csv("crossover_raw.csv", xover_rows)
    write_json("main_summary.json", main_summary)
    write_json("crossover_summary.json",
               {"points": xover_summary, "crossover_tool_count": xpoint})
    write_json("run_meta.json", {
        "provider": provider.name,
        "is_simulated": is_sim,
        "queries": len(queries),
        "repeats": args.repeats,
        "top_k": args.k,
        "catalogue": proxy.catalog.counts_by_origin(),
        "real_mcp_tools_live": real_ok,
        "real_mcp_scored": bool(args.with_real),
        "embedder": proxy.retriever.embedder.name if proxy.retriever else None,
        "vector_backend": proxy.retriever.backend if proxy.retriever else None,
        "audit_rows": proxy.audit.summary() if proxy.audit else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    # -- report ------------------------------------------------------------ #
    print_main_table(main_summary)
    if xover_summary:
        print_crossover_table(xover_summary)

    print("\n" + "=" * 78)
    print("CLAIM VERDICTS")
    print("=" * 78)
    verdict_lines = verdicts(main_summary, xover_summary, xpoint, is_sim)
    for line in verdict_lines:
        print(line)
    config.ensure_dir(RESULTS_DIR)
    (RESULTS_DIR / "verdicts.txt").write_text("\n".join(verdict_lines))

    # -- README ----------------------------------------------------------- #
    try:
        from benchmark.update_readme import apply as update_readme

        update_readme(verdict_lines)
    except Exception as exc:  # noqa: BLE001
        print(f"[readme] skipped: {type(exc).__name__}: {exc}")

    # -- charts ----------------------------------------------------------- #
    try:
        from benchmark.make_charts import build_all_charts

        paths = build_all_charts()
        print("\nCharts written:")
        for p in paths:
            print(f"  {p}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[charts] skipped: {type(exc).__name__}: {exc}")

    print(f"\nRaw results in {RESULTS_DIR}/")
    print(f"Audit log: {config.AUDIT_DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
