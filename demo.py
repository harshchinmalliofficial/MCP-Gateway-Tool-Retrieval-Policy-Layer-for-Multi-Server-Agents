"""Quick interactive demo of the gateway proxy.

    python demo.py "reboot EC2 instance i-0abc123"
    python demo.py "page the on-call, checkout is down" --k 3 --llm

Shows the top-k tools FAISS would hand the LLM (instead of all ~160), any
policy flags, and the measured fetch / retrieval cost. With --llm it also
asks the configured provider which tool it would call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from gateway.llm import MissingAPIKeyError, get_provider
from gateway.proxy import build_default_proxy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="the user request")
    ap.add_argument("--k", type=int, default=config.RETRIEVAL_TOP_K)
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--llm", action="store_true", help="also ask the LLM to pick")
    args = ap.parse_args()

    print("building gateway (this embeds ~160 tools once)...")
    proxy = build_default_proxy(include_real=not args.no_real,
                                allow_network=not args.no_real, with_audit=False)
    print(f"catalogue: {len(proxy.catalog.tools)} tools "
          f"{proxy.catalog.counts_by_origin()}")

    prepared = proxy.prepare(args.query, k=args.k)
    print(f"\nquery: {args.query!r}")
    print(f"fetch: {prepared.fetch_seconds * 1000:.1f}ms "
          f"(from_cache={prepared.fetch_from_cache})   "
          f"retrieval: {prepared.retrieval_seconds * 1000:.1f}ms   "
          f"prompt est: {prepared.prompt_token_estimate} tok "
          f"(vs ~{proxy.catalog.tools.__len__() * 55} for the full catalogue)")

    print(f"\ntop-{args.k} tools handed to the LLM:")
    for i, t in enumerate(prepared.tools, 1):
        print(f"  {i}. {t.name:<32} [{t.origin}]  {t.description}")

    if prepared.flags:
        print("\npolicy flags:")
        for f in prepared.flags:
            print(f"  ! {f.tool_name}: {f.kind} - {f.detail}")

    if args.llm:
        try:
            provider = get_provider()
        except MissingAPIKeyError as exc:
            print("\n[--llm] " + str(exc))
            return 0
        sel = provider.select(args.query, prepared.tools)
        tag = " (SIMULATED)" if sel.is_simulated else ""
        print(f"\nLLM ({sel.provider}{tag}) picked: {sel.tool_name}   "
              f"[{sel.total_tokens} tokens, {sel.latency_seconds:.2f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
