"""The gateway proxy: cache + retriever + policy, wired together.

``GatewayProxy.prepare(query)`` is the whole value proposition in one call:

    raw MCP servers ->  [cache]  ->  full tool catalogue
                                 ->  [FAISS retriever]  -> top-k candidates
                                 ->  [policy]           -> filtered + flagged
                                 ->  the small tool set handed to the LLM

It returns a :class:`PreparedToolset` with the tools *and* the measurements
(fetch time, retrieval time, token estimate, flags) the benchmark needs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import config
from gateway.cache import TTLCache
from gateway.llm import _build_user_prompt, _rough_token_estimate
from gateway.policy import AuditLog, Flag, PolicyEngine
from gateway.retriever import FaissRetriever
from gateway.tools import Tool, ToolCatalog, build_catalog

CATALOG_CACHE_KEY = "mcp:tool-catalogue"


@dataclass
class PreparedToolset:
    query: str
    tools: list[Tool]                 # what the LLM should see
    retrieved_names: list[str]
    flags: list[Flag]
    fetch_seconds: float
    fetch_from_cache: bool
    retrieval_seconds: float
    prompt_token_estimate: int        # tokens if you sent `tools` to the LLM
    full_catalogue_size: int
    used_retrieval: bool

    @property
    def flagged(self) -> bool:
        return bool(self.flags)


@dataclass
class GatewayProxy:
    catalog: ToolCatalog
    retriever: FaissRetriever | None = None      # None => retrieval disabled
    policy: PolicyEngine = field(default_factory=PolicyEngine)
    cache: TTLCache = field(default_factory=TTLCache)
    audit: AuditLog | None = None
    top_k: int = config.RETRIEVAL_TOP_K
    # Simulated cold-fetch latency per server so cache vs no-cache is visible
    # even when the real MCP servers are local/fast.
    simulated_fetch_latency_ms: float = config.SIMULATED_FETCH_LATENCY_MS

    # -- internals ------------------------------------------------------------ #
    def _load_catalogue(self) -> list[Tool]:
        """The 'expensive' fetch we cache. Adds a simulated per-server delay to
        stand in for real network round-trips to remote MCP servers."""
        servers = {t.server for t in self.catalog.tools}
        time.sleep(len(servers) * self.simulated_fetch_latency_ms / 1000.0)
        return list(self.catalog.tools)

    def fetch_catalogue(self, use_cache: bool) -> tuple[list[Tool], float, bool]:
        if not use_cache:
            start = time.perf_counter()
            tools = self._load_catalogue()
            return tools, time.perf_counter() - start, False
        res = self.cache.fetch(CATALOG_CACHE_KEY, self._load_catalogue)
        return res.value, res.fetch_seconds, res.from_cache

    # -- main API ---------------------------------------------------------- #
    def prepare(self, query: str, *, use_cache: bool = True,
                use_retrieval: bool = True, k: int | None = None,
                setup_label: str = "C") -> PreparedToolset:
        tools, fetch_s, from_cache = self.fetch_catalogue(use_cache)

        retrieval_s = 0.0
        if use_retrieval and self.retriever is not None:
            hits, retrieval_s = self.retriever.retrieve(query, k or self.top_k)
            candidates = [h.tool for h in hits]
        else:
            candidates = list(tools)

        report = self.policy.evaluate(candidates, persist_hashes=False)
        final_tools = report.allowed
        flags = report.all_flags

        token_est = _rough_token_estimate(_build_user_prompt(query, final_tools))

        prepared = PreparedToolset(
            query=query,
            tools=final_tools,
            retrieved_names=[t.name for t in final_tools],
            flags=flags,
            fetch_seconds=fetch_s,
            fetch_from_cache=from_cache,
            retrieval_seconds=retrieval_s,
            prompt_token_estimate=token_est,
            full_catalogue_size=len(tools),
            used_retrieval=bool(use_retrieval and self.retriever is not None),
        )
        return prepared

    def record_call(self, prepared: PreparedToolset, *, setup: str,
                    tool_chosen: str | None, correct_tool: str | None) -> None:
        if self.audit is None:
            return
        is_correct = (None if correct_tool is None or tool_chosen is None
                      else tool_chosen == correct_tool)
        self.audit.record(
            setup=setup,
            query=prepared.query,
            tools_retrieved=prepared.retrieved_names,
            tool_chosen=tool_chosen,
            correct_tool=correct_tool,
            is_correct=is_correct,
            flags=prepared.flags,
        )


def build_default_proxy(*, include_real: bool = True, allow_network: bool = True,
                        with_retriever: bool = True,
                        with_audit: bool = True) -> GatewayProxy:
    catalog = build_catalog(include_real=include_real, allow_network=allow_network)
    retriever = FaissRetriever(catalog) if with_retriever else None
    policy = PolicyEngine()
    policy.register_baseline(catalog.tools)
    audit = AuditLog() if with_audit else None
    return GatewayProxy(catalog=catalog, retriever=retriever, policy=policy, audit=audit)
