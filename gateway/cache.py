"""In-memory TTL cache for fetched tool definitions.

The gateway fetches tool schemas from (possibly slow, possibly remote) MCP
servers.  Re-fetching them on every user query is wasteful, so we cache them
with a time-to-live.

The whole point of this module for the benchmark is that it is *measurable*:
``fetch`` always records how long the underlying fetch took and whether the
result came from the cache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

import config

T = TypeVar("T")


@dataclass
class FetchResult(Generic[T]):
    value: T
    from_cache: bool
    fetch_seconds: float  # wall time spent in the (real or cache) fetch


@dataclass
class _Entry(Generic[T]):
    value: T
    stored_at: float


@dataclass
class TTLCache(Generic[T]):
    """A dict with per-entry expiry plus hit/miss + timing counters."""

    ttl_seconds: float = config.CACHE_TTL_SECONDS
    _store: dict[str, _Entry[T]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    total_fetch_seconds: float = 0.0  # time spent in *cold* fetches only

    def _fresh(self, entry: _Entry[T]) -> bool:
        return (time.time() - entry.stored_at) < self.ttl_seconds

    def peek(self, key: str) -> T | None:
        entry = self._store.get(key)
        if entry and self._fresh(entry):
            return entry.value
        return None

    def invalidate(self, key: str | None = None) -> None:
        if key is None:
            self._store.clear()
        else:
            self._store.pop(key, None)

    def fetch(self, key: str, loader: Callable[[], T]) -> FetchResult[T]:
        """Return ``key`` from cache if fresh, else call ``loader`` and store it.

        ``fetch_seconds`` is the time actually spent: ~0 on a hit, the real
        loader cost on a miss.
        """
        entry = self._store.get(key)
        if entry and self._fresh(entry):
            start = time.perf_counter()
            value = entry.value
            elapsed = time.perf_counter() - start
            self.hits += 1
            return FetchResult(value=value, from_cache=True, fetch_seconds=elapsed)

        start = time.perf_counter()
        value = loader()
        elapsed = time.perf_counter() - start
        self._store[key] = _Entry(value=value, stored_at=time.time())
        self.misses += 1
        self.total_fetch_seconds += elapsed
        return FetchResult(value=value, from_cache=False, fetch_seconds=elapsed)

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "cold_fetch_seconds_total": self.total_fetch_seconds,
        }
