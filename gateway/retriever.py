"""Semantic (hybrid) tool retrieval with FAISS.

At startup we embed every tool's ``name + description`` into a FAISS index.
Per query we embed the user's question, take a wider semantic shortlist, then
re-rank it with a blended ``alpha * semantic + (1 - alpha) * lexical`` score
(lexical = query/tool token overlap) and return the top-k - so the LLM only
ever sees a handful of candidates instead of the whole catalogue.  Pure vector
search is the ``alpha = 1.0`` special case.

Embeddings are pluggable (``Embedder`` protocol).  The default is the free,
local ``all-MiniLM-L6-v2`` sentence-transformer - no API cost, no network after
the first model download.  A dependency-free ``HashingEmbedder`` is provided as
a fallback so the module still imports and runs in a minimal environment
(clearly lower quality; the benchmark prints which one is active).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

import config
from gateway.tools import Tool, ToolCatalog

try:  # pragma: no cover - import guard
    import faiss  # type: ignore

    _HAVE_FAISS = True
except Exception:  # noqa: BLE001
    _HAVE_FAISS = False


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised row vectors."""
        ...


_LEX_STOP = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is", "are",
    "with", "how", "do", "i", "my", "our", "please", "can", "you", "me", "that",
    "this", "from", "by", "it", "need", "want", "get", "so", "up", "new", "all",
}


def _tokenise(text: str) -> list[str]:
    """lowercase alphanumeric tokens, split on non-alnum and on '_' / '.', minus
    a small stop list. Shared by the retriever's lexical half."""
    raw = re.findall(r"[a-z0-9]+", text.lower().replace("_", " ").replace(".", " "))
    return [w for w in raw if w not in _LEX_STOP and len(w) > 1]


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype("float32")


class SentenceTransformerEmbedder:
    """Free, local embeddings via sentence-transformers."""

    def __init__(self, model_name: str = config.EMBED_MODEL_NAME):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name.split('/')[-1]}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32")


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens vectors. No dependencies, no network.

    Good enough to demonstrate the pipeline; noticeably weaker than a real
    sentence encoder. Used only when sentence-transformers is unavailable.
    """

    def __init__(self, dim: int = 512):
        self.name = f"hashing-bow(dim={dim})"
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        out, cur = [], []
        for ch in text.lower():
            if ch.isalnum():
                cur.append(ch)
            elif cur:
                out.append("".join(cur))
                cur = []
        if cur:
            out.append("".join(cur))
        return out

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for tok in self._tokens(text):
                h = hash((tok, "a")) % self.dim
                mat[i, h] += 1.0
                # a cheap bigram-ish signal
                h2 = hash((tok, "b")) % self.dim
                mat[i, h2] += 0.5
        return _l2_normalise(mat)


def make_default_embedder() -> Embedder:
    try:
        return SentenceTransformerEmbedder()
    except Exception as exc:  # noqa: BLE001
        print(f"  [retriever] sentence-transformers unavailable ({type(exc).__name__}); "
              f"falling back to HashingEmbedder")
        return HashingEmbedder()


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #


@dataclass
class RetrievalHit:
    tool: Tool
    score: float


class _NumpyFlatIP:
    """Tiny stand-in for faiss.IndexFlatIP when faiss isn't installed."""

    def __init__(self, matrix: np.ndarray):
        self._m = matrix

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sims = q @ self._m.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores, idx


class FaissRetriever:
    """Embed a catalogue once, then answer top-k tool queries."""

    def __init__(self, catalog: ToolCatalog, embedder: Embedder | None = None,
                 top_k: int = config.RETRIEVAL_TOP_K,
                 pool: int = config.RETRIEVAL_POOL,
                 alpha: float = config.RETRIEVAL_ALPHA):
        self.embedder = embedder or make_default_embedder()
        self.top_k = top_k
        self.pool = pool           # semantic shortlist size before re-ranking
        self.alpha = alpha         # weight on the semantic score (1.0 => pure vector)
        self._tools: list[Tool] = list(catalog.tools)
        self._lex_tokens: list[set[str]] = [
            set(_tokenise(t.lexical_text())) for t in self._tools
        ]

        t0 = time.perf_counter()
        matrix = self.embedder.encode([t.embedding_text() for t in self._tools])
        matrix = _l2_normalise(np.asarray(matrix, dtype="float32"))
        if _HAVE_FAISS:
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self._index: object = index
            self.backend = "faiss.IndexFlatIP"
        else:
            self._index = _NumpyFlatIP(matrix)
            self.backend = "numpy-fallback"
        self.build_seconds = time.perf_counter() - t0

    def __len__(self) -> int:
        return len(self._tools)

    def retrieve(self, query: str, k: int | None = None) -> tuple[list[RetrievalHit], float]:
        """Hybrid top-k retrieval.

        1. embed the query, take the top ``pool`` tools by cosine similarity
           (FAISS),
        2. re-rank that shortlist by ``alpha * semantic + (1 - alpha) * lexical``
           where lexical = query/tool token overlap,
        3. return the top ``k``.

        Returns (hits, retrieval_seconds); the timing is the whole extra cost
        the gateway adds per query.
        """
        k = min(k or self.top_k, len(self._tools))
        pool = min(max(self.pool, k), len(self._tools))
        t0 = time.perf_counter()

        q = self.embedder.encode([query])
        q = _l2_normalise(np.asarray(q, dtype="float32"))
        sem_scores, sem_idx = self._index.search(q, pool)  # type: ignore[union-attr]
        sem_scores, sem_idx = sem_scores[0], sem_idx[0]

        q_tokens = set(_tokenise(query))
        cand: list[tuple[float, float, int]] = []  # (blended, semantic, tool_idx)
        sem_min, sem_max = float(sem_scores.min()), float(sem_scores.max())
        span = (sem_max - sem_min) or 1.0
        for s, i in zip(sem_scores, sem_idx):
            i = int(i)
            if i < 0:
                continue
            sem_norm = (float(s) - sem_min) / span
            lex = self._lexical_score(q_tokens, self._lex_tokens[i])
            blended = self.alpha * sem_norm + (1.0 - self.alpha) * lex
            cand.append((blended, float(s), i))

        cand.sort(key=lambda c: c[0], reverse=True)
        elapsed = time.perf_counter() - t0
        hits = [RetrievalHit(tool=self._tools[i], score=blended)
                for blended, _sem, i in cand[:k]]
        return hits, elapsed

    @staticmethod
    def _lexical_score(q_tokens: set[str], t_tokens: set[str]) -> float:
        if not q_tokens or not t_tokens:
            return 0.0
        overlap = len(q_tokens & t_tokens)
        return overlap / len(q_tokens)  # fraction of the query covered by the tool
