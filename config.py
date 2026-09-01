"""Central configuration for mcp-gateway.

Every tunable lives here so the benchmark and the gateway agree on the same
knobs.  Anything secret (API keys) is read from the environment or a local
``.env`` file - never hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "benchmark" / "results"
AUDIT_DB_PATH = DATA_DIR / "audit.sqlite3"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Minimal .env loader (avoids a python-dotenv dependency)
# --------------------------------------------------------------------------- #
def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Populate os.environ from a KEY=VALUE file if present. Never overrides
    a variable that is already set in the real environment."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


# --------------------------------------------------------------------------- #
# LLM provider — swap the whole backend with ONE value
# --------------------------------------------------------------------------- #
# "groq"       -> Groq chat completions (preferred: fast + free)
# "gemini"     -> Google Gemini (fallback)
# "auto"       -> use groq if GROQ_API_KEY is set, else gemini
# "simulated"  -> deterministic offline matcher (NOT a real model; for plumbing
#                 tests and CI). Accuracy numbers produced this way are labelled
#                 SIMULATED everywhere they appear.
LLM_PROVIDER = os.environ.get("MCP_GATEWAY_PROVIDER", "auto")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
# Tried in order if GROQ_MODEL has been retired (Groq rotates its lineup often).
GROQ_FALLBACK_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b",
                        "qwen/qwen3.8-27b", "llama-3.3-70b-versatile"]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_FALLBACK_MODELS = ["gemini-flash-lite-latest", "gemini-3.1-flash-lite",
                          "gemini-flash-latest", "gemini-2.5-flash"]

# Keep the model honest: we only ever want a tool name back. Reasoning models
# (Groq's gpt-oss / qwen3) spend completion tokens on hidden reasoning before
# the answer, so the cap can't be tiny; `reasoning_effort=low` keeps it short.
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))
GROQ_REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "low")
LLM_REQUEST_TIMEOUT = float(os.environ.get("LLM_REQUEST_TIMEOUT", "45"))
# Fixed pause after every successful LLM call, to stay under free-tier RPM caps.
LLM_PACE_SECONDS = float(os.environ.get("LLM_PACE_SECONDS", "0"))


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "5"))
# Hybrid retrieval: pull a wider semantic shortlist, then re-rank it with a
# blended (semantic + lexical) score. alpha=1.0 => pure vector search.
RETRIEVAL_POOL = int(os.environ.get("RETRIEVAL_POOL", "30"))
RETRIEVAL_ALPHA = float(os.environ.get("RETRIEVAL_ALPHA", "0.6"))


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
CACHE_TTL_SECONDS = float(os.environ.get("CACHE_TTL_SECONDS", "300"))
# Simulated per-server network latency for a "cold" tool fetch, so the
# cache-vs-no-cache difference is measurable without depending on a flaky
# remote host. Real MCP servers (when reachable) use their real latency.
SIMULATED_FETCH_LATENCY_MS = float(os.environ.get("SIMULATED_FETCH_LATENCY_MS", "20"))


# --------------------------------------------------------------------------- #
# Synthetic tool sprawl
# --------------------------------------------------------------------------- #
SYNTHETIC_TOOL_COUNT = int(os.environ.get("SYNTHETIC_TOOL_COUNT", "130"))
SYNTHETIC_SEED = int(os.environ.get("SYNTHETIC_SEED", "7"))


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
CROSSOVER_TOOL_COUNTS = [10, 20, 40, 80, 150]
BENCH_REPEATS = int(os.environ.get("BENCH_REPEATS", "1"))


def provider_help() -> str:
    return (
        "No LLM API key found.\n"
        "  * Groq (preferred, free):  export GROQ_API_KEY=...   "
        "(get one at https://console.groq.com/keys)\n"
        "  * Gemini (fallback):       export GEMINI_API_KEY=...  "
        "(https://aistudio.google.com/apikey)\n"
        "Or add either line to a .env file in the project root.\n"
        "To run the pipeline with no key (accuracy numbers will be labelled "
        "SIMULATED), set:  export MCP_GATEWAY_PROVIDER=simulated"
    )
