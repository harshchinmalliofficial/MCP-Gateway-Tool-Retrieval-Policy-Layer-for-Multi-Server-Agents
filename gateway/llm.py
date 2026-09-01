"""Swappable LLM provider for the *tool-selection* step.

Given a user query and a list of candidate tool schemas, the provider must
return the single tool name it would call.  We measure three things for every
call: which tool it picked, how many tokens went in/out, and how long it took.

Backends (choose with ``config.LLM_PROVIDER`` / ``MCP_GATEWAY_PROVIDER``):

* ``groq``      - Groq chat completions.  Preferred: fast and free.
* ``gemini``    - Google Gemini.  Fallback.
* ``auto``      - groq if ``GROQ_API_KEY`` set, else gemini, else error.
* ``simulated`` - deterministic offline lexical matcher.  **Not a real model.**
                  Every number derived from it is labelled SIMULATED.

No API keys are ever hard-coded; they come from the environment / ``.env``.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import config
from gateway.tools import Tool


class MissingAPIKeyError(RuntimeError):
    """Raised when a real provider is requested but no key is available."""


@dataclass
class Selection:
    tool_name: str | None      # what the model picked (None if unparseable)
    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    provider: str
    is_simulated: bool

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_SYSTEM_PROMPT = (
    "You are a tool router. Given a user request and a JSON list of available "
    "tools, choose the ONE tool that best accomplishes the request. "
    "Reply with ONLY the tool's exact `name` value - no punctuation, no prose, "
    "no explanation. If nothing fits, reply exactly NONE."
)


def _build_user_prompt(query: str, tools: list[Tool]) -> str:
    catalogue = [t.llm_schema() for t in tools]
    return (
        f"User request:\n{query}\n\n"
        f"Available tools ({len(tools)}):\n"
        f"{json.dumps(catalogue, indent=1)}\n\n"
        "Answer with the single best tool name:"
    )


def _rough_token_estimate(text: str) -> int:
    # ~4 chars/token heuristic; only used when a provider omits usage stats.
    return max(1, round(len(text) / 4))


def _extract_tool_name(raw: str, valid: set[str]) -> str | None:
    cleaned = raw.strip().strip("`").strip().strip(".").strip()
    if cleaned in valid:
        return cleaned
    # model sometimes wraps or annotates - find the first valid name mentioned
    for token in re.split(r"[\s,\n`\"']+", cleaned):
        if token in valid:
            return token
    for name in valid:
        if name and name in cleaned:
            return name
    return None


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


class BaseProvider:
    name: str = "base"
    is_simulated: bool = False

    def select(self, query: str, tools: list[Tool]) -> Selection:  # pragma: no cover
        raise NotImplementedError


class GroqProvider(BaseProvider):
    name = "groq"

    def __init__(self, model: str = config.GROQ_MODEL):
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise MissingAPIKeyError("GROQ_API_KEY is not set")
        from groq import Groq  # lazy import

        self._client = Groq(api_key=key, timeout=config.LLM_REQUEST_TIMEOUT,
                            max_retries=0)
        # Primary model first, then known-good fallbacks if it has been retired
        # (Groq rotates its hosted lineup frequently).
        self._models = [model, *config.GROQ_FALLBACK_MODELS]
        seen: set[str] = set()
        self._models = [m for m in self._models if not (m in seen or seen.add(m))]
        self._model = self._models[0]
        self._use_reasoning_effort = True  # dropped automatically if unsupported

    @staticmethod
    def _is_model_gone(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(s in msg for s in ("decommission", "model_not_found",
                                      "does not exist", "no longer supported",
                                      "model_not_active", "not found for model"))

    def _create(self, messages: list[dict]):
        last_exc: Exception | None = None
        for m in self._models:
            try:
                kwargs: dict = dict(
                    model=m, temperature=config.LLM_TEMPERATURE,
                    max_tokens=config.LLM_MAX_TOKENS, messages=messages,
                )
                if self._use_reasoning_effort and config.GROQ_REASONING_EFFORT:
                    kwargs["reasoning_effort"] = config.GROQ_REASONING_EFFORT
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception as exc:  # noqa: BLE001
                    if "reasoning_effort" in str(exc).lower():
                        self._use_reasoning_effort = False
                        kwargs.pop("reasoning_effort", None)
                        resp = self._client.chat.completions.create(**kwargs)
                    else:
                        raise
                self._model = m
                return resp
            except Exception as exc:  # noqa: BLE001
                if self._is_model_gone(exc):
                    last_exc = exc
                    continue
                raise
        raise RuntimeError(f"no usable Groq model from {self._models}: {last_exc}")

    def select(self, query: str, tools: list[Tool]) -> Selection:
        user = _build_user_prompt(query, tools)
        valid = {t.name for t in tools}
        t0 = time.perf_counter()
        resp = self._create([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ])
        latency = time.perf_counter() - t0
        raw = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        ptok = getattr(usage, "prompt_tokens", None) or _rough_token_estimate(
            _SYSTEM_PROMPT + user)
        ctok = getattr(usage, "completion_tokens", None) or _rough_token_estimate(raw)
        return Selection(
            tool_name=_extract_tool_name(raw, valid),
            raw_response=raw,
            prompt_tokens=int(ptok),
            completion_tokens=int(ctok),
            latency_seconds=latency,
            provider=f"groq:{self._model}",
            is_simulated=False,
        )


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, model: str = config.GEMINI_MODEL):
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise MissingAPIKeyError("GEMINI_API_KEY is not set")
        import google.generativeai as genai  # lazy import

        genai.configure(api_key=key)
        self._genai = genai
        self._models = [model, *config.GEMINI_FALLBACK_MODELS]
        seen: set[str] = set()
        self._models = [m for m in self._models if not (m in seen or seen.add(m))]
        self._model_name = self._models[0]

    def _generate(self, user: str):
        cfg = self._genai.types.GenerationConfig(
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.LLM_MAX_TOKENS,
        )
        last_exc: Exception | None = None
        for name in self._models:
            try:
                model = self._genai.GenerativeModel(
                    name, system_instruction=_SYSTEM_PROMPT)
                resp = model.generate_content(user, generation_config=cfg)
                self._model_name = name
                return resp
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if any(s in msg for s in ("not found", "no longer available", "404",
                                          "quota", "429", "not available to new users")):
                    last_exc = exc
                    continue
                raise
        raise RuntimeError(f"no usable Gemini model from {self._models}: {last_exc}")

    def select(self, query: str, tools: list[Tool]) -> Selection:
        user = _build_user_prompt(query, tools)
        valid = {t.name for t in tools}
        t0 = time.perf_counter()
        resp = self._generate(user)
        latency = time.perf_counter() - t0
        try:
            raw = (resp.text or "").strip()
        except Exception:  # noqa: BLE001 - no text part (safety block / truncation)
            raw = ""
            try:
                parts = resp.candidates[0].content.parts
                raw = " ".join(getattr(p, "text", "") for p in parts).strip()
            except Exception:  # noqa: BLE001
                pass
        um = getattr(resp, "usage_metadata", None)
        ptok = getattr(um, "prompt_token_count", None) or _rough_token_estimate(
            _SYSTEM_PROMPT + user)
        ctok = getattr(um, "candidates_token_count", None) or _rough_token_estimate(raw)
        return Selection(
            tool_name=_extract_tool_name(raw, valid),
            raw_response=raw,
            prompt_tokens=int(ptok),
            completion_tokens=int(ctok),
            latency_seconds=latency,
            provider=f"gemini:{self._model_name}",
            is_simulated=False,
        )


class SimulatedProvider(BaseProvider):
    """Offline, deterministic lexical matcher. NOT a real model.

    Scores each candidate tool by token overlap between the query and the
    tool's name+description, with a small bonus for exact keyword hits. It has
    no semantic understanding - it is only here so the full pipeline (caching,
    retrieval, policy, audit, charts) is runnable and reproducible with no API
    key.  Its accuracy sits between "keyword search" and "real LLM".
    """

    name = "simulated"
    is_simulated = True

    _STOP = {
        "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is",
        "are", "with", "how", "do", "i", "my", "our", "please", "can", "you",
        "me", "that", "this", "from", "by", "it", "need", "want", "get",
    }

    def _tok(self, text: str) -> list[str]:
        return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in self._STOP]

    def select(self, query: str, tools: list[Tool]) -> Selection:
        t0 = time.perf_counter()
        q = self._tok(query)
        qset = set(q)
        best_name, best_score = None, -1.0
        for t in tools:
            name_toks = set(re.findall(r"[a-z0-9]+", t.name.lower()))
            desc_toks = set(self._tok(t.description))
            score = 2.0 * len(qset & name_toks) + 1.0 * len(qset & desc_toks)
            # verb/object nudges
            if any(w in t.name.lower() for w in q):
                score += 0.5
            if score > best_score:
                best_name, best_score = t.name, score
        raw = best_name or "NONE"
        latency = time.perf_counter() - t0
        user = _build_user_prompt(query, tools)
        return Selection(
            tool_name=best_name if best_score > 0 else None,
            raw_response=raw,
            prompt_tokens=_rough_token_estimate(_SYSTEM_PROMPT + user),
            completion_tokens=_rough_token_estimate(raw),
            latency_seconds=latency,
            provider="simulated:lexical-v1",
            is_simulated=True,
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def get_provider(kind: str | None = None) -> BaseProvider:
    kind = (kind or config.LLM_PROVIDER).lower()

    if kind == "auto":
        if os.environ.get("GROQ_API_KEY"):
            kind = "groq"
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            kind = "gemini"
        else:
            raise MissingAPIKeyError(config.provider_help())

    if kind == "groq":
        try:
            return GroqProvider()
        except MissingAPIKeyError:
            if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
                print("  [llm] GROQ_API_KEY missing - falling back to Gemini")
                return GeminiProvider()
            raise MissingAPIKeyError(config.provider_help())
    if kind == "gemini":
        return GeminiProvider()
    if kind == "simulated":
        return SimulatedProvider()

    raise ValueError(f"Unknown LLM provider {kind!r} "
                     "(expected groq | gemini | auto | simulated)")
