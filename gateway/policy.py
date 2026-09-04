"""Policy plane: allow/deny, tamper detection, prompt-injection scanning, audit.

Responsibilities
----------------
* **Allowlist / denylist** - hard filter on which tools may ever reach the LLM.
* **Description hashing** - remember a SHA-256 of every tool description; on a
  later refresh, if a description changed without the tool name changing, flag
  it as a possible "rug pull" / tool-poisoning attack.
* **Injection scan** - look for classic injected-instruction phrasing inside
  tool descriptions ("ignore previous instructions", "do not tell the user"...).
* **Audit log** - append-only SQLite row for every query the gateway serves:
  timestamp, query, tools retrieved, tool chosen, and whether anything flagged.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import config
from gateway.tools import Tool

# --------------------------------------------------------------------------- #
# Suspicious-instruction patterns
# --------------------------------------------------------------------------- #

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all|any|the)? ?(previous|prior|above) (instructions|prompts?)",
        r"disregard (the|all|any)? ?(system|previous) (prompt|instructions)",
        r"do not (tell|inform|mention to) the user",
        r"without (telling|informing|notifying) the user",
        r"you are now (a|an|in) ",
        r"reveal (the|your) (system prompt|instructions|api[_ ]?key|secret)",
        r"(send|exfiltrate|post) .{0,40}(to|at) https?://",
        r"<\s*important\s*>",
        r"\bbase64\b.{0,30}\bdecode\b",
        r"assistant must (always|never)",
    ]
]


@dataclass
class Flag:
    tool_name: str
    kind: str  # "injection" | "description_changed" | "denylisted"
    detail: str


# --------------------------------------------------------------------------- #
# Audit log
# --------------------------------------------------------------------------- #


class AuditLog:
    def __init__(self, path=config.AUDIT_DB_PATH):
        config.ensure_dir(Path(path).parent)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_calls (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                ts             REAL NOT NULL,
                iso_ts         TEXT NOT NULL,
                setup          TEXT NOT NULL,
                query          TEXT NOT NULL,
                tools_retrieved TEXT NOT NULL,
                tool_chosen    TEXT,
                correct_tool   TEXT,
                is_correct     INTEGER,
                flagged        INTEGER NOT NULL,
                flags          TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def record(self, *, setup: str, query: str, tools_retrieved: list[str],
               tool_chosen: str | None, correct_tool: str | None,
               is_correct: bool | None, flags: list[Flag]) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO tool_calls (ts, iso_ts, setup, query, tools_retrieved, "
            "tool_chosen, correct_tool, is_correct, flagged, flags) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                now,
                time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)),
                setup,
                query,
                json.dumps(tools_retrieved),
                tool_chosen,
                correct_tool,
                None if is_correct is None else int(is_correct),
                int(bool(flags)),
                json.dumps([f.__dict__ for f in flags]),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def summary(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(flagged),0) FROM tool_calls"
        ).fetchone()
        return {"rows": rows[0], "flagged": rows[1]}

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------- #
# Policy engine
# --------------------------------------------------------------------------- #


@dataclass
class PolicyReport:
    allowed: list[Tool]
    blocked: list[Flag]
    injection_flags: list[Flag]
    changed_flags: list[Flag]

    @property
    def all_flags(self) -> list[Flag]:
        return self.blocked + self.injection_flags + self.changed_flags


@dataclass
class PolicyEngine:
    allowlist: set[str] | None = None          # None => allow everything not denied
    denylist: set[str] = field(default_factory=set)
    _hashes: dict[str, str] = field(default_factory=dict)  # tool name -> desc hash
    _hash_store = None

    def __post_init__(self) -> None:
        self._hash_store = config.DATA_DIR / "desc_hashes.json"
        if self._hash_store.exists():
            self._hashes = json.loads(self._hash_store.read_text())

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _hash(desc: str) -> str:
        return hashlib.sha256(desc.strip().encode("utf-8")).hexdigest()

    def scan_injection(self, tool: Tool) -> Flag | None:
        for pat in _INJECTION_PATTERNS:
            m = pat.search(tool.description)
            if m:
                return Flag(tool.name, "injection",
                           f"matched /{pat.pattern}/ -> {m.group(0)!r}")
        return None

    def check_description_change(self, tool: Tool, *, persist: bool = True) -> Flag | None:
        h = self._hash(tool.description)
        prev = self._hashes.get(tool.name)
        flag = None
        if prev is not None and prev != h:
            flag = Flag(tool.name, "description_changed",
                       f"description hash changed {prev[:12]}... -> {h[:12]}...")
        if persist:
            self._hashes[tool.name] = h
        return flag

    def register_baseline(self, tools: list[Tool]) -> None:
        """Record current descriptions as the trusted baseline."""
        for t in tools:
            self._hashes[t.name] = self._hash(t.description)
        self._flush()

    def _flush(self) -> None:
        if self._hash_store is not None:
            config.ensure_dir(self._hash_store.parent)
            self._hash_store.write_text(json.dumps(self._hashes, indent=2))

    # -- main entrypoint ------------------------------------------------------ #
    def evaluate(self, tools: list[Tool], *, persist_hashes: bool = True) -> PolicyReport:
        allowed: list[Tool] = []
        blocked: list[Flag] = []
        injection_flags: list[Flag] = []
        changed_flags: list[Flag] = []

        for t in tools:
            if t.name in self.denylist:
                blocked.append(Flag(t.name, "denylisted", "tool is on the denylist"))
                continue
            if self.allowlist is not None and t.name not in self.allowlist:
                blocked.append(Flag(t.name, "denylisted", "tool not on the allowlist"))
                continue

            inj = self.scan_injection(t)
            if inj:
                injection_flags.append(inj)
            chg = self.check_description_change(t, persist=persist_hashes)
            if chg:
                changed_flags.append(chg)

            # Flagged tools are still returned, but marked - the agent / caller
            # decides. Denylisted tools are hard-removed.
            allowed.append(t)

        if persist_hashes:
            self._flush()
        return PolicyReport(allowed, blocked, injection_flags, changed_flags)
