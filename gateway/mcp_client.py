"""A tiny, dependency-free MCP stdio client.

Just enough of the Model Context Protocol to launch a server process, perform
the ``initialize`` handshake and pull its ``tools/list``.  Uses newline-delimited
JSON-RPC 2.0 over the child process's stdin/stdout, which is what the MCP stdio
transport specifies.

This is deliberately minimal - we only *read* tool schemas, we never call tools
through it - so the gateway can talk to real public MCP servers without pulling
in the full MCP SDK.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "2024-11-05"


@dataclass
class MCPServerSpec:
    """How to launch one MCP server."""

    key: str
    command: list[str]
    note: str = ""


@dataclass
class _Pending:
    event: threading.Event = field(default_factory=threading.Event)
    payload: dict[str, Any] | None = None


class MCPStdioClient:
    """Context-managed stdio JSON-RPC client for a single MCP server."""

    def __init__(self, command: list[str], startup_timeout: float = 60.0):
        self._command = command
        self._startup_timeout = startup_timeout
        self._proc: subprocess.Popen[str] | None = None
        self._next_id = 1
        self._pending: dict[int, _Pending] = {}
        self._reader: threading.Thread | None = None
        self._alive = False

    # -- lifecycle --------------------------------------------------------- #
    def __enter__(self) -> "MCPStdioClient":
        self._proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._alive = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._alive = False
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()

    # -- io -------------------------------------------------------------- #
    def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg_id = msg.get("id")
            if msg_id in self._pending:
                self._pending[msg_id].payload = msg
                self._pending[msg_id].event.set()

    def _send(self, obj: dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _request(self, method: str, params: dict[str, Any] | None = None,
                 timeout: float = 30.0) -> dict[str, Any]:
        msg_id = self._next_id
        self._next_id += 1
        pending = _Pending()
        self._pending[msg_id] = pending
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method,
                    "params": params or {}})
        if not pending.event.wait(timeout):
            raise TimeoutError(f"MCP request {method!r} timed out after {timeout}s")
        payload = pending.payload or {}
        del self._pending[msg_id]
        if "error" in payload:
            raise RuntimeError(f"MCP error for {method!r}: {payload['error']}")
        return payload.get("result", {})

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # -- public api ---------------------------------------------------------- #
    def initialize(self) -> dict[str, Any]:
        result = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "mcp-gateway", "version": "0.1.0"},
            },
            timeout=self._startup_timeout,
        )
        self._notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = self._request("tools/list", params)
            tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools


def fetch_tools_from_server(spec: MCPServerSpec, startup_timeout: float = 60.0
                            ) -> list[dict[str, Any]]:
    """Launch ``spec``, handshake, return its raw tool dicts. Raises on failure."""
    start = time.perf_counter()
    with MCPStdioClient(spec.command, startup_timeout=startup_timeout) as client:
        client.initialize()
        tools = client.list_tools()
    _ = time.perf_counter() - start
    return tools
