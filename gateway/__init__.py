# Copyright (c) 2026 Harsh Chinmalli
# Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

"""mcp-gateway: a retrieval-and-policy layer between an AI agent and many MCP servers."""

from gateway.proxy import GatewayProxy
from gateway.retriever import FaissRetriever
from gateway.tools import build_catalog as load_catalogue

__version__ = "0.1.0"

__all__ = ["GatewayProxy", "FaissRetriever", "load_catalogue"]
