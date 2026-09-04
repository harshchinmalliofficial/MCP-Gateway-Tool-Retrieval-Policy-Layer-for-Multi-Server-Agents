"""mcp-gateway: a retrieval-and-policy layer between an AI agent and many MCP servers."""

from gateway.proxy import GatewayProxy
from gateway.retriever import FaissRetriever
from gateway.tools import build_catalog as load_catalogue

__version__ = "0.1.0"

__all__ = ["GatewayProxy", "FaissRetriever", "load_catalogue"]
