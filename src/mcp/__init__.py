"""
MCP (Model Context Protocol) Integration for Antonio Evo.

Per Antonio Evo Unified Spec (v3.1):
MCP is a CAPABILITY PROVIDER, not a tool layer.

- Antonio NEVER calls MCP directly
- Antonio NEVER selects capabilities
- Antonio NEVER triggers side effects

Antonio MAY ONLY:
- Describe what a capability does
- Help compile task input
- Explain consequences
"""

from .server import MCPServer, MCPTool, MCPResource, get_mcp_server
from .capabilities import (
    CapabilityType,
    ApprovalStatus,
    CapabilityDefinition,
    CapabilityRequest,
    CapabilityResult,
    CapabilityRegistry,
    CapabilityGate,
    MCPCapabilityProvider,
)

__all__ = [
    # MCP Server
    "MCPServer",
    "MCPTool",
    "MCPResource",
    "get_mcp_server",
    # Capabilities (v3.1)
    "CapabilityType",
    "ApprovalStatus",
    "CapabilityDefinition",
    "CapabilityRequest",
    "CapabilityResult",
    "CapabilityRegistry",
    "CapabilityGate",
    "MCPCapabilityProvider",
]
