"""
MCP Server for Antonio Evo.

Implements Model Context Protocol server to expose capabilities:
- Memory search and stats
- Emotional analysis
- Personality traits
- RAG document search
- Image generation and analysis
- LLM routing

Philosophy: Capabilities are tools. Tools require definitions.
"""

import json
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class MCPToolCategory(Enum):
    """Tool categories for organization."""
    MEMORY = "memory"
    EMOTIONAL = "emotional"
    PERSONALITY = "personality"
    RAG = "rag"
    GENERATION = "generation"
    ANALYSIS = "analysis"
    SYSTEM = "system"


@dataclass
class MCPToolParameter:
    """Parameter definition for an MCP tool."""
    name: str
    type: str  # string, number, boolean, array, object
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[str]] = None

    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class MCPTool:
    """Definition of an MCP tool."""
    name: str
    description: str
    category: MCPToolCategory
    parameters: List[MCPToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None
    requires_approval: bool = False
    cost_bearing: bool = False

    def to_mcp_format(self) -> Dict[str, Any]:
        """Convert to MCP tool format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class MCPResource:
    """Definition of an MCP resource."""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"
    handler: Optional[Callable] = None

    def to_mcp_format(self) -> Dict[str, Any]:
        """Convert to MCP resource format."""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


class MCPServer:
    """
    MCP Server implementation.

    Exposes Antonio's capabilities as MCP tools and resources.
    Can be integrated with external MCP clients.
    """

    def __init__(self, orchestrator: Optional[Any] = None):
        """
        Initialize MCP server.

        Args:
            orchestrator: Reference to main Orchestrator for capability access
        """
        self.orchestrator = orchestrator
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self._lock = threading.Lock()
        self._initialized = False

        # Register built-in tools
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register built-in Antonio tools."""

        # Memory tools
        self.register_tool(MCPTool(
            name="memory_search",
            description="Search Antonio's evolutionary memory for past interactions",
            category=MCPToolCategory.MEMORY,
            parameters=[
                MCPToolParameter(
                    name="query",
                    type="string",
                    description="Search query to find relevant memories",
                ),
                MCPToolParameter(
                    name="limit",
                    type="number",
                    description="Maximum number of results to return",
                    required=False,
                    default=5,
                ),
            ],
        ))

        self.register_tool(MCPTool(
            name="memory_stats",
            description="Get statistics about Antonio's memory system",
            category=MCPToolCategory.MEMORY,
            parameters=[],
        ))

        # Emotional tools
        self.register_tool(MCPTool(
            name="analyze_emotion",
            description="Analyze the emotional content of a message",
            category=MCPToolCategory.EMOTIONAL,
            parameters=[
                MCPToolParameter(
                    name="text",
                    type="string",
                    description="Text to analyze for emotional content",
                ),
            ],
        ))

        self.register_tool(MCPTool(
            name="get_emotional_context",
            description="Get the current emotional context for a session",
            category=MCPToolCategory.EMOTIONAL,
            parameters=[
                MCPToolParameter(
                    name="session_id",
                    type="string",
                    description="Session ID to get emotional context for",
                    required=False,
                ),
            ],
        ))

        # Personality tools
        self.register_tool(MCPTool(
            name="get_personality",
            description="Get Antonio's current personality traits",
            category=MCPToolCategory.PERSONALITY,
            parameters=[],
        ))

        self.register_tool(MCPTool(
            name="provide_feedback",
            description="Provide feedback to influence personality evolution",
            category=MCPToolCategory.PERSONALITY,
            parameters=[
                MCPToolParameter(
                    name="feedback_type",
                    type="string",
                    description="Type of feedback",
                    enum=["positive", "negative", "too_long", "too_short", "humor_good"],
                ),
            ],
        ))

        # RAG tools
        self.register_tool(MCPTool(
            name="search_documents",
            description="Search knowledge base documents using semantic search",
            category=MCPToolCategory.RAG,
            parameters=[
                MCPToolParameter(
                    name="query",
                    type="string",
                    description="Search query for document retrieval",
                ),
                MCPToolParameter(
                    name="limit",
                    type="number",
                    description="Maximum number of documents to return",
                    required=False,
                    default=5,
                ),
            ],
        ))

        # Generation tools
        self.register_tool(MCPTool(
            name="generate_image",
            description="Generate an image from a text prompt using Z-Image Turbo",
            category=MCPToolCategory.GENERATION,
            parameters=[
                MCPToolParameter(
                    name="prompt",
                    type="string",
                    description="Text description of the image to generate",
                ),
                MCPToolParameter(
                    name="width",
                    type="number",
                    description="Image width in pixels",
                    required=False,
                    default=512,
                ),
                MCPToolParameter(
                    name="height",
                    type="number",
                    description="Image height in pixels",
                    required=False,
                    default=512,
                ),
            ],
            requires_approval=True,
            cost_bearing=False,  # Local generation
        ))

        # Analysis tools
        self.register_tool(MCPTool(
            name="analyze_image",
            description="Analyze an image using CLIP for understanding",
            category=MCPToolCategory.ANALYSIS,
            parameters=[
                MCPToolParameter(
                    name="image_path",
                    type="string",
                    description="Path to the image file to analyze",
                    required=False,
                ),
                MCPToolParameter(
                    name="image_base64",
                    type="string",
                    description="Base64-encoded image data",
                    required=False,
                ),
            ],
        ))

        # System tools
        self.register_tool(MCPTool(
            name="health_check",
            description="Check Antonio's system health and component status",
            category=MCPToolCategory.SYSTEM,
            parameters=[],
        ))

        self.register_tool(MCPTool(
            name="get_profile",
            description="Get current runtime profile and capabilities",
            category=MCPToolCategory.SYSTEM,
            parameters=[],
        ))

        self.register_tool(MCPTool(
            name="chat",
            description="Send a message to Antonio for processing",
            category=MCPToolCategory.SYSTEM,
            parameters=[
                MCPToolParameter(
                    name="message",
                    type="string",
                    description="Message to send to Antonio",
                ),
                MCPToolParameter(
                    name="session_id",
                    type="string",
                    description="Session ID for context continuity",
                    required=False,
                ),
            ],
        ))

        # Register resources
        self._register_builtin_resources()

    def _register_builtin_resources(self):
        """Register built-in MCP resources."""

        self.register_resource(MCPResource(
            uri="antonio://memory/stats",
            name="Memory Statistics",
            description="Current memory system statistics",
        ))

        self.register_resource(MCPResource(
            uri="antonio://personality/profile",
            name="Personality Profile",
            description="Current personality trait values",
        ))

        self.register_resource(MCPResource(
            uri="antonio://emotional/context",
            name="Emotional Context",
            description="Current emotional analysis context",
        ))

        self.register_resource(MCPResource(
            uri="antonio://profile/capabilities",
            name="Profile Capabilities",
            description="Current runtime profile capabilities",
        ))

        self.register_resource(MCPResource(
            uri="antonio://llm/status",
            name="LLM Status",
            description="Status of available LLM endpoints",
        ))

    def register_tool(self, tool: MCPTool) -> None:
        """Register an MCP tool."""
        with self._lock:
            self.tools[tool.name] = tool
        logger.debug(f"Registered MCP tool: {tool.name}")

    def register_resource(self, resource: MCPResource) -> None:
        """Register an MCP resource."""
        with self._lock:
            self.resources[resource.uri] = resource
        logger.debug(f"Registered MCP resource: {resource.uri}")

    def set_orchestrator(self, orchestrator: Any) -> None:
        """Set orchestrator reference for tool execution."""
        self.orchestrator = orchestrator
        self._setup_tool_handlers()
        self._initialized = True

    def _setup_tool_handlers(self):
        """Setup handlers for registered tools."""
        if not self.orchestrator:
            return

        # Memory search handler
        if "memory_search" in self.tools:
            self.tools["memory_search"].handler = self._handle_memory_search

        # Health check handler
        if "health_check" in self.tools:
            self.tools["health_check"].handler = self._handle_health_check

        # Chat handler
        if "chat" in self.tools:
            self.tools["chat"].handler = self._handle_chat

        # Get profile handler
        if "get_profile" in self.tools:
            self.tools["get_profile"].handler = self._handle_get_profile

        # Get personality handler
        if "get_personality" in self.tools:
            self.tools["get_personality"].handler = self._handle_get_personality

    def _handle_memory_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory_search tool call."""
        query = params.get("query", "")
        limit = params.get("limit", 5)

        if not self.orchestrator:
            return {"error": "Orchestrator not available"}

        results = self.orchestrator.memory_search(query, limit)
        return {"results": results}

    def _handle_health_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health_check tool call."""
        if not self.orchestrator:
            return {"error": "Orchestrator not available"}

        return self.orchestrator.health_check()

    def _handle_chat(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chat tool call."""
        message = params.get("message", "")
        session_id = params.get("session_id")

        if not self.orchestrator:
            return {"error": "Orchestrator not available"}

        if session_id:
            self.orchestrator.current_session_id = session_id
        elif not self.orchestrator.current_session_id:
            self.orchestrator.start_session()

        result = self.orchestrator.process(message)
        return result

    def _handle_get_profile(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_profile tool call."""
        if not self.orchestrator:
            return {"error": "Orchestrator not available"}

        return self.orchestrator.profile_manager.get_stats()

    def _handle_get_personality(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_personality tool call."""
        if not self.orchestrator or not self.orchestrator.personality_engine:
            return {"error": "Personality engine not available"}

        profile = self.orchestrator.personality_engine.get_profile()
        return profile.to_dict()

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools in MCP format."""
        return [tool.to_mcp_format() for tool in self.tools.values()]

    def list_resources(self) -> List[Dict[str, Any]]:
        """List all available resources in MCP format."""
        return [resource.to_mcp_format() for resource in self.resources.values()]

    def call_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a registered tool.

        Args:
            name: Tool name
            params: Tool parameters

        Returns:
            Tool execution result
        """
        tool = self.tools.get(name)
        if not tool:
            return {
                "error": f"Unknown tool: {name}",
                "available_tools": list(self.tools.keys()),
            }

        if not tool.handler:
            return {"error": f"Tool {name} has no handler configured"}

        try:
            # Validate required parameters
            for param in tool.parameters:
                if param.required and param.name not in params:
                    if param.default is not None:
                        params[param.name] = param.default
                    else:
                        return {"error": f"Missing required parameter: {param.name}"}

            result = tool.handler(params)
            return {
                "success": True,
                "result": result,
            }

        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        resource = self.resources.get(uri)
        if not resource:
            return {
                "error": f"Unknown resource: {uri}",
                "available_resources": list(self.resources.keys()),
            }

        # Map URIs to handlers
        if uri == "antonio://memory/stats" and self.orchestrator:
            health = self.orchestrator.health_check()
            return {"content": health.get("memory", {})}

        elif uri == "antonio://personality/profile":
            if self.orchestrator and self.orchestrator.personality_engine:
                profile = self.orchestrator.personality_engine.get_profile()
                return {"content": profile.to_dict()}

        elif uri == "antonio://emotional/context":
            if self.orchestrator and self.orchestrator.emotional_memory:
                stats = self.orchestrator.emotional_memory.get_stats()
                return {"content": stats}

        elif uri == "antonio://profile/capabilities":
            if self.orchestrator:
                return {"content": self.orchestrator.profile_manager.get_stats()}

        elif uri == "antonio://llm/status":
            if self.orchestrator and self.orchestrator.llm_manager:
                return {"content": self.orchestrator.llm_manager.get_stats()}

        return {"error": "Resource handler not configured"}

    def get_stats(self) -> Dict[str, Any]:
        """Get MCP server statistics."""
        return {
            "version": "1.0",
            "enabled": True,
            "initialized": self._initialized,
            "tools_count": len(self.tools),
            "resources_count": len(self.resources),
            "tools_by_category": self._tools_by_category(),
        }

    def _tools_by_category(self) -> Dict[str, int]:
        """Count tools by category."""
        counts = {}
        for tool in self.tools.values():
            cat = tool.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts


# Singleton instance
_mcp_server: Optional[MCPServer] = None


def get_mcp_server(orchestrator: Optional[Any] = None) -> MCPServer:
    """Get or create the MCP server singleton."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer(orchestrator)
    elif orchestrator and not _mcp_server._initialized:
        _mcp_server.set_orchestrator(orchestrator)
    return _mcp_server
