"""Web Search tool - wraps existing Tavily WebSearchService."""

from .registry import ToolResult

DEFINITION = {
    "name": "web_search",
    "description": (
        "Search the web for current information, news, prices, weather, "
        "sports results, or any topic requiring up-to-date data. "
        "Use this when the user asks about recent events or facts you're unsure about."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string",
            }
        },
        "required": ["query"],
    },
}


def create_handler(service):
    """Create web_search tool handler bound to a WebSearchService instance."""

    def web_search(query: str) -> ToolResult:
        if not service or not service.is_available():
            return ToolResult(
                success=False,
                output="Web search not configured (missing Tavily API key). "
                       "Ask the user to add their key in Settings > Web Search.",
            )
        result = service.search(query)
        if result.success:
            return ToolResult(
                success=True,
                output=result.to_context(),
                elapsed_ms=result.elapsed_ms,
            )
        return ToolResult(
            success=False,
            output=f"Search failed: {result.error}",
            error=result.error,
        )

    return web_search
