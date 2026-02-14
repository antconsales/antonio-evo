"""Knowledge Search tool — wraps QdrantRAG for semantic knowledge base search."""

import time
from .registry import ToolResult

DEFINITION = {
    "name": "knowledge_search",
    "description": (
        "Search the local knowledge base for relevant information from indexed documents. "
        "Use this when the user asks about topics that may be covered in the knowledge base, "
        "such as project documentation, notes, or reference materials."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant knowledge",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 3)",
            },
        },
        "required": ["query"],
    },
}


def create_handler(rag_client):
    """Create knowledge_search tool handler bound to a QdrantRAG instance."""

    def knowledge_search(query: str, limit: int = 3) -> ToolResult:
        if not rag_client or not rag_client.is_available():
            return ToolResult(
                success=False,
                output="Knowledge base not available. RAG is not initialized or has no indexed documents.",
            )

        start = time.time()
        results = rag_client.search(query, limit=limit)
        elapsed = int((time.time() - start) * 1000)

        if not results:
            return ToolResult(
                success=True,
                output="No relevant documents found in the knowledge base.",
                elapsed_ms=elapsed,
            )

        # Format results as context text
        parts = [f"Found {len(results)} relevant knowledge base entries:\n"]
        for i, r in enumerate(results, 1):
            parts.append(f"--- [{i}] Source: {r.source} (score: {r.score:.2f}) ---")
            parts.append(r.text)
            parts.append("")

        return ToolResult(
            success=True,
            output="\n".join(parts),
            elapsed_ms=elapsed,
        )

    return knowledge_search
