"""Knowledge Graph tool — query entities and relationships from the knowledge graph."""

import time
from .registry import ToolResult

DEFINITION = {
    "name": "knowledge_graph",
    "description": (
        "Search the knowledge graph for entities and relationships. "
        "Use this when the user asks about people, projects, tools, or concepts "
        "they have previously discussed. Returns entities with their connections."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find entities",
            },
            "entity_type": {
                "type": "string",
                "description": "Optional filter: person, project, tool, concept, place, organization",
            },
        },
        "required": ["query"],
    },
}


def create_handler(kg_service):
    """Create knowledge_graph tool handler bound to a KnowledgeGraphService instance."""

    def knowledge_graph(query: str, entity_type: str = None) -> ToolResult:
        if not kg_service:
            return ToolResult(
                success=False,
                output="Knowledge graph not available.",
            )

        start = time.time()
        results = kg_service.search_entities(query, entity_type=entity_type, limit=10)
        elapsed = int((time.time() - start) * 1000)

        if not results:
            return ToolResult(
                success=True,
                output="No matching entities found in the knowledge graph.",
                elapsed_ms=elapsed,
            )

        parts = [f"Found {len(results)} entities in the knowledge graph:\n"]
        for e in results:
            line = f"- {e['name']} ({e['entity_type']})"
            if e.get("description"):
                line += f": {e['description']}"
            line += f" [mentioned {e['frequency']}x, confidence: {e['confidence']:.2f}]"
            parts.append(line)

        return ToolResult(
            success=True,
            output="\n".join(parts),
            elapsed_ms=elapsed,
        )

    return knowledge_graph
