"""
RAG Indexer CLI - Command-line tool for document indexing

Usage:
    python -m src.rag.indexer                    # Index default path
    python -m src.rag.indexer /path/to/docs      # Index specific path
    python -m src.rag.indexer --clear            # Clear and reindex
    python -m src.rag.indexer --stats            # Show statistics
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config.env_loader import get_config
from src.rag.qdrant_client import QdrantRAG


def main():
    parser = argparse.ArgumentParser(
        description="Index documents for RAG search"
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to documents directory (default from config)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before indexing"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics only"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Test search query"
    )

    args = parser.parse_args()

    # Load config
    config = get_config()

    # Initialize RAG client
    rag_config = {
        "server_url": config.qdrant_server,
        "embedding_model": config.rag_embedding_model,
        "docs_path": config.rag_docs_path,
        "chunk_size": config.rag_chunk_size,
        "chunk_overlap": config.rag_chunk_overlap,
        "top_k": config.rag_top_k
    }

    rag = QdrantRAG(rag_config)

    if not rag.is_available():
        print("ERROR: RAG not available.")
        print("Make sure Qdrant is running and dependencies are installed:")
        print("  pip install qdrant-client sentence-transformers")
        print("  docker-compose up -d qdrant")
        return 1

    # Show stats
    if args.stats:
        stats = rag.get_stats()
        print("\n=== RAG Index Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return 0

    # Test search
    if args.search:
        print(f"\nSearching for: {args.search}")
        results = rag.search(args.search)
        if results:
            print(f"\nFound {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                print(f"--- Result {i} (score: {result.score:.3f}) ---")
                print(f"Source: {result.source}")
                print(f"Text: {result.text[:200]}...")
                print()
        else:
            print("No results found.")
        return 0

    # Get document path
    docs_path = args.path or config.rag_docs_path

    # Create directory if it doesn't exist
    if not os.path.exists(docs_path):
        print(f"Creating documents directory: {docs_path}")
        os.makedirs(docs_path, exist_ok=True)
        print("Add your MD/TXT files to this directory and run indexer again.")
        return 0

    # Clear if requested
    if args.clear:
        print("Clearing existing index...")
        rag.clear()

    # Index documents
    print(f"\nIndexing documents from: {docs_path}")
    count = rag.index_documents(docs_path)

    if count > 0:
        print(f"\nSuccessfully indexed {count} chunks")
        stats = rag.get_stats()
        print(f"Total points in index: {stats.get('points_count', 'unknown')}")
    else:
        print("\nNo documents to index.")
        print(f"Add MD/TXT files to: {docs_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
