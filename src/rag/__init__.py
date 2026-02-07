"""
RAG (Retrieval-Augmented Generation) Module

Provides semantic search over local documents using Qdrant vector database.
Enables the chatbot to answer questions based on MD/TXT files.

Components:
- QdrantRAG: Vector database client for semantic search
- DocumentLoader: Load and chunk MD/TXT documents
- Indexer: CLI for document indexing
"""

from .qdrant_client import QdrantRAG
from .document_loader import DocumentLoader, Document, Chunk

__all__ = ["QdrantRAG", "DocumentLoader", "Document", "Chunk"]
