"""
Qdrant RAG Client - Vector database for semantic search

Provides:
- Document indexing with embeddings
- Semantic search for relevant context
- Integration with existing memory system
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .document_loader import DocumentLoader, Chunk

logger = logging.getLogger(__name__)

# Try to import qdrant-client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed. RAG features disabled.")

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. RAG features disabled.")


@dataclass
class SearchResult:
    """Result from semantic search."""
    text: str
    source: str
    score: float
    metadata: Dict[str, Any]


class QdrantRAG:
    """
    Qdrant-based RAG client for semantic search.

    Features:
    - Index MD/TXT documents into vector database
    - Semantic search using sentence embeddings
    - Graceful degradation when Qdrant unavailable
    """

    COLLECTION_NAME = "antonio_knowledge"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Qdrant RAG client.

        Args:
            config: Configuration dictionary with:
                - server_url: Qdrant server URL
                - embedding_model: Sentence transformer model name
                - docs_path: Path to documents directory
                - chunk_size: Chunk size for documents
                - chunk_overlap: Overlap between chunks
                - top_k: Default number of results
        """
        self.server_url = config.get("server_url", "http://localhost:6333")
        self.embedding_model_name = config.get(
            "embedding_model", "all-MiniLM-L6-v2"
        )
        self.docs_path = config.get("docs_path", "data/knowledge")
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.top_k = config.get("top_k", 3)

        self._client: Optional["QdrantClient"] = None
        self._embedding_model: Optional["SentenceTransformer"] = None
        self._embedding_dim: int = 384  # Default for all-MiniLM-L6-v2

        self._document_loader = DocumentLoader(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # Initialize if dependencies available
        if QDRANT_AVAILABLE and EMBEDDINGS_AVAILABLE:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize Qdrant client and embedding model."""
        try:
            # Parse server URL
            if self.server_url.startswith("http"):
                # HTTP connection
                host = self.server_url.replace("http://", "").replace("https://", "")
                if ":" in host:
                    host, port = host.split(":")
                    port = int(port)
                else:
                    port = 6333

                self._client = QdrantClient(host=host, port=port)
            else:
                # Local path for in-memory or file-based storage
                self._client = QdrantClient(path=self.server_url)

            logger.info(f"Connected to Qdrant at {self.server_url}")

        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant: {e}")
            self._client = None

        try:
            # Load embedding model
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            self._embedding_dim = self._embedding_model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded embedding model: {self.embedding_model_name} "
                f"(dim={self._embedding_dim})"
            )
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self._embedding_model = None

    def is_available(self) -> bool:
        """Check if RAG is available."""
        return (
            self._client is not None and
            self._embedding_model is not None
        )

    def _ensure_collection(self) -> bool:
        """Create collection if it doesn't exist."""
        if not self._client:
            return False

        try:
            collections = self._client.get_collections().collections
            exists = any(c.name == self.COLLECTION_NAME for c in collections)

            if not exists:
                self._client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self._embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.COLLECTION_NAME}")

            return True

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            return False

    def index_documents(self, path: Optional[str] = None) -> int:
        """
        Index all documents from a directory.

        Args:
            path: Path to documents directory (default: self.docs_path)

        Returns:
            Number of chunks indexed
        """
        if not self.is_available():
            logger.warning("RAG not available, skipping indexing")
            return 0

        if not self._ensure_collection():
            return 0

        path = path or self.docs_path

        # Load and chunk documents
        chunks = list(self._document_loader.load_and_chunk(path))

        if not chunks:
            logger.info(f"No documents found in {path}")
            return 0

        logger.info(f"Indexing {len(chunks)} chunks from {path}")

        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedding_model.encode(texts, show_progress_bar=True)

        # Create points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Generate stable ID from source and index
            point_id = self._generate_point_id(chunk.source, chunk.chunk_index)

            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata
                }
            ))

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self._client.upsert(
                    collection_name=self.COLLECTION_NAME,
                    points=batch
                )
            except Exception as e:
                logger.error(f"Failed to index batch {i}: {e}")

        logger.info(f"Indexed {len(points)} chunks successfully")
        return len(points)

    def _generate_point_id(self, source: str, chunk_index: int) -> int:
        """Generate stable numeric ID for a point."""
        # Create hash from source and index
        id_string = f"{source}:{chunk_index}"
        hash_bytes = hashlib.md5(id_string.encode()).digest()
        # Use first 8 bytes as unsigned int
        return int.from_bytes(hash_bytes[:8], byteorder="big") % (2**63)

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        source_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            limit: Maximum number of results
            source_filter: Filter by source file path (partial match)

        Returns:
            List of SearchResult objects
        """
        if not self.is_available():
            return []

        limit = limit or self.top_k

        try:
            # Generate query embedding
            query_vector = self._embedding_model.encode(query).tolist()

            # Build filter if specified
            query_filter = None
            if source_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source_filter)
                        )
                    ]
                )

            # Search
            results = self._client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter
            )

            return [
                SearchResult(
                    text=hit.payload["text"],
                    source=hit.payload["source"],
                    score=hit.score,
                    metadata=hit.payload.get("metadata", {})
                )
                for hit in results
            ]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        bm25_results: List[Dict[str, Any]] = None,
        limit: int = 5,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector similarity with BM25 results (v8.0).

        Uses Reciprocal Rank Fusion (RRF) to merge vector and BM25 result lists.

        Args:
            query: Search query
            bm25_results: List of dicts with 'text' and 'source' from BM25 memory search
            limit: Maximum results
            bm25_weight: Weight for BM25 scores in fusion
            vector_weight: Weight for vector scores in fusion

        Returns:
            Fused list of SearchResult ordered by combined score
        """
        # Get vector results
        vector_results = self.search(query, limit=limit * 2)

        if not bm25_results:
            return vector_results[:limit]

        # RRF constant (standard value)
        k = 60

        # Score vector results with RRF
        scored = {}
        for rank, result in enumerate(vector_results):
            key = f"{result.source}:{result.text[:50]}"
            rrf_score = vector_weight / (k + rank + 1)
            scored[key] = {
                "result": result,
                "score": rrf_score,
            }

        # Score BM25 results with RRF
        for rank, bm25 in enumerate(bm25_results):
            text = bm25.get("text", "")[:200]
            source = bm25.get("source", "memory")
            key = f"{source}:{text[:50]}"
            rrf_score = bm25_weight / (k + rank + 1)
            if key in scored:
                scored[key]["score"] += rrf_score
            else:
                scored[key] = {
                    "result": SearchResult(
                        text=text,
                        source=source,
                        score=bm25.get("score", 0.5),
                        metadata=bm25.get("metadata", {}),
                    ),
                    "score": rrf_score,
                }

        # Sort by fused score and return top results
        fused = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        return [item["result"] for item in fused[:limit]]

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self._client:
            return {"available": False}

        try:
            collection = self._client.get_collection(self.COLLECTION_NAME)
            return {
                "available": True,
                "collection": self.COLLECTION_NAME,
                "points_count": collection.points_count,
                "vectors_count": collection.vectors_count,
                "embedding_model": self.embedding_model_name,
                "embedding_dim": self._embedding_dim
            }
        except Exception as e:
            return {
                "available": True,
                "error": str(e)
            }

    def clear(self) -> bool:
        """Clear all documents from the collection."""
        if not self._client:
            return False

        try:
            self._client.delete_collection(self.COLLECTION_NAME)
            logger.info(f"Deleted collection: {self.COLLECTION_NAME}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
