"""
Document Loader - Load and chunk MD/TXT documents for RAG

Supports:
- Markdown (.md) files with header extraction
- Plain text (.txt) files
- Recursive directory scanning
- Overlapping chunks for context preservation
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a loaded document."""
    content: str
    source: str  # File path
    filename: str
    extension: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Represents a document chunk for embedding."""
    text: str
    source: str  # Original file path
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Generate unique chunk ID."""
        return f"{self.source}:{self.chunk_index}"


class DocumentLoader:
    """
    Load and chunk documents from a directory.

    Supports MD and TXT files with configurable chunking.
    """

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".markdown", ".csv", ".json"}

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50
    ):
        """
        Initialize document loader.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def load_directory(
        self,
        path: str,
        recursive: bool = True
    ) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            path: Directory path
            recursive: Whether to scan subdirectories

        Returns:
            List of Document objects
        """
        documents = []
        base_path = Path(path)

        if not base_path.exists():
            logger.warning(f"Directory not found: {path}")
            return documents

        if not base_path.is_dir():
            logger.warning(f"Not a directory: {path}")
            return documents

        # Get all files
        if recursive:
            files = base_path.rglob("*")
        else:
            files = base_path.glob("*")

        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                doc = self.load_file(str(file_path))
                if doc:
                    documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from {path}")
        return documents

    def load_file(self, path: str) -> Optional[Document]:
        """
        Load a single document file.

        Args:
            path: File path

        Returns:
            Document object or None if loading fails
        """
        file_path = Path(path)

        if not file_path.exists():
            logger.warning(f"File not found: {path}")
            return None

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract metadata (v8.0 enhanced)
            metadata = self._extract_common_metadata(content)
            if file_path.suffix.lower() in {".md", ".markdown"}:
                metadata.update(self._extract_markdown_metadata(content))

            return Document(
                content=content,
                source=str(file_path),
                filename=file_path.name,
                extension=file_path.suffix.lower(),
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None

    def _extract_common_metadata(self, content: str) -> dict:
        """Extract common metadata from any document (v8.0)."""
        metadata = {}
        metadata["word_count"] = len(content.split())
        metadata["char_count"] = len(content)

        # Detect code blocks
        code_blocks = re.findall(r"```[\s\S]*?```", content)
        metadata["code_block_count"] = len(code_blocks)

        # Count links
        links = re.findall(r"https?://\S+", content)
        metadata["link_count"] = len(links)

        # Simple language hint (check for common words)
        lower = content.lower()[:500]
        if any(w in lower for w in ["il ", "la ", "che ", "di ", "per "]):
            metadata["language_hint"] = "it"
        elif any(w in lower for w in ["the ", "is ", "and ", "for ", "with "]):
            metadata["language_hint"] = "en"

        return metadata

    def _extract_markdown_metadata(self, content: str) -> dict:
        """Extract metadata from markdown content."""
        metadata = {}

        # Extract title (first H1)
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()

        # Extract headers for structure
        headers = re.findall(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)
        if headers:
            metadata["headers"] = [
                {"level": len(h[0]), "text": h[1].strip()}
                for h in headers[:10]  # Limit to first 10 headers
            ]

        # Check for YAML frontmatter
        frontmatter_match = re.match(r"^---\n(.+?)\n---", content, re.DOTALL)
        if frontmatter_match:
            metadata["has_frontmatter"] = True

        return metadata

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split document into overlapping chunks.

        Uses a simple character-based splitting with sentence awareness.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        content = document.content
        chunks = []

        # Clean content
        content = self._clean_content(content)

        if len(content) <= self.chunk_size:
            # Document is smaller than chunk size
            if len(content) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=content,
                    source=document.source,
                    chunk_index=0,
                    metadata={
                        "filename": document.filename,
                        **document.metadata
                    }
                ))
            return chunks

        # Split into chunks with overlap
        start = 0
        chunk_index = 0

        while start < len(content):
            # Calculate end position
            end = start + self.chunk_size

            if end >= len(content):
                # Last chunk
                chunk_text = content[start:]
            else:
                # Find a good break point (sentence end, paragraph, etc.)
                chunk_text = content[start:end]
                break_point = self._find_break_point(chunk_text)

                if break_point > self.min_chunk_size:
                    chunk_text = chunk_text[:break_point]
                    end = start + break_point

            # Add chunk if not too small
            chunk_text = chunk_text.strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    source=document.source,
                    chunk_index=chunk_index,
                    metadata={
                        "filename": document.filename,
                        "start_char": start,
                        **document.metadata
                    }
                ))
                chunk_index += 1

            # Move start position (with overlap)
            start = end - self.chunk_overlap
            if start <= 0 and chunk_index > 0:
                break  # Avoid infinite loop

        return chunks

    def _clean_content(self, content: str) -> str:
        """Clean document content for chunking."""
        # Remove excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r" {2,}", " ", content)

        # Remove markdown code block markers (keep content)
        content = re.sub(r"```\w*\n?", "", content)

        return content.strip()

    def _find_break_point(self, text: str) -> int:
        """
        Find a good break point in text.

        Prioritizes: paragraph > sentence > word boundary
        """
        # Try paragraph break (double newline)
        para_break = text.rfind("\n\n")
        if para_break > len(text) * 0.5:
            return para_break + 2

        # Try sentence break (., !, ?)
        for char in [".", "!", "?"]:
            sent_break = text.rfind(char)
            if sent_break > len(text) * 0.5:
                return sent_break + 1

        # Try single newline
        line_break = text.rfind("\n")
        if line_break > len(text) * 0.5:
            return line_break + 1

        # Try word break (space)
        word_break = text.rfind(" ")
        if word_break > len(text) * 0.5:
            return word_break + 1

        # No good break point found
        return len(text)

    def load_and_chunk(
        self,
        path: str,
        recursive: bool = True
    ) -> Iterator[Chunk]:
        """
        Load documents and yield chunks.

        Args:
            path: Directory or file path
            recursive: Whether to scan subdirectories

        Yields:
            Chunk objects
        """
        path_obj = Path(path)

        if path_obj.is_file():
            doc = self.load_file(path)
            if doc:
                yield from self.chunk_document(doc)
        else:
            documents = self.load_directory(path, recursive)
            for doc in documents:
                yield from self.chunk_document(doc)
