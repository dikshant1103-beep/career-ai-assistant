"""Ingestion orchestrator.

Walks the `data/` directory, loads everything, chunks it, embeds it, and
persists it in ChromaDB. Each top-level subfolder under `data/` becomes the
`category` metadata field — so you can later filter retrieval by, say,
{"category": "resume"}.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

from src.ingestion.chunker import SemanticChunker
from src.ingestion.loaders import load_directory, load_document
from src.rag.vectorstore import VectorStore
from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


CATEGORY_DIRS = ["resumes", "thesis", "papers", "linkedin", "sop", "jobs"]


class Ingestor:
    """Co-ordinates the full ingest pipeline."""

    def __init__(self, vectorstore: VectorStore | None = None):
        self.settings = get_settings()
        self.chunker = SemanticChunker()
        self.vectorstore = vectorstore or VectorStore()

    # ------------------------------------------------------------------ #
    def ingest_path(self, path: Path | str, category: str | None = None) -> int:
        """Ingest a single file. Returns number of chunks stored."""
        docs = load_document(Path(path), category=category)
        return self._chunk_and_store(docs)

    def ingest_directory(self, directory: Path | str, category: str | None = None) -> int:
        docs = load_directory(directory, category=category)
        return self._chunk_and_store(docs)

    def ingest_all(self) -> Dict[str, int]:
        """Ingest the entire `data/` directory, one category folder at a time."""
        data_dir = self.settings.data_path
        results: Dict[str, int] = {}
        for cat in CATEGORY_DIRS:
            sub = data_dir / cat
            if not sub.exists():
                continue
            label = cat.rstrip("s")  # "resumes" -> "resume"
            count = self.ingest_directory(sub, category=label)
            results[cat] = count
        total = sum(results.values())
        log.info("Ingest complete: %d chunks total across %d categories", total, len(results))
        return results

    # ------------------------------------------------------------------ #
    def _chunk_and_store(self, docs: List[Document]) -> int:
        if not docs:
            return 0
        chunks = self.chunker.split(docs)
        if not chunks:
            return 0
        self.vectorstore.add_documents(chunks)
        return len(chunks)

    def reset(self) -> None:
        """Drop the vector store and start fresh."""
        self.vectorstore.reset()
