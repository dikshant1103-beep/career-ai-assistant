"""Document chunking.

We use LangChain's RecursiveCharacterTextSplitter as a strong default
(splits on paragraph -> sentence -> word boundaries) and optionally fall
back to a token-aware splitter via tiktoken.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class SemanticChunker:
    """Wrapper that produces chunks suitable for embedding.

    The default settings are tuned for resume / thesis / job-description style
    content (~900 chars with 120 chars overlap).
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        s = get_settings()
        self.chunk_size = chunk_size or s.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or s.CHUNK_OVERLAP
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        chunks = self._splitter.split_documents(docs)
        # attach a stable chunk index per source file
        per_source_counter: dict[str, int] = {}
        for c in chunks:
            src = c.metadata.get("source", "unknown")
            per_source_counter[src] = per_source_counter.get(src, 0) + 1
            c.metadata["chunk_index"] = per_source_counter[src]
        log.info("Chunked %d document(s) into %d chunk(s)", len(docs), len(chunks))
        return chunks
