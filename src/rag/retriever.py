"""Retrieval helpers used by every feature that needs profile context.

A typical call:

    retriever = Retriever()
    context = retriever.context_for("battery RUL Mamba project")
    # 'context' is a single string of relevant chunks, ready to feed to Claude.
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from src.rag.vectorstore import VectorStore
from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class Retriever:
    """Wraps the vector store with prompt-ready context formatting."""

    def __init__(self, store: Optional[VectorStore] = None) -> None:
        self.store = store or VectorStore()
        self.settings = get_settings()

    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        category: Optional[str] = None,
    ) -> List[Document]:
        """Pure similarity search."""
        k = k or self.settings.RAG_TOP_K
        filt = {"category": category} if category else None
        return self.store.similarity_search(query, k=k, metadata_filter=filt)

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        category: Optional[str] = None,
    ) -> List[tuple[Document, float]]:
        k = k or self.settings.RAG_TOP_K
        filt = {"category": category} if category else None
        return self.store.similarity_search_with_score(query, k=k, metadata_filter=filt)

    # ------------------------------------------------------------------ #
    def context_for(
        self,
        query: str,
        k: Optional[int] = None,
        category: Optional[str] = None,
        max_chars: int = 8000,
    ) -> str:
        """Return a single prompt-ready context string."""
        docs = self.retrieve(query, k=k, category=category)
        return self.format_context(docs, max_chars=max_chars)

    @staticmethod
    def format_context(docs: List[Document], max_chars: int = 8000) -> str:
        """Pretty-print retrieved docs into a single string.

        Truncates to ``max_chars`` to avoid blowing past Claude's context window.
        """
        if not docs:
            return "(no relevant context found in your profile)"

        parts: list[str] = []
        used = 0
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("filename", "unknown")
            cat = d.metadata.get("category", "?")
            page = d.metadata.get("page")
            header = f"[{i}] source={src} | category={cat}"
            if page is not None:
                header += f" | page={page}"
            block = f"{header}\n{d.page_content.strip()}"
            if used + len(block) > max_chars:
                remaining = max_chars - used
                if remaining > 200:
                    parts.append(block[:remaining] + "\n…[truncated]")
                break
            parts.append(block)
            used += len(block)
        return "\n\n---\n\n".join(parts)
