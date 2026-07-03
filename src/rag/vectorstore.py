"""ChromaDB-backed persistent vector store.

Wraps ``langchain_chroma.Chroma`` with conveniences:
- add documents (with auto-generated stable ids based on content hash + source)
- similarity search with optional metadata filter
- reset (drop the collection and rebuild)
- count
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.rag.embedder import get_embedder
from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


def _doc_id(doc: Document) -> str:
    """Stable, deterministic id so re-ingesting the same chunk is idempotent."""
    src = doc.metadata.get("source", "")
    idx = doc.metadata.get("chunk_index", "")
    page = doc.metadata.get("page", "")
    body = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{Path(src).name}:{page}:{idx}:{body}"


class VectorStore:
    """Persistent Chroma collection."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        s = get_settings()
        self.persist_dir: Path = Path(persist_dir) if persist_dir else s.chroma_path
        self.collection_name: str = collection_name or s.CHROMA_COLLECTION
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._store: Chroma | None = None

    # ------------------------------------------------------------------ #
    @property
    def store(self) -> Chroma:
        if self._store is None:
            log.info("Opening Chroma collection '%s' at %s", self.collection_name, self.persist_dir)
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=get_embedder(),
                persist_directory=str(self.persist_dir),
            )
        return self._store

    # ------------------------------------------------------------------ #
    def add_documents(self, docs: Iterable[Document]) -> int:
        docs = list(docs)
        if not docs:
            return 0
        ids = [_doc_id(d) for d in docs]
        self.store.add_documents(documents=docs, ids=ids)
        log.info("Stored %d chunk(s) in vector store", len(docs))
        return len(docs)

    def similarity_search(
        self,
        query: str,
        k: int = 6,
        metadata_filter: Optional[dict] = None,
    ) -> List[Document]:
        return self.store.similarity_search(query, k=k, filter=metadata_filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 6,
        metadata_filter: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k, filter=metadata_filter)

    def count(self) -> int:
        try:
            return self.store._collection.count()  # type: ignore[attr-defined]
        except Exception:
            return 0

    def reset(self) -> None:
        """Delete the entire persistent store on disk."""
        self._store = None
        if self.persist_dir.exists():
            log.warning("Resetting vector store at %s", self.persist_dir)
            for child in self.persist_dir.iterdir():
                if child.name == ".gitkeep":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
