"""Embedding model loader.

We use HuggingFace sentence-transformers via langchain-huggingface. The model
is downloaded once on first use and cached in ~/.cache/huggingface.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    """Return a cached embedding model instance."""
    s = get_settings()
    log.info("Loading embedding model: %s (device=%s)", s.EMBEDDING_MODEL, s.EMBEDDING_DEVICE)
    return HuggingFaceEmbeddings(
        model_name=s.EMBEDDING_MODEL,
        model_kwargs={"device": s.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
