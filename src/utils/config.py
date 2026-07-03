"""Centralised configuration loaded from `.env` and the environment.

All other modules read settings via :func:`get_settings`. This keeps secrets
out of source code and makes it easy to override values for tests.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from `.env` (Pydantic v2)."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Anthropic ---
    ANTHROPIC_API_KEY: str = Field(default="", description="Claude API key")
    CLAUDE_MODEL: str = Field(default="claude-sonnet-4-6")
    CLAUDE_MAX_TOKENS: int = Field(default=4096)
    CLAUDE_TEMPERATURE: float = Field(default=0.3)

    # --- Embeddings ---
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DEVICE: str = Field(default="cpu")

    # --- Vector store ---
    CHROMA_PERSIST_DIR: str = Field(default="./embeddings")
    CHROMA_COLLECTION: str = Field(default="career_profile")

    # --- Chunking ---
    CHUNK_SIZE: int = Field(default=900)
    CHUNK_OVERLAP: int = Field(default=120)

    # --- Retrieval ---
    RAG_TOP_K: int = Field(default=6)

    # --- Paths ---
    DATA_DIR: str = Field(default="./data")
    PROMPTS_DIR: str = Field(default="./prompts")
    DB_PATH: str = Field(default="./career_assistant.db")

    # --- Logging ---
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: Optional[str] = Field(default="./career_assistant.log")

    # --- Convenience absolute paths ---
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_path(self) -> Path:
        p = (PROJECT_ROOT / self.DATA_DIR).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def prompts_path(self) -> Path:
        p = (PROJECT_ROOT / self.PROMPTS_DIR).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_path(self) -> Path:
        p = (PROJECT_ROOT / self.CHROMA_PERSIST_DIR).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_full_path(self) -> Path:
        return (PROJECT_ROOT / self.DB_PATH).resolve()

    def assert_api_key(self) -> None:
        """Raise a friendly error if no API key is configured."""
        if not self.ANTHROPIC_API_KEY or self.ANTHROPIC_API_KEY.startswith("sk-ant-replace"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is missing. Copy `.env.example` to `.env` and "
                "set ANTHROPIC_API_KEY from https://console.anthropic.com/settings/keys"
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
