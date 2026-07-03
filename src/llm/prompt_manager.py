"""Loads prompt templates from `prompts/`.

Templates are plain `.txt` files using Python `str.format`-style placeholders,
e.g. `{jd}` or `{context}`. The manager caches contents in memory.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class PromptManager:
    """Read + render prompt templates."""

    def __init__(self, prompts_dir: Path | None = None) -> None:
        self.prompts_dir = Path(prompts_dir) if prompts_dir else get_settings().prompts_path
        self._cache: Dict[str, str] = {}

    def load(self, name: str) -> str:
        """Load a template by name (without `.txt`)."""
        if name in self._cache:
            return self._cache[name]
        path = self.prompts_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        text = path.read_text(encoding="utf-8")
        self._cache[name] = text
        return text

    def render(self, name: str, **kwargs) -> str:
        """Load `name` and substitute `{placeholders}`.

        Uses `str.format_map` with a defaulting dict so missing keys become
        an empty string instead of raising — keeps templates forgiving.
        """
        tmpl = self.load(name)
        return tmpl.format_map(_DefaultDict(kwargs))


class _DefaultDict(dict):
    def __missing__(self, key):  # noqa: D401
        log.warning("Missing template variable '%s' — substituting empty string", key)
        return ""


@lru_cache(maxsize=1)
def get_prompt_manager() -> PromptManager:
    return PromptManager()
