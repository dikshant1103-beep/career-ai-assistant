"""Pytest fixtures.

We deliberately avoid network / Claude API calls in unit tests by using the
``mocker`` fixture from pytest-mock to patch :class:`ClaudeClient`.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Avoid the real .env touching tests (and avoid the assert_api_key check)
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")
os.environ.setdefault("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / ".pytest_chroma"))
os.environ.setdefault("DB_PATH", str(PROJECT_ROOT / ".pytest_db.sqlite"))
os.environ.setdefault("LOG_LEVEL", "WARNING")
