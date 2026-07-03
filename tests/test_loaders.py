from pathlib import Path

from src.ingestion.loaders import load_document


def test_load_markdown(tmp_path: Path):
    p = tmp_path / "note.md"
    p.write_text("# Hello\n\nThis is a test resume.", encoding="utf-8")
    docs = load_document(p, category="resume")
    assert len(docs) == 1
    assert "Hello" in docs[0].page_content
    assert docs[0].metadata["category"] == "resume"
    assert docs[0].metadata["filename"] == "note.md"


def test_unsupported_extension(tmp_path: Path):
    p = tmp_path / "x.xyz"
    p.write_text("ignored", encoding="utf-8")
    assert load_document(p) == []
