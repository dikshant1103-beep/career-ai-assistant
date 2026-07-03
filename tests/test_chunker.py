from langchain_core.documents import Document

from src.ingestion.chunker import SemanticChunker


def test_chunker_splits_long_text():
    long_text = ("Paragraph one. " * 50 + "\n\n" + "Paragraph two. " * 50) * 3
    docs = [Document(page_content=long_text, metadata={"source": "x.md", "filename": "x.md"})]
    chunker = SemanticChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.split(docs)
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata.get("chunk_index") is not None
        assert len(c.page_content) <= 400 + 50


def test_chunker_empty_input():
    chunker = SemanticChunker()
    assert chunker.split([]) == []
