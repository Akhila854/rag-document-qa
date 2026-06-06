import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag.chunker import chunk_text


def test_chunker_returns_chunks():
    sample_text = "This is a test sentence. " * 100
    chunks = chunk_text(sample_text)
    assert len(chunks) > 0


def test_chunker_respects_size():
    sample_text = "Word " * 500
    chunks = chunk_text(sample_text)
    for chunk in chunks:
        assert len(chunk) <= 600  # chunk_size + small buffer


def test_empty_text():
    chunks = chunk_text("")
    assert isinstance(chunks, list)