import pytest
from src.embedder import Embedder
from config import EMBEDDING_MODEL


def test_embedder_shape():
    embedder = Embedder(EMBEDDING_MODEL)
    emb = embedder.encode(["hello world", "machine learning"])

    assert emb.shape[0] == 2
    assert emb.ndim == 2
    assert emb.shape[1] > 0