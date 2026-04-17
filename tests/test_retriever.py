import numpy as np
from src.retriever import Retriever


def test_retriever_search(sample_docs, fixed_embeddings):
    retriever = Retriever(fixed_embeddings, sample_docs)
    q = np.random.rand(8).astype("float32")

    hits = retriever.search(q, k=2)

    assert len(hits) == 2
    assert "document" in hits[0]
    assert "distance" in hits[0]
    assert "index" in hits[0]
    assert isinstance(hits[0]["distance"], float)