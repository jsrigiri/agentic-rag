import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_docs():
    return [
        "Machine learning is a field of artificial intelligence.",
        "FastAPI is used to build APIs in Python.",
        "FAISS is a library for similarity search.",
        "Docker is a containerization platform.",
    ]


@pytest.fixture
def fixed_embeddings(sample_docs):
    np.random.seed(42)
    return np.random.rand(len(sample_docs), 8).astype("float32")


@pytest.fixture
def ranker_regression_xy():
    X = pd.DataFrame({
        "query_len": [20, 30, 25, 18, 16, 40, 28, 24],
        "doc_len": [100, 110, 80, 75, 90, 120, 95, 60],
        "len_diff": [80, 80, 55, 57, 74, 80, 67, 36],
        "cosine_sim": [0.90, 0.20, 0.85, 0.15, 0.75, 0.10, 0.82, 0.31],
        "shared_token_count": [3, 0, 2, 0, 2, 0, 2, 1],
    })
    y = pd.Series([0.95, 0.05, 0.88, 0.10, 0.80, 0.12, 0.84, 0.25])
    return X, y


@pytest.fixture
def ranker_classification_xy():
    X = pd.DataFrame({
        "query_len": [20, 30, 25, 18, 16, 40, 28, 24],
        "doc_len": [100, 110, 80, 75, 90, 120, 95, 60],
        "len_diff": [80, 80, 55, 57, 74, 80, 67, 36],
        "cosine_sim": [0.90, 0.20, 0.85, 0.15, 0.75, 0.10, 0.82, 0.31],
        "shared_token_count": [3, 0, 2, 0, 2, 0, 2, 1],
    })
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
    return X, y