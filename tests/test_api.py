import os
import sys
import pickle
import importlib
import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from config import (
    ARTIFACTS_DIR,
    RANKER_MODEL_PATH,
    RANKER_FEATURES_PATH,
    STORE_PATH,
    TASK_TYPE,
    RANKER_MODEL_TYPE,
)
from src.ranker_model import train_ranker_model


def setup_test_artifacts():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs("vector_store", exist_ok=True)

    docs = [
        "Machine learning is a field of artificial intelligence.",
        "FastAPI is used to build APIs.",
        "FAISS is a similarity search library.",
    ]

    # Match sentence-transformer embedding dimension more closely
    # to reduce surprises if your pipeline assumes a certain shape
    np.random.seed(42)
    embeddings = np.random.rand(len(docs), 384).astype("float32")

    with open(STORE_PATH, "wb") as f:
        pickle.dump((embeddings, docs), f)

    X = pd.DataFrame({
        "query_len": [20, 30, 25, 18, 16, 40],
        "doc_len": [100, 110, 80, 75, 90, 120],
        "len_diff": [80, 80, 55, 57, 74, 80],
        "cosine_sim": [0.90, 0.20, 0.85, 0.15, 0.75, 0.10],
        "shared_token_count": [3, 0, 2, 0, 2, 0],
    })

    if TASK_TYPE == "classification":
        y = pd.Series([1, 0, 1, 0, 1, 0])
    else:
        y = pd.Series([0.95, 0.05, 0.88, 0.10, 0.80, 0.12])

    model, _ = train_ranker_model(
        X,
        y,
        model_type=RANKER_MODEL_TYPE,
        use_gpu=False,
    )

    joblib.dump(model, RANKER_MODEL_PATH)
    joblib.dump(list(X.columns), RANKER_FEATURES_PATH)


def load_fresh_app():
    # Force a fresh import so api.py re-reads newly created artifacts
    if "api" in sys.modules:
        del sys.modules["api"]
    api = importlib.import_module("api")
    return api.app


def test_api_root_and_ask():
    setup_test_artifacts()
    app = load_fresh_app()
    client = TestClient(app)

    root_resp = client.get("/")
    assert root_resp.status_code == 200
    root_body = root_resp.json()
    print("ROOT BODY:", root_body)

    assert isinstance(root_body, dict)
    assert root_body.get("status") == "running"

    ask_resp = client.post("/ask", json={"query": "What is machine learning?"})
    assert ask_resp.status_code == 200

    body = ask_resp.json()
    print("ASK BODY:", body)

    assert isinstance(body, dict)
    assert body.get("query") == "What is machine learning?"
    assert body.get("task_type") == TASK_TYPE
    assert body.get("ranker_model_type") == RANKER_MODEL_TYPE

    assert "answer" in body
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0

    assert "review" in body

    assert "ranked_context" in body
    assert isinstance(body["ranked_context"], list)
    assert len(body["ranked_context"]) >= 1

    first = body["ranked_context"][0]
    print("FIRST RANKED ITEM:", first)

    assert isinstance(first, dict)
    assert "document" in first
    assert "rank_score" in first

    # allow either naming convention if your code differs
    assert ("retrieval_distance" in first) or ("distance" in first)