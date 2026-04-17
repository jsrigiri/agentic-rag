import pickle
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import (
    EMBEDDING_MODEL,
    TOP_K,
    TASK_TYPE,
    RANKER_MODEL_TYPE,
    RANKER_MODEL_PATH,
    RANKER_FEATURES_PATH,
    STORE_PATH,
)
from src.embedder import Embedder
from src.retriever import Retriever
from src.ranker_features import build_query_doc_features
from src.agent import Agent
from src.critic import Critic

app = FastAPI()

with open(STORE_PATH, "rb") as f:
    embeddings, docs = pickle.load(f)

ranker_model = joblib.load(RANKER_MODEL_PATH)
ranker_feature_columns = joblib.load(RANKER_FEATURES_PATH)

embedder = Embedder(EMBEDDING_MODEL)
retriever = Retriever(embeddings, docs)
agent = Agent()
critic = Critic()


class QueryRequest(BaseModel):
    query: str = Field(..., example="What is machine learning?")


@app.get("/")
def root():
    return {"status": "running", "message": "Use POST /ask or open /docs"}


@app.post("/ask")
def ask(request: QueryRequest):
    q_emb = embedder.encode([request.query])[0]
    initial_hits = retriever.search(q_emb, k=5)

    scored_hits = []
    for hit in initial_hits:
        d_emb = embedder.encode([hit["document"]])[0]
        feats = build_query_doc_features(request.query, hit["document"], q_emb, d_emb)
        X = pd.DataFrame([[feats[c] for c in ranker_feature_columns]], columns=ranker_feature_columns)

        if TASK_TYPE == "classification" and hasattr(ranker_model, "predict_proba"):
            score = float(ranker_model.predict_proba(X)[0, 1])
        else:
            score = float(ranker_model.predict(X)[0])

        scored_hits.append({
            "document": hit["document"],
            "retrieval_distance": hit["distance"],
            "rank_score": score,
        })

    ranked_docs = sorted(scored_hits, key=lambda x: x["rank_score"], reverse=True)[:TOP_K]

    answer = agent.generate_answer(request.query, ranked_docs)
    review = critic.review(answer, task_type=TASK_TYPE)

    return {
        "query": request.query,
        "task_type": TASK_TYPE,
        "ranker_model_type": RANKER_MODEL_TYPE,
        "ranked_context": ranked_docs,
        "answer": answer,
        "review": review,
    }