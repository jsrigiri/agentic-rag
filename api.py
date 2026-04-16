import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from src.embedder import Embedder
from src.retriever import Retriever
from src.agent import Agent
from src.critic import Critic
from config import EMBEDDING_MODEL, TOP_K

app = FastAPI()

with open("vector_store/store.pkl", "rb") as f:
    embeddings, docs = pickle.load(f)

embedder = Embedder(EMBEDDING_MODEL)
retriever = Retriever(embeddings, docs)
agent = Agent()
critic = Critic()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask(request: QueryRequest):
    q_emb = embedder.encode([request.query])[0]
    context = retriever.search(q_emb, TOP_K)

    answer = agent.generate_answer(request.query, context)
    review = critic.review(answer)

    return {
        "query": request.query,
        "context": context,
        "answer": answer,
        "review": review
    }