import pickle
from src.embedder import Embedder
from src.retriever import Retriever
from src.agent import Agent
from src.critic import Critic
from config import EMBEDDING_MODEL, TOP_K

# Load store
with open("vector_store/store.pkl", "rb") as f:
    embeddings, docs = pickle.load(f)

embedder = Embedder(EMBEDDING_MODEL)
retriever = Retriever(embeddings, docs)
agent = Agent()
critic = Critic()

query = "What is machine learning?"

q_emb = embedder.encode([query])[0]
context = retriever.search(q_emb, TOP_K)

answer = agent.generate_answer(query, context)
review = critic.review(answer)

print("Answer:", answer)
print("Review:", review)