import os
import pickle
from src.embedder import Embedder
from config import EMBEDDING_MODEL

os.makedirs("vector_store", exist_ok=True)

with open("data/documents.txt") as f:
    docs = f.readlines()

embedder = Embedder(EMBEDDING_MODEL)
embeddings = embedder.encode(docs)

with open("vector_store/store.pkl", "wb") as f:
    pickle.dump((embeddings, docs), f)

print("Vector store created")