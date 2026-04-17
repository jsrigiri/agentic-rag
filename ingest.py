import os
import pickle
from src.embedder import Embedder
from config import EMBEDDING_MODEL, STORE_PATH

os.makedirs("vector_store", exist_ok=True)

docs = []
with open("data/documents.txt", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            docs.append(line.strip())

embedder = Embedder(EMBEDDING_MODEL)
embeddings = embedder.encode(docs)

with open(STORE_PATH, "wb") as f:
    pickle.dump((embeddings, docs), f)

print("Vector store created:", STORE_PATH)
print("Documents:", len(docs))