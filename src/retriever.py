import faiss
import numpy as np


class Retriever:
    def __init__(self, embeddings, documents):
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.asarray(embeddings, dtype="float32"))
        self.documents = documents

    def search(self, query_embedding, k=3):
        q = np.asarray([query_embedding], dtype="float32")
        distances, indices = self.index.search(q, k)
        return [
            {
                "document": self.documents[i],
                "distance": float(distances[0][j]),
                "index": int(i),
            }
            for j, i in enumerate(indices[0])
        ]