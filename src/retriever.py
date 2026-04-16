import faiss
import numpy as np

class Retriever:
    def __init__(self, embeddings, documents):
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.documents = documents

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(
            np.array([query_embedding]), k
        )
        return [self.documents[i] for i in indices[0]]