import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        emb = self.model.encode(texts)
        return np.asarray(emb, dtype=float)