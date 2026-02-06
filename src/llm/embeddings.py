from sentence_transformers import SentenceTransformer
import numpy as np

class Embeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return np.array(self.model.encode(texts))
