
import faiss
import numpy as np

class FaissVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)

    def add(self, embeddings):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)

    def search(self, query_embedding, top_k=3):
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        return distances, indices

