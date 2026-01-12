import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Any, Tuple

from src.config.settings import EMBEDDING_DIMENSION


class FaissVectorStore:
    """
    FAISS-based vector store with index-aligned metadata.

    Invariant:
    - FAISS index position i <-> metadata[i]
    """

    def __init__(self) -> None:
        # Inner Product index (used with normalized vectors = cosine similarity)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)

        # Index-aligned metadata store
        self.metadata: List[Dict[str, Any]] = []

    def _normalize(self, vector: List[float]) -> np.ndarray:
        arr = np.array(vector, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(arr)
        return arr

    def add(
        self,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Add a single embedding with its metadata.

        Metadata must correspond to exactly one vector.
        """

        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {EMBEDDING_DIMENSION}, got {len(embedding)}"
            )

        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dict")

        vector = self._normalize(embedding)

        # Order matters: FAISS add + metadata append must stay aligned
        self.index.add(vector)
        self.metadata.append(metadata)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store and return metadata + similarity score.
        """

        if self.index.ntotal == 0:
            return []

        if len(query_embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Query embedding dimension mismatch. "
                f"Expected {EMBEDDING_DIMENSION}, got {len(query_embedding)}"
            )

        query_vector = self._normalize(query_embedding)

        scores, indices = self.index.search(query_vector, top_k)

        results: List[Dict[str, Any]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            entry = dict(self.metadata[idx])  # defensive copy
            entry["score"] = float(score)
            results.append(entry)

        return results


    def save(self, path: str) -> None:
        """
        Persist FAISS index and aligned metadata to disk.
        """

        os.makedirs(path, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(path, "index.faiss"),
        )

        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str) -> None:
        """
        Load FAISS index and aligned metadata from disk.
        """

        index_path = os.path.join(path, "index.faiss")
        metadata_path = os.path.join(path, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Persisted FAISS store not found")

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)