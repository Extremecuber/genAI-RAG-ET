from typing import List, Dict
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
    ) -> None:
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        search_results: List[Dict],
    ) -> List[Dict]:
        """
        search_results must contain:
        - text
        - doc_id
        - chunk_id
        - score (from FAISS)
        """

        if not search_results:
            return []

        pairs = [
            (query, result["text"])
            for result in search_results
        ]

        ce_scores = self.model.predict(pairs)

        reranked = []
        for result, ce_score in zip(search_results, ce_scores):
            entry = dict(result)
            entry["rerank_score"] = float(ce_score)
            reranked.append(entry)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return reranked[: self.top_n]
