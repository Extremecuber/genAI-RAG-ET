# src/context/assembler.py

from typing import List, Dict


class ContextAssembler:
    def __init__(self, max_chars: int = 3000):
        """
        max_chars: hard cap on assembled context length
        """
        self.max_chars = max_chars

    def assemble(self, search_results: List[Dict]) -> str:
        """
        search_results: list of dicts with keys:
            - doc_id
            - chunk_id
            - score
            - text
        """

        context_blocks = []
        total_chars = 0

        for rank, result in enumerate(search_results, start=1):
            block = (
                f"[Rank {rank}]\n"
                f"Document: {result['doc_id']}\n"
                f"Chunk: {result['chunk_id']}\n"
                f"Score: {result['score']:.4f}\n"
                f"Content:\n{result['text']}\n"
            )

            block_len = len(block)

            if total_chars + block_len > self.max_chars:
                break

            context_blocks.append(block)
            total_chars += block_len

        return "\n---\n".join(context_blocks)
