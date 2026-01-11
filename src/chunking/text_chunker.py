from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> List[str]:
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks: List[str] = []

    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += step

    return chunks
