from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    Guarantees:
    - Returns at least one chunk for non-empty text
    - overlap < chunk_size
    """

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()

    # ✅ If text is smaller than chunk size, return single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks
