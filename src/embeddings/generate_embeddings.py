import requests
from src.config.settings import (
    HF_ROUTER_BASE_URL,
    HF_EMBEDDING_MODEL,
    HF_HEADERS,
)


def generate_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    url = (
        f"{HF_ROUTER_BASE_URL}/"
        f"{HF_EMBEDDING_MODEL}/pipeline/feature-extraction"
    )

    payload = {
        "inputs": text
    }

    response = requests.post(
        url,
        headers=HF_HEADERS,
        json=payload,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"HF API error {response.status_code}: {response.text}"
        )

    data = response.json()

    # HF returns either:
    # - List[float] for single input
    # - List[List[float]] for batch input

    if isinstance(data, list) and data and isinstance(data[0], float):
        return data

    if isinstance(data, list) and data and isinstance(data[0], list):
        return data[0]

    raise RuntimeError(f"Unexpected HF response format: {data}")


if __name__ == "__main__":
    sample_text = "Today is a sunny day and I will get some ice cream."
    embedding = generate_embedding(sample_text)
    print(f"Embedding length: {len(embedding)}")
