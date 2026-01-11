from src.embeddings.generate_embeddings import generate_embedding


def main() -> None:
    text = "Today is a sunny day and I will get some ice cream."
    embedding = generate_embedding(text)
    print(f"Embedding length: {len(embedding)}")


if __name__ == "__main__":
    main()
