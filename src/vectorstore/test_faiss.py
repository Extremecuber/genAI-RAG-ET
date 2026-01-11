from src.chunking.text_chunker import chunk_text
from src.embeddings.generate_embeddings import generate_embedding
from src.vectorstore.faiss_store import FaissVectorStore


def main() -> None:
    store = FaissVectorStore()

    documents = [
        {
            "doc_id": "doc_1",
            "text": "Hyderabad is very hot during summer and has dry weather conditions.",
        },
        {
            "doc_id": "doc_2",
            "text": "Bangalore has pleasant weather most of the year with moderate temperatures.",
        },
        {
            "doc_id": "doc_3",
            "text": "Delhi experiences extreme winters and summers with large temperature variation.",
        },
    ]

    # Ingest documents (chunk → embed → store)
    for doc in documents:
        chunks = chunk_text(
            text=doc["text"],
            chunk_size=8,
            overlap=2,
        )

        for chunk_id, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)

            metadata = {
                "text": chunk,
                "doc_id": doc["doc_id"],
                "chunk_id": chunk_id,
            }

            store.add(
                embedding=embedding,
                metadata=metadata,
            )

    # Query
    query = "Which city has very hot summers?"
    query_embedding = generate_embedding(query)

    results = store.search(
        query_embedding=query_embedding,
        top_k=5,
    )

    print("\nSearch Results:\n")
    for rank, result in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"  Doc ID   : {result['doc_id']}")
        print(f"  Chunk ID : {result['chunk_id']}")
        print(f"  Score    : {result['score']:.4f}")
        print(f"  Text     : {result['text']}")
        print("-" * 40)


if __name__ == "__main__":
    main()
