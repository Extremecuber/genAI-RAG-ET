from src.ingestion.text_loader import load_text_documents
from src.chunking.text_chunker import chunk_text
from src.embeddings.generate_embeddings import generate_embedding
from src.vectorstore.faiss_store import FaissVectorStore


import os

PERSIST_PATH = "storage/faiss_store"


def ingest_and_save() -> None:
    store = FaissVectorStore()

    documents = load_text_documents("data/ipl/")
    print(f"Loaded documents: {len(documents)}")

    for doc in documents:
        chunks = chunk_text(
            text=doc["text"],
            chunk_size=200,
            overlap=50,
        )
        print(f"Doc {doc['doc_id']} chunks: {len(chunks)}")

        for chunk_id, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)

            metadata = {
                "doc_id": doc["doc_id"],
                "chunk_id": chunk_id,
                "text": chunk,
            }

            store.add(
                embedding=embedding,
                metadata=metadata,
            )

    print("ABS PERSIST PATH:", os.path.abspath(PERSIST_PATH))
    store.save(PERSIST_PATH)

    
    print("Store persisted to disk.")


def load_and_query() -> None:
    store = FaissVectorStore()
    store.load(PERSIST_PATH)

    query = "Which team has won the most IPL titles?"
    query_embedding = generate_embedding(query)

    results = store.search(query_embedding, top_k=3)

    print("\nSearch Results (Loaded Store):\n")
    for rank, result in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"  Doc ID   : {result['doc_id']}")
        print(f"  Chunk ID : {result['chunk_id']}")
        print(f"  Score    : {result['score']:.4f}")
        print(f"  Text     : {result['text']}")
        print("-" * 40)


def main() -> None:
    ingest_and_save()
    load_and_query()


if __name__ == "__main__":
    main()
