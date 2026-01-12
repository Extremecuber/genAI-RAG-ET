from src.ingestion.text_loader import load_text_documents
from src.chunking.text_chunker import chunk_text
from src.embeddings.generate_embeddings import generate_embedding
from src.vectorstore.faiss_store import FaissVectorStore
from src.context.assembler import ContextAssembler


def main() -> None:
    store = FaissVectorStore()
    assembler = ContextAssembler(max_chars=3000)

    # Load documents from data directory
    documents = load_text_documents("data")

    # Ingest documents
    for doc in documents:
        chunks = chunk_text(
            text=doc["text"],
            chunk_size=200,
            overlap=50,
        )

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

    # Query
    query = "Which team has won the most IPL titles?"
    query_embedding = generate_embedding(query)

    search_results = store.search(
        query_embedding=query_embedding,
        top_k=5,
    )

    # Assemble context
    context = assembler.assemble(search_results)

    print("\n=== ASSEMBLED CONTEXT ===\n")
    print(context)


if __name__ == "__main__":
    main()
