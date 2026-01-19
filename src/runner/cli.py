import sys
from pathlib import Path

from src.ingestion.loader import load_documents
from src.chunking.text_chunker import chunk_text
from src.embeddings.generate_embeddings import generate_embedding
from src.vectorstore.faiss_store import FaissVectorStore
from src.context.assembler import ContextAssembler
from src.reranking.cross_encoder import CrossEncoderReranker
from src.llm.ollama_client import OllamaClient
from src.prompts.rag_prompt import build_rag_prompt

PERSIST_PATH = "storage/faiss_store"
DATA_PATH = "data/formattest"


def ingest() -> None:
    print("Starting ingestion...")

    store = FaissVectorStore()
    documents = load_documents(DATA_PATH)

    print(f"Loaded {len(documents)} documents")

    for doc in documents:
        chunks = chunk_text(
            text=doc["text"],
            chunk_size=200,
            overlap=50,
        )

        print(f"Document {doc['doc_id']} → {len(chunks)} chunks")

        for chunk_id, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)

            metadata = {
                "doc_id": doc["doc_id"],
                "chunk_id": chunk_id,
                "text": chunk,
            }

            store.add(embedding, metadata)

    store.save(PERSIST_PATH)
    print("Ingestion complete. Store persisted.")


def query(user_query: str) -> None:
    print("Loading vector store...")
    store = FaissVectorStore.load(PERSIST_PATH)
    print(f"Index size: {store.index.ntotal}")

    assembler = ContextAssembler(max_chars=3000)
    reranker = CrossEncoderReranker()
    llm = OllamaClient(model="llama3:8b")

    print(f"\nUser query: {user_query}")

    query_embedding = generate_embedding(user_query)
    candidates = store.search(query_embedding, top_k=8)

    if not candidates:
        print("No documents retrieved.")
        return

    reranked = reranker.rerank(user_query, candidates)
    context = assembler.assemble(reranked)
    prompt = build_rag_prompt(context=context, query=user_query)

    print("\nGenerating answer...\n")
    answer = llm.generate(prompt)

    print("=== Answer ===\n")
    print(answer)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  ingest")
        print('  query "<your question>"')
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "ingest":
        ingest()

    elif mode == "query":
        if len(sys.argv) < 3:
            print("Please provide a query string.")
            sys.exit(1)

        user_query = " ".join(sys.argv[2:])
        query(user_query)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
