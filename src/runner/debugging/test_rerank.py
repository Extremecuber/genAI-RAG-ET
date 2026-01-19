from src.vectorstore.faiss_store import FaissVectorStore
from src.embeddings.generate_embeddings import generate_embedding
from src.context.assembler import ContextAssembler
from src.reranking.cross_encoder import CrossEncoderReranker

PERSIST_PATH = "storage/faiss_store"


def main() -> None:
    store = FaissVectorStore.load(PERSIST_PATH)

    query = "Which city has very hot summers?"

    query_embedding = generate_embedding(query)

    # Step 1: high-recall retrieval
    results = store.search(
        query_embedding=query_embedding,
        top_k=20,
    )

    print(f"\nRetrieved {len(results)} candidates (FAISS)\n")

    # Step 2: reranking
    reranker = CrossEncoderReranker(top_n=5)
    reranked_results = reranker.rerank(query, results)

    print("After cross-encoder reranking:\n")
    for rank, r in enumerate(reranked_results, start=1):
        print(f"Rank {rank}")
        print(f"  Doc ID        : {r['doc_id']}")
        print(f"  Chunk ID      : {r['chunk_id']}")
        print(f"  FAISS Score   : {r['score']:.4f}")
        print(f"  Rerank Score  : {r['rerank_score']:.4f}")
        print(f"  Text          : {r['text']}")
        print("-" * 40)

    # Step 3: assemble context
    assembler = ContextAssembler(max_chars=3000)
    context = assembler.assemble(reranked_results)

    print("\nFinal Assembled Context:\n")
    print(context)


if __name__ == "__main__":
    main()
