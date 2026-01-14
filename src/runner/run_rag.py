from src.vectorstore.faiss_store import FaissVectorStore
from src.embeddings.generate_embeddings import generate_embedding
from src.context.assembler import ContextAssembler
from src.reranking.cross_encoder import CrossEncoderReranker
from src.llm.ollama_client import OllamaClient
from src.prompts.rag_prompt import build_rag_prompt

PERSIST_PATH = "vectorstore"


def main():
    # 1. Load vector store
    store = FaissVectorStore()
    store.load(PERSIST_PATH)

    # 2. Initialize components
    assembler = ContextAssembler(max_chars=3000)
    reranker = CrossEncoderReranker()
    llm = OllamaClient(model="llama3:8b")

    # 3. User query
    query = "Which IPL team has won the most titles?"

    # 4. Embed query
    query_embedding = generate_embedding(query)

    # 5. Retrieve
    candidates = store.search(query_embedding, top_k=8)

    if not candidates:
        print("No documents retrieved.")
        return

    # 6. Rerank
    reranked = reranker.rerank(query, candidates)

    # 7. Assemble context
    context = assembler.assemble(reranked)

    # 8. Build prompt
    prompt = build_rag_prompt(context=context, query=query)

    # 9. Generate answer
    answer = llm.generate(prompt)

    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
