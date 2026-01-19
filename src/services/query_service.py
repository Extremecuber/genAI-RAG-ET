from src.vectorstore.faiss_store import FaissVectorStore
from src.embeddings.generate_embeddings import generate_embedding
from src.context.assembler import ContextAssembler
from src.reranking.cross_encoder import CrossEncoderReranker
from src.llm.ollama_client import OllamaClient
from src.prompts.rag_prompt import build_rag_prompt

PERSIST_PATH = "storage/faiss_store"


def run_query(user_query: str) -> str:
    store = FaissVectorStore.load(PERSIST_PATH)

    assembler = ContextAssembler(max_chars=3000)
    reranker = CrossEncoderReranker()
    llm = OllamaClient(model="llama3:8b")

    query_embedding = generate_embedding(user_query)
    candidates = store.search(query_embedding, top_k=8)

    if not candidates:
        return "No relevant documents found."

    reranked = reranker.rerank(user_query, candidates)
    context = assembler.assemble(reranked)
    prompt = build_rag_prompt(context=context, query=user_query)

    return llm.generate(prompt)
