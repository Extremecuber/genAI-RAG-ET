import os
from src.embeddings.generate_embeddings import generate_embedding
from src.vectorstore.faiss_store import FaissVectorStore
from src.chunking.text_chunker import chunk_text

DATA_DIR = "data/ipl"  # folder with txt files

def main():
    store = FaissVectorStore()
    
    # Read all .txt files
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".txt"):
            continue
        doc_id = filename.replace(".txt", "")
        filepath = os.path.join(DATA_DIR, filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        # Optional chunking
        chunks = chunk_text(text, chunk_size=100)  # adjust size if needed
        for chunk_id, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)
            metadata = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk
            }
            store.add(embedding=embedding, metadata=metadata)

    # Example query
    query = "Which IPL team focuses on strong batting?"
    query_embedding = generate_embedding(query)
    results = store.search(query_embedding=query_embedding, top_k=3)

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
