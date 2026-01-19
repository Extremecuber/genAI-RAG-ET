from pathlib import Path
from typing import List

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.docx_loader import load_docx
from src.ingestion.text_loader import load_txt
from src.chunking.text_chunker import chunk_text
from src.embeddings.generate_embeddings import generate_embedding
from src.vectorstore.faiss_store import FaissVectorStore

PERSIST_PATH = "storage/faiss_store"


def ingest_files(file_paths: List[Path]) -> int:
    store = FaissVectorStore()

    total_chunks = 0

    for path in file_paths:
        ext = path.suffix.lower()
        doc_id = path.stem

        if ext == ".txt":
            doc = load_txt(str(path), doc_id)
        elif ext == ".pdf":
            doc = load_pdf(str(path), doc_id)
        elif ext == ".docx":
            doc = load_docx(str(path), doc_id)
        else:
            continue

        print(
    f"[DEBUG] {doc['doc_id']} | "
    f"text_length={len(doc['text'])} | "
    f"preview={doc['text'][:200]!r}"
)

        chunks = chunk_text(
            text=doc["text"],
            chunk_size=200,
            overlap=50,
        )

        for chunk_id, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)

    store.save(PERSIST_PATH)
    return total_chunks
