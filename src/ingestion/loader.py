import os
from typing import List, Dict

from src.ingestion.text_loader import load_txt
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.docx_loader import load_docx


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def load_documents(data_dir: str) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []

    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)

        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        doc_id = os.path.splitext(filename)[0]

        if ext == ".txt":
            documents.append(load_txt(path, doc_id))

        elif ext == ".pdf":
            documents.append(load_pdf(path, doc_id))

        elif ext == ".docx":
            documents.append(load_docx(path, doc_id))

    return documents
