from docx import Document
from typing import Dict

def load_docx(path: str, doc_id: str) -> Dict[str, str]:
    document = Document(path)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]

    return {
        "doc_id": doc_id,
        "text": "\n".join(paragraphs),
    }
