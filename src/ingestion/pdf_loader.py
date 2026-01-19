from pypdf import PdfReader
from typing import Dict

def load_pdf(path: str, doc_id: str) -> Dict[str, str]:
    reader = PdfReader(path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return {
        "doc_id": doc_id,
        "text": "\n".join(pages),
    }
