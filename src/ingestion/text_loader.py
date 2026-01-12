from pathlib import Path
from typing import List, Dict


def load_text_documents(directory: str) -> List[Dict[str, str]]:
    base_path = Path(directory)

    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(f"Invalid directory: {directory}")

    documents: List[Dict[str, str]] = []

    for file_path in base_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "doc_id": file_path.stem,
            "text": text,
        })

    return documents
