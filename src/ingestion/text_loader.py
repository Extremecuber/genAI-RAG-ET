from pathlib import Path
from typing import Dict


def load_txt(path: str, doc_id: str) -> Dict[str, str]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(path)

    text = file_path.read_text(encoding="utf-8")

    return {
        "doc_id": doc_id,
        "text": text,
    }
