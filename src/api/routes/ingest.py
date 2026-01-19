from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from pathlib import Path
import shutil
import uuid

from src.services.ingestion_service import ingest_files

router = APIRouter()

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/ingest")
def ingest_endpoint(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    saved_files = []

    try:
        for file in files:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in {".txt", ".pdf", ".docx"}:
                continue

            temp_name = f"{uuid.uuid4()}{suffix}"
            temp_path = UPLOAD_DIR / temp_name

            with temp_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_files.append(temp_path)

        chunks = ingest_files(saved_files)

        return {
            "status": "success",
            "files_ingested": len(saved_files),
            "chunks_created": chunks,
        }

    finally:
        for path in saved_files:
            if path.exists():
                path.unlink()
