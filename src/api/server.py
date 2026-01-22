# src/api/server.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path

from src.api.routes import ingest, query


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG API",
        version="0.1.0",
    )

    # health check
    @app.get("/health")
    def health():
        return {"status": "ok"}

    # serve frontend
    @app.get("/", response_class=HTMLResponse)
    def serve_ui():
        index_path = Path("src/web/index.html")
        return index_path.read_text(encoding="utf-8")

    # register routes
    app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
    app.include_router(query.router, prefix="/query", tags=["query"])

    return app


app = create_app()
