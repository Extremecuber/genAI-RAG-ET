# src/api/server.py

from fastapi import FastAPI

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

    # register routes
    app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
    app.include_router(query.router, prefix="/query", tags=["query"])

    return app


app = create_app()
