# src/api/routes/query.py

from fastapi import APIRouter, HTTPException
from src.api.schemas import QueryRequest, QueryResponse
from src.services.query_service import run_query

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    answer = run_query(payload.query)

    return QueryResponse(answer=answer)
