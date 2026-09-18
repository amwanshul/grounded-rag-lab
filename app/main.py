from fastapi import FastAPI, Query

from .pipeline import build_demo_pipeline
from .schemas import AnswerRequest, AnswerResponse, SearchResponse, SearchResult


app = FastAPI(
    title="Grounded RAG Lab",
    version="1.0.0",
    description="Evidence-first retrieval API with fusion, citations and abstention.",
)

pipeline = build_demo_pipeline()


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(pipeline.chunks)}


@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(min_length=2), k: int = Query(default=5, ge=1, le=20)):
    results = pipeline.search(q, k)
    return {"query": q, "results": [SearchResult(**r) for r in results]}


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest):
    result = pipeline.answer(request.query, request.k)
    return AnswerResponse(query=request.query, **result)
