from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    title: str
    text: str
    score: float
    citation: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class AnswerRequest(BaseModel):
    query: str = Field(min_length=2)
    k: int = Field(default=5, ge=1, le=20)


class AnswerResponse(BaseModel):
    query: str
    status: str
    answer: str
    citations: list[str]
    evidence: list[SearchResult]
