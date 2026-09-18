# Grounded RAG Lab

Evidence-first retrieval for document question answering.

This project is a deliberately transparent RAG system built around **chunking, retrieval, fusion, reranking, citation tracking, abstention, and evaluation**.

It is designed as a portfolio-grade reference implementation rather than a chatbot wrapper.

## What it demonstrates

- Document ingestion and deterministic chunking
- BM25 lexical retrieval
- TF-IDF retrieval baseline
- Optional dense retrieval with Sentence Transformers
- Reciprocal Rank Fusion (RRF)
- Optional cross-encoder reranking
- Source-aware citations
- "Not enough evidence" abstention
- Golden-set evaluation with Recall@K and MRR
- FastAPI service with /search, /answer and /health
- Deterministic tests that run without an LLM

## Architecture

~~~mermaid
flowchart LR
    A[Documents] --> B[Chunker]
    B --> C[Chunk Store]
    C --> D1[BM25]
    C --> D2[TF-IDF]
    C --> D3[Dense Embeddings]
    D1 --> E[RRF Fusion]
    D2 --> E
    D3 --> E
    E --> F[Cross-Encoder Reranker]
    F --> G{Evidence sufficient?}
    G -- No --> H[Abstain]
    G -- Yes --> I[Context + Citations]
    I --> J[Answer Generator]
~~~

## Quick start

~~~bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
~~~

Open the API docs at http://127.0.0.1:8000/docs.

## Search

~~~bash
curl "http://127.0.0.1:8000/search?q=gradient%20descent&k=5"
~~~

## Evaluate

~~~bash
python -m evaluation.run_eval
~~~

The evaluation script reports Recall@5 and MRR. Re-run it after changing chunking, retrieval weights or reranking.

## Design decisions

### Why RRF?

Different retrievers produce scores on incompatible scales. RRF combines rankings rather than pretending BM25, TF-IDF and dense cosine scores are directly comparable.

### Why abstention?

A RAG system should not be forced to answer every question. If the best evidence score is below a configurable threshold, the API returns an explicit abstention signal.

### Why citations?

Every retrieved chunk retains its document ID, title and character span. The answer layer can therefore cite evidence instead of returning an opaque block of generated text.

## Environment

Dense retrieval and LLM generation are optional. The repository remains usable in retrieval-only mode with no API key.

~~~text
RAG_DENSE=0
RAG_RERANK=0
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=
~~~

## Project structure

~~~text
app/
  main.py              # FastAPI service
  schemas.py           # request/response models
  retrieval.py         # BM25, TF-IDF, optional dense retrieval
  fusion.py            # reciprocal-rank fusion
  rerank.py            # optional cross-encoder reranker
  pipeline.py          # end-to-end evidence pipeline
  ingest.py            # document loading + chunking
data/docs/             # small public demo corpus
evaluation/            # golden set + retrieval evaluation
tests/                 # deterministic unit tests
~~~

## Roadmap

- [ ] Persistent vector/index storage
- [ ] Larger dense-vs-sparse benchmark
- [ ] LLM answer generation with structured citations
- [ ] Query rewriting and retrieval diagnostics
- [ ] RAG regression gate in CI
- [ ] Latency and cost tracing

## License

MIT
