from app.ingest import chunk_text
from app.retrieval import BM25, TfidfRetriever


def corpus():
    return chunk_text(
        "demo",
        "Demo",
        "gradient descent uses a learning rate. "
        "websockets provide bidirectional communication. "
        "precision and recall evaluate classifiers.",
        size=80,
        overlap=10,
    )


def test_bm25_finds_relevant_text():
    chunks = corpus()
    results = BM25(chunks).search("learning rate", k=1)
    assert results
    assert "learning rate" in chunks[results[0][0]].text


def test_tfidf_finds_relevant_text():
    chunks = corpus()
    results = TfidfRetriever(chunks).search("bidirectional communication", k=1)
    assert results
    assert "websockets" in chunks[results[0][0]].text
