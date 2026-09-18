from pathlib import Path

from .fusion import reciprocal_rank_fusion
from .ingest import Chunk, load_demo_corpus
from .retrieval import BM25, DenseRetriever, TfidfRetriever


class EvidencePipeline:
    def __init__(
        self,
        chunks: list[Chunk],
        use_dense: bool = False,
        use_reranker: bool = False,
        abstain_threshold: float = 0.02,
    ):
        self.chunks = chunks
        self.bm25 = BM25(chunks)
        self.tfidf = TfidfRetriever(chunks)
        self.dense = DenseRetriever(chunks) if use_dense else None
        self.reranker = None
        if use_reranker:
            from .rerank import CrossEncoderReranker
            self.reranker = CrossEncoderReranker()
        self.abstain_threshold = abstain_threshold

    def search(self, query: str, k: int = 5) -> list[dict]:
        per_retriever = [
            self.bm25.search(query, k=max(k, 10)),
            self.tfidf.search(query, k=max(k, 10)),
        ]
        if self.dense:
            per_retriever.append(self.dense.search(query, k=max(k, 10)))

        fused = reciprocal_rank_fusion(per_retriever)
        candidates = []

        for index, fusion_score in fused[: max(k * 3, 10)]:
            chunk = self.chunks[index]
            candidates.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "text": chunk.text,
                "score": fusion_score,
                "citation": f"[{chunk.title}#{chunk.chunk_id}]",
            })

        if self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates, k=k)
        else:
            candidates = candidates[:k]

        for rank, item in enumerate(candidates, start=1):
            item["rank"] = rank

        return candidates

    def answer(self, query: str, k: int = 5) -> dict:
        evidence = self.search(query, k)
        if not evidence:
            return {
                "status": "abstained",
                "answer": "I don't have enough evidence in the indexed corpus to answer that.",
                "citations": [],
                "evidence": [],
            }

        best = evidence[0]
        if best["score"] < self.abstain_threshold:
            return {
                "status": "abstained",
                "answer": "I don't have enough evidence in the indexed corpus to answer that.",
                "citations": [],
                "evidence": evidence,
            }

        citations = [item["citation"] for item in evidence]
        answer = (
            "Evidence retrieved from the corpus. A generation model can use these "
            "ranked passages as grounded context.\n\n"
            + "\n".join(f"- {item['text']} {item['citation']}" for item in evidence)
        )
        return {
            "status": "grounded_context",
            "answer": answer,
            "citations": citations,
            "evidence": evidence,
        }


def build_demo_pipeline() -> EvidencePipeline:
    root = Path(__file__).resolve().parents[1] / "data" / "docs"
    return EvidencePipeline(load_demo_corpus(root))
