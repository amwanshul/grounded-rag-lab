from typing import Sequence


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: Sequence[dict], k: int = 5) -> list[dict]:
        pairs = [(query, item["text"]) for item in candidates]
        scores = self.model.predict(pairs)

        ranked = []
        for item, score in zip(candidates, scores):
            copy = dict(item)
            copy["score"] = float(score)
            ranked.append(copy)

        return sorted(ranked, key=lambda x: x["score"], reverse=True)[:k]
