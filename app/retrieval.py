from collections import Counter
import math
import re
from dataclasses import asdict
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(c.text) for c in self.chunks]
        self.lengths = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(self.lengths) else 0.0

        df = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q = tokenize(query)
        scored = []
        for i, doc in enumerate(self.docs):
            counts = Counter(doc)
            score = 0.0
            for term in q:
                if term not in counts:
                    continue
                tf = counts[term]
                denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avgdl, 1.0))
                score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1) / denom)
            scored.append((i, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


class TfidfRetriever:
    def __init__(self, chunks: Iterable[Chunk]):
        self.chunks = list(chunks)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        vector = self.vectorizer.transform([query])
        scores = (self.matrix @ vector.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


class DenseRetriever:
    def __init__(self, chunks: Iterable[Chunk], model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.chunks = list(chunks)
        self.model = SentenceTransformer(model_name)
        self.matrix = self.model.encode(
            [c.text for c in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        vector = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.matrix @ vector
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]


def result_payload(chunks: list[Chunk], ranked: list[tuple[int, float]]) -> list[dict]:
    output = []
    for i, score in ranked:
        chunk = chunks[i]
        item = asdict(chunk)
        item["score"] = float(score)
        output.append(item)
    return output
