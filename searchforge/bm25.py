"""BM25 ranking (Okapi, k1=1.5 b=0.75) over the positional index."""

from __future__ import annotations

import math

from .index import InvertedIndex

K1 = 1.5
B = 0.75


class BM25Scorer:
    def __init__(self, index: InvertedIndex, k1: float = K1, b: float = B) -> None:
        self.index = index
        self.k1 = k1
        self.b = b

    def idf(self, term: str) -> float:
        return self.index.idf(term)

    def score_term(self, doc_id: str, term: str) -> float:
        postings = {p.doc_id: p for p in self.index.postings_for(term)}
        posting = postings.get(doc_id)
        if posting is None:
            return 0.0
        dl = self.index.doc_lengths.get(doc_id, 0)
        avgdl = self.index.avg_doc_length()
        if avgdl <= 0:
            return 0.0
        tf = posting.tf
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (dl / avgdl))
        return math.log(1.0) + self.idf(term) * (numerator / denominator)

    def score(self, doc_id: str, terms: list[str]) -> float:
        return sum(self.score_term(doc_id, t) for t in terms)

    def candidates(self, terms: list[str]) -> set[str]:
        """Union of docs containing any query term."""
        out: set[str] = set()
        for term in terms:
            for posting in self.index.postings_for(term):
                out.add(posting.doc_id)
        return out
