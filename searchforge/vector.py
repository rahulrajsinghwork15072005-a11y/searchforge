"""Sparse semantic layer: term co-occurrence vectors, cosine hybrid, PRF."""

from __future__ import annotations

import math

from .index import InvertedIndex
from .tokenizer import tokenize


class SemanticVectors:
    """Word-context vectors built from a sliding window over each document.

    Term vectors live sparsely in dicts; document vectors are the weighted
    average of their terms' vectors. Cosine similarity drives the semantic
    signal blended with BM25.
    """

    WINDOW = 8
    MAX_CONTEXT = 40

    def __init__(self, index: InvertedIndex) -> None:
        self.index = index
        self.term_vectors: dict[str, dict[str, float]] = {}
        self._build()

    def _build(self) -> None:

        for doc in self.index.docs.values():
            tokens = tokenize(doc["text"])
            for i, term in enumerate(tokens):
                vec = self.term_vectors.setdefault(term, {})
                lo = max(0, i - self.WINDOW)
                hi = min(len(tokens), i + self.WINDOW + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    ctx = tokens[j]
                    if ctx == term:
                        continue
                    weight = 1.0 / (1.0 + abs(i - j))
                    vec[ctx] = vec.get(ctx, 0.0) + weight
                if len(vec) > self.MAX_CONTEXT * 4:
                    trimmed = sorted(vec.items(), key=lambda kv: -kv[1])[: self.MAX_CONTEXT]
                    self.term_vectors[term] = dict(trimmed)
                    vec = self.term_vectors[term]

    def related_terms(self, term: str, top_n: int = 3) -> list[str]:
        vec = self.term_vectors.get(term)
        if not vec:
            return []
        ranked = sorted(vec.items(), key=lambda kv: -kv[1])
        return [w for w, _ in ranked[:top_n]]

    def expand_terms(self, terms: list[str], top_n: int = 3) -> list[str]:
        expanded = list(terms)
        known = set(terms)
        for term in terms:
            for related in self.related_terms(term, top_n):
                if related not in known:
                    expanded.append(related)
                    known.add(related)
        return expanded

    def document_vector(self, doc_id: str) -> dict[str, float]:
        doc = self.index.docs.get(doc_id)
        if doc is None:
            return {}
        tokens = tokenize(doc["text"])
        agg: dict[str, float] = {}
        for term in tokens:
            for ctx, weight in self.term_vectors.get(term, {}).items():
                agg[ctx] = agg.get(ctx, 0.0) + weight
        norm = math.sqrt(sum(v * v for v in agg.values())) or 1.0
        return {k: v / norm for k, v in agg.items()}

    def query_vector(self, terms: list[str]) -> dict[str, float]:
        agg: dict[str, float] = {}
        for term in terms:
            for ctx, weight in self.term_vectors.get(term, {}).items():
                agg[ctx] = agg.get(ctx, 0.0) + weight
        norm = math.sqrt(sum(v * v for v in agg.values())) or 1.0
        return {k: v / norm for k, v in agg.items()}

    def cosine(self, doc_id: str, query_terms: list[str]) -> float:
        dv = self.document_vector(doc_id)
        qv = self.query_vector(query_terms)
        if not dv or not qv:
            return 0.0
        small, large = (qv, dv) if len(qv) < len(dv) else (dv, qv)
        dot = sum(w * large.get(k, 0.0) for k, w in small.items())
        return dot


def pseudo_relevance_feedback(
    scorer,
    semantic: SemanticVectors | None,
    index: InvertedIndex,
    terms: list[str],
    top_docs: int = 3,
    expansion_terms: int = 5,
) -> list[str]:
    """RM3-lite: pull frequent terms from the current top docs back into the query."""
    initial = sorted(
        scorer.candidates(terms),
        key=lambda d: -scorer.score(d, terms),
    )[:top_docs]
    counts: dict[str, int] = {}
    from .tokenizer import tokenize as tok

    for doc_id in initial:
        doc = index.docs.get(doc_id)
        if not doc:
            continue
        for token in tok(doc["text"]):
            counts[token] = counts.get(token, 0) + 1
    expanded = list(terms)
    known = set(terms)
    added = 0
    for term, _count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if term in known or added >= expansion_terms:
            continue
        if semantic is not None and semantic.term_vectors.get(term) is not None and len(term) < 4:
            continue
        expanded.append(term)
        known.add(term)
        added += 1
    return expanded
