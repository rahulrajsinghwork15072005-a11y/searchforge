"""Pairwise RankNet-style linear learning-to-rank with click-through signals."""

from __future__ import annotations

import json
import math
import random


class FeatureExtractor:
    """Turns a (query, document) pair into a dense feature vector."""

    def __init__(self, searcher) -> None:
        self.searcher = searcher

    def features(self, query: str, doc_id: str) -> list[float]:
        from .query import parse_query

        parsed = parse_query(query)
        terms = parsed.terms + [t for p in parsed.phrases for t in p]
        bm25 = self.searcher.scorer.score(doc_id, terms)
        normalized_bm25 = bm25 / (1.0 + bm25)

        phrase_hits = 0
        for phrase_terms in parsed.phrases:
            from .search import _phrase_positions

            if _phrase_positions(self.searcher.index, doc_id, phrase_terms):
                phrase_hits += 1

        doc_text = self.searcher.index.docs[doc_id]["text"].lower()
        title_like = sum(1 for t in terms if t in doc_text[:60])

        coverage = sum(
            1
            for t in set(terms)
            if any(p.doc_id == doc_id for p in self.searcher.index.postings_for(t))
        ) / max(1, len(set(terms)))

        semantic = 0.0
        if self.searcher.semantic is not None:
            semantic = self.searcher.semantic.cosine(doc_id, terms)

        return [
            1.0,
            normalized_bm25,
            min(1.0, phrase_hits),
            title_like / max(1, len(set(terms))),
            coverage,
            semantic,
        ]


class RankNetLinear:
    """Pairwise logistic model: P(rel(d1)>rel(d2)) = sigmoid(w·f1 - w·f2)."""

    def __init__(self, n_features: int) -> None:
        self.weights = [0.0] * n_features

    def score(self, features: list[float]) -> float:
        return sum(w * f for w, f in zip(self.weights, features))

    def train(
        self,
        samples: dict[str, dict[str, int]],
        extract,
        epochs: int = 200,
        lr: float = 0.05,
        seed: int = 7,
    ) -> dict:
        """samples: {query: {doc_id: grade 0..3}}."""
        rng = random.Random(seed)
        pairs = []
        for _query, judgments in samples.items():
            docs = list(judgments)
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    a, b = docs[i], docs[j]
                    ga, gb = judgments[a], judgments[b]
                    if ga != gb:
                        pairs.append((_query, a, b, ga > gb))
        if not pairs:
            return {"pairs": 0, "epochs": 0}

        cache: dict[tuple[str, str], list[float]] = {}
        last_loss = 0.0
        for _epoch in range(epochs):
            rng.shuffle(pairs)
            total_loss = 0.0
            grad = [0.0] * len(self.weights)
            for query_id, winner, loser, winner_is_a in pairs:
                fw = self._feat(cache, query_id, winner, extract)
                fl = self._feat(cache, query_id, loser, extract)
                diff = self._dot(fw) - self._dot(fl)
                sigma = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, diff))))
                target = 1.0 if winner_is_a else 0.0
                loss = -(
                    target * math.log(sigma + 1e-12)
                    + (1 - target) * math.log(1 - sigma + 1e-12)
                )
                total_loss += loss
                sign = 1.0 if winner_is_a else -1.0
                coefficient = (sigma - target) * sign
                for k in range(len(self.weights)):
                    grad[k] += coefficient * (fw[k] - fl[k])
            for k in range(len(self.weights)):
                self.weights[k] -= lr * grad[k] / len(pairs)
            last_loss = total_loss / len(pairs)
        return {"pairs": len(pairs), "epochs": epochs, "loss": round(last_loss, 5)}

    def _dot(self, features: list[float]) -> float:
        return sum(w * f for w, f in zip(self.weights, features))

    def _feat(self, cache, query: str, doc_id: str, extract) -> list[float]:
        key = (query, doc_id)
        if key not in cache:
            cache[key] = extract(query, doc_id)
        return cache[key]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"weights": self.weights}, f)

    @classmethod
    def load(cls, path: str) -> RankNetLinear:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        model = cls(len(payload["weights"]))
        model.weights = payload["weights"]
        return model


def ctr_boost(clicks: dict[str, int], impressions: dict[str, int], prior: float = 5.0) -> float:
    """Smoothed CTR for one doc: (clicks + prior*global_ctr) / (impressions + prior)."""
    total_clicks = sum(clicks.values())
    total_impressions = max(1, sum(impressions.values()))
    global_ctr = total_clicks / total_impressions
    doc_clicks = clicks.get("_doc", 0) if "_doc" in clicks else 0
    doc_views = impressions.get("_doc", 0) if "_doc" in impressions else 0
    return (doc_clicks + prior * global_ctr) / (doc_views + prior)


_ = json, math, random
