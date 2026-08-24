"""Hash-partitioned shards with a coordinator that fans out GLOBAL IDF."""

from __future__ import annotations

import zlib

from .bm25 import BM25Scorer
from .index import InvertedIndex


def shard_for(doc_id: str, n_shards: int) -> int:
    return zlib.crc32(doc_id.encode("utf-8")) % n_shards


class ShardedIndex:
    """N independent sub-indexes; document routing by crc32(doc_id)."""

    def __init__(self, n_shards: int = 3) -> None:
        if n_shards < 1:
            raise ValueError("need at least one shard")
        self.n_shards = n_shards
        self.shards = [InvertedIndex() for _ in range(n_shards)]

    def add(self, doc_id: str, text: str, fields: dict | None = None) -> None:
        self.shards[shard_for(doc_id, self.n_shards)].add(doc_id, text, fields)

    def remove(self, doc_id: str) -> bool:
        return self.shards[shard_for(doc_id, self.n_shards)].remove(doc_id)

    @property
    def doc_count(self) -> int:
        return sum(s.doc_count for s in self.shards)

    def global_document_frequency(self, term: str) -> int:
        return sum(shard.document_frequency(term) for shard in self.shards)

    def global_idf(self, term: str) -> float:
        import math

        n = max(1, self.doc_count)
        df = self.global_document_frequency(term)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, terms: list[str], limit: int = 10) -> list[tuple[str, float]]:
        """Fan out with per-shard scorers OVERRIDDEN to use global IDF —
        without this, the same term scores differently on each shard and
        merged rankings are wrong."""
        merged: dict[str, float] = {}
        for shard in self.shards:
            scorer = BM25Scorer(shard)
            scorer.idf = lambda term: self.global_idf(term)
            for doc_id in scorer.candidates(terms):
                score = scorer.score(doc_id, terms)
                merged[doc_id] = max(merged.get(doc_id, -1e18), score)
        ranked = sorted(merged.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:limit]

    def save_all(self, base_path: str) -> None:
        for i, shard in enumerate(self.shards):
            shard.save(f"{base_path}.shard{i}")

    @classmethod
    def load_all(cls, base_path: str, n_shards: int) -> ShardedIndex:
        sharded = cls(n_shards)
        for i in range(n_shards):
            sharded.shards[i] = InvertedIndex.load(f"{base_path}.shard{i}")
        return sharded
