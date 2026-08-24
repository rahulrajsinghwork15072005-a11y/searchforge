"""Positional inverted index with document store, persistence, and facets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class Posting:
    doc_id: str
    tf: int
    positions: list[int] = field(default_factory=list)


class InvertedIndex:
    """term -> [Posting(doc_id, tf, positions)] plus per-doc metadata."""

    def __init__(self) -> None:
        self.postings: dict[str, list[Posting]] = {}
        self.docs: dict[str, dict] = {}
        self.doc_lengths: dict[str, int] = {}
        self.total_tokens = 0
        self.facet_values: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ write

    def add(self, doc_id: str, text: str, fields: dict | None = None) -> None:
        from .tokenizer import tokenize

        if doc_id in self.docs:
            self.remove(doc_id)
        fields = fields or {}
        tokens = tokenize(text)
        self.docs[doc_id] = {"id": doc_id, "text": text, **fields}
        self.doc_lengths[doc_id] = len(tokens)
        self.total_tokens += len(tokens)

        seen: dict[str, list[int]] = {}
        for pos, term in enumerate(tokens):
            seen.setdefault(term, []).append(pos)
        for term, positions in seen.items():
            self.postings.setdefault(term, []).append(
                Posting(doc_id=doc_id, tf=len(positions), positions=positions)
            )

        for facet_name in ("type", "site", "tag"):
            if facet_name in fields:
                self.facet_values.setdefault(facet_name, set()).add(str(fields[facet_name]))

    def remove(self, doc_id: str) -> bool:
        if doc_id not in self.docs:
            return False
        removed_len = self.doc_lengths.pop(doc_id)
        self.total_tokens -= removed_len
        for term in list(self.postings):
            before = len(self.postings[term])
            self.postings[term] = [p for p in self.postings[term] if p.doc_id != doc_id]
            if len(self.postings[term]) < before:
                pass
            if not self.postings[term]:
                del self.postings[term]
        del self.docs[doc_id]
        return True

    # ------------------------------------------------------------------- read

    @property
    def doc_count(self) -> int:
        return len(self.docs)

    def avg_doc_length(self) -> float:
        return (self.total_tokens / self.doc_count) if self.doc_count else 0.0

    def document_frequency(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def idf(self, term: str) -> float:
        n = max(1, self.doc_count)
        df = self.document_frequency(term)
        return math_log((n - df + 0.5) / (df + 0.5) + 1.0)

    def postings_for(self, term: str) -> list[Posting]:
        return self.postings.get(term, [])

    def vocabulary(self) -> list[str]:
        return sorted(self.postings)

    def facets_for_doc(self, doc_id: str) -> dict:
        doc = self.docs.get(doc_id, {})
        return {k: doc[k] for k in ("type", "site", "tag", "title") if k in doc}

    # ------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "docs": self.docs,
            "doc_lengths": self.doc_lengths,
            "total_tokens": self.total_tokens,
            "postings": {
                term: [{"doc": p.doc_id, "tf": p.tf, "pos": p.positions} for p in plist]
                for term, plist in self.postings.items()
            },
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> InvertedIndex:
        index = cls()
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        index.docs = payload["docs"]
        index.doc_lengths = payload["doc_lengths"]
        index.total_tokens = payload["total_tokens"]
        for term, entries in payload["postings"].items():
            index.postings[term] = [
                Posting(doc_id=e["doc"], tf=e["tf"], positions=e["pos"]) for e in entries
            ]
        for doc in index.docs.values():
            for facet_name in ("type", "site", "tag", "title"):
                if facet_name in doc:
                    index.facet_values.setdefault(facet_name, set()).add(str(doc[facet_name]))
        return index


def math_log(x: float) -> float:
    import math

    return math.log(x)
