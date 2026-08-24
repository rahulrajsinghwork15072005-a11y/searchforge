"""Search orchestration: BM25 + phrase + filters + typo tolerance + hybrid blend
+ pseudo-relevance feedback + snippets + facets + autocomplete + explain."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from .bm25 import BM25Scorer
from .index import InvertedIndex
from .query import ParsedQuery, parse_query
from .spell import SpellCorrector
from .tokenizer import raw_tokens


@dataclass
class SearchHit:
    doc_id: str
    score: float
    snippet: str
    facets: dict = field(default_factory=dict)
    matched_phrases: int = 0
    bm25: float = 0.0
    semantic: float = 0.0


@dataclass
class SearchResult:
    hits: list[SearchHit]
    total_candidates: int
    corrected: dict[str, str] = field(default_factory=dict)
    expanded_terms: list[str] = field(default_factory=list)


def _phrase_positions(index: InvertedIndex, doc_id: str, phrase_terms: list[str]) -> bool:
    """Positional adjacent-match for a phrase within one document."""
    position_lists = []
    for term in phrase_terms:
        plist = {p.doc_id: p.positions for p in index.postings_for(term)}
        positions = plist.get(doc_id)
        if not positions:
            return False
        position_lists.append(set(positions))
    return any(
        all((start + offset) in position_lists[offset] for offset in range(1, len(phrase_terms)))
        for start in position_lists[0]
    )


def make_snippet(text: str, terms: list[str], radius: int = 60, max_len: int = 220) -> str:
    """HTML-safe snippet centred on the first match, matches wrapped in <mark>."""
    escaped = html.escape(text)
    lowered = text.lower()
    hit_at = -1
    for term in terms:
        pos = lowered.find(term)
        if pos != -1 and (hit_at == -1 or pos < hit_at):
            hit_at = pos
    if hit_at == -1:
        snippet = escaped[:max_len].rstrip()
        return snippet + ("…" if len(escaped) > max_len else "")
    start = max(0, hit_at - radius)
    end = min(len(text), hit_at + max_len - radius)
    piece = text[start:end]
    word_patterns = [r"\w*" + re.escape(t) + r"\w*" for t in set(terms)]
    pattern = re.compile("(" + "|".join(sorted(word_patterns, key=len, reverse=True)) + ")",
                         re.IGNORECASE)
    marked = pattern.sub(lambda m: f"<mark>{html.escape(m.group(1))}</mark>", html.escape(piece))
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + marked + suffix


class Searcher:
    def __init__(
        self,
        index: InvertedIndex,
        semantic=None,
        alpha_bm25: float = 0.75,
        phrase_boost: float = 2.0,
        enable_typo_tolerance: bool = True,
        enable_prf: bool = False,
    ) -> None:
        self.index = index
        self.scorer = BM25Scorer(index)
        self.semantic = semantic
        self.alpha_bm25 = alpha_bm25
        self.phrase_boost = phrase_boost
        self.speller = SpellCorrector(index)
        self.enable_typo_tolerance = enable_typo_tolerance
        self.enable_prf = enable_prf

    # ------------------------------------------------------------------ search

    def search(self, query: str, limit: int = 10, use_hybrid: bool = False) -> SearchResult:
        parsed = parse_query(query)
        corrections: dict[str, str] = {}
        working_terms = list(parsed.terms)
        for phrase in parsed.phrases:
            working_terms.extend(phrase)

        if self.enable_typo_tolerance and working_terms:
            fixes = self.speller.correct_query_terms(working_terms)
            for original, fix in fixes.items():
                corrections[original] = fix
                working_terms = [fix if t == original else t for t in working_terms]

        expanded_terms: list[str] = []
        if self.enable_prf and working_terms:
            expanded_terms = self._prf_expand(parsed, working_terms)

        semantic_terms = working_terms + (
            expanded_terms[len(working_terms):] if expanded_terms else []
        )
        all_terms_for_ranking = working_terms + [
            t for t in expanded_terms if t not in working_terms
        ]

        candidates = self.scorer.candidates(working_terms)
        if parsed.phrases and not candidates:
            for phrase in parsed.phrases:
                candidates |= self.scorer.candidates(phrase)

        pure_phrase_query = bool(parsed.phrases) and not parsed.terms

        results: list[SearchHit] = []
        for doc_id in candidates:
            if not self._passes_filters(doc_id, parsed):
                continue
            if self._has_exclusion(doc_id, parsed):
                continue

            bm25_score = self.scorer.score(doc_id, all_terms_for_ranking)
            phrase_hits = sum(
                1
                for phrase in parsed.phrases
                if _phrase_positions(self.index, doc_id, phrase)
            )
            if pure_phrase_query and phrase_hits == 0:
                continue
            score = bm25_score + phrase_hits * self.phrase_boost

            semantic_component = 0.0
            if use_hybrid and self.semantic is not None:
                cosine = self.semantic.cosine(doc_id, semantic_terms)
                normalized_bm25 = bm25_score / (1.0 + bm25_score)
                semantic_component = cosine
                score = (self.alpha_bm25 * normalized_bm25 + (1 - self.alpha_bm25) * cosine) * (
                    1 + phrase_hits * self.phrase_boost
                )

            doc = self.index.docs[doc_id]
            highlight_terms = set(working_terms)
            for phrase in parsed.phrases:
                highlight_terms.update(phrase)
            results.append(
                SearchHit(
                    doc_id=doc_id,
                    score=score,
                    snippet=make_snippet(doc["text"], sorted(highlight_terms)),
                    facets=self.index.facets_for_doc(doc_id),
                    matched_phrases=phrase_hits,
                    bm25=bm25_score,
                    semantic=semantic_component,
                )
            )

        results.sort(key=lambda h: (-h.score, h.doc_id))
        return SearchResult(
            hits=results[:limit],
            total_candidates=len(results),
            corrected=corrections,
            expanded_terms=expanded_terms,
        )

    # ------------------------------------------------------------- utilities

    def _prf_expand(self, parsed: ParsedQuery, working_terms: list[str]) -> list[str]:
        from .vector import pseudo_relevance_feedback

        return pseudo_relevance_feedback(
            self.scorer, self.semantic, self.index, working_terms
        )

    def _passes_filters(self, doc_id: str, parsed: ParsedQuery) -> bool:
        doc = self.index.docs[doc_id]
        for facet_name, wanted in parsed.filters.items():
            actual = str(doc.get(facet_name, "")).lower()
            if actual != wanted:
                return False
        return True

    def _has_exclusion(self, doc_id: str, parsed: ParsedQuery) -> bool:
        from .tokenizer import tokenize

        doc_text_tokens = set(tokenize(self.index.docs[doc_id]["text"]))
        return any(term in doc_text_tokens for term in parsed.exclusions)

    def autocomplete(self, prefix: str, limit: int = 8) -> list[str]:
        prefix_clean = re.sub(r"[^a-z0-9_]", "", prefix.lower())
        if not prefix_clean:
            return []
        matches = [
            (term, self.index.document_frequency(term))
            for term in self.index.postings
            if term.startswith(prefix_clean)
        ]
        matches.sort(key=lambda pair: (-pair[1], pair[0]))
        return [term for term, _ in matches[:limit]]

    def facet_counts(self, query: str) -> dict[str, dict[str, int]]:
        parsed = parse_query(query)
        terms = parsed.terms + [t for p in parsed.phrases for t in p]
        counts: dict[str, dict[str, int]] = {"type": {}, "site": {}, "tag": {}, "title": {}}
        for doc_id in self.scorer.candidates(terms):
            if self._has_exclusion(doc_id, parsed):
                continue
            for facet_name, value in self.index.facets_for_doc(doc_id).items():
                counts[facet_name][value] = counts[facet_name].get(value, 0) + 1
        return counts


_ = raw_tokens
