"""Bounded Levenshtein edit distance for typo tolerance."""

from __future__ import annotations


def levenshtein_within(a: str, b: str, max_distance: int) -> int | None:
    """Edit distance if <= max_distance (banded DP), else None. O(min*band)."""
    if abs(len(a) - len(b)) > max_distance:
        return None
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        band_lo = max(1, i - max_distance)
        band_hi = min(len(b), i + max_distance)
        current = [i] + [max_distance + 1] * len(b)
        row_min = current[0]
        for j in range(band_lo, band_hi + 1):
            cost = 0 if ca == b[j - 1] else 1
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            current[j] = value
            row_min = min(row_min, value)
        if row_min > max_distance:
            return None
        previous = current
    distance = previous[len(b)]
    return distance if distance <= max_distance else None


class SpellCorrector:
    def __init__(self, index) -> None:
        self.index = index
        self._vocab = sorted(index.postings.keys())
        self._by_length: dict[int, list[str]] = {}
        for word in self._vocab:
            self._by_length.setdefault(len(word), []).append(word)

    def correction(self, term: str, max_distance: int = 2):
        """Best vocabulary match within edit distance; None when term is fine."""
        if term in self.index.postings:
            return None
        best = None
        best_distance = max_distance + 1
        candidates = set()
        for length in range(len(term) - max_distance, len(term) + max_distance + 1):
            candidates.update(self._by_length.get(length, ()))
        for word in candidates:
            distance = levenshtein_within(term, word, max_distance)
            if distance is None:
                continue
            better = (
                best is None
                or distance < best_distance
                or (distance == best_distance and word < best)
            )
            if better:
                best = word
                best_distance = distance
        return best

    def correct_query_terms(self, terms: list[str], max_distance: int = 2) -> dict[str, str]:
        corrections = {}
        for term in terms:
            fix = self.correction(term, max_distance)
            if fix is not None:
                corrections[term] = fix
        return corrections
