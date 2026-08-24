"""Query language parser.

Supports:
- plain terms          -> ranked OR-candidates under BM25
- "exact phrases"      -> positional match requirement + boost
- -term / -"phrase"    -> exclusions
- type:value site:host tag:x  -> facet filters
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tokenizer import tokenize


@dataclass
class ParsedQuery:
    terms: list[str] = field(default_factory=list)
    raw_terms: list[str] = field(default_factory=list)
    phrases: list[list[str]] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    filters: dict[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.terms and not self.phrases and not self.exclusions and not self.filters


_FIELDS = ("type", "site", "tag")


def parse_query(query: str) -> ParsedQuery:
    parsed = ParsedQuery()
    i = 0
    n = len(query)
    while i < n:
        ch = query[i]
        if ch.isspace():
            i += 1
            continue
        if ch == '"':
            end = query.find('"', i + 1)
            if end == -1:
                phrase_text = query[i + 1 :]
                i = n
            else:
                phrase_text = query[i + 1 : end]
                i = end + 1
            terms = tokenize(phrase_text)
            if terms:
                parsed.phrases.append(terms)
                parsed.raw_terms.extend(
                    _raw_split(phrase_text)
                )
            continue
        if ch == "-":
            rest = query[i + 1 :]
            if rest.startswith('"'):
                end = rest.find('"', 1)
                if end != -1:
                    excl_terms = tokenize(rest[1:end])
                    parsed.exclusions.extend(excl_terms)
                    i = i + 1 + end + 1
                    continue
            j = i + 1
            while j < n and not query[j].isspace():
                j += 1
            word = query[i + 1 : j]
            i = j
            excluded = tokenize(word)
            parsed.exclusions.extend(excluded)
            continue

        j = i
        while j < n and not query[j].isspace():
            j += 1
        word = query[i:j]
        i = j

        matched_field = False
        for field_name in _FIELDS:
            prefix = field_name + ":"
            if word.lower().startswith(prefix):
                value = word[len(prefix) :]
                if value:
                    parsed.filters[field_name] = value.lower()
                matched_field = True
                break
        if matched_field:
            continue

        terms = tokenize(word)
        if not terms:
            continue
        parsed.terms.extend(terms)
        parsed.raw_terms.extend(_raw_split(word))

    return parsed


def _raw_split(text: str) -> list[str]:
    from .tokenizer import raw_tokens

    return [t for t in raw_tokens(text) if t not in ("",)]
