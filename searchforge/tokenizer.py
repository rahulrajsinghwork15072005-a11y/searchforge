"""Tokenization: normalization, stop-word removal, light suffix stemming."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOP_WORD_LIST = """a an and are as at be by for from has have he her his i in is it
its of on or that the their this these those to was were will with what which when
where how who whom why not no but if then than so do does did done can could should
would may might must shall into over under about after before between during""".split()

STOP_WORDS = frozenset(_STOP_WORD_LIST)

_SUFFIXES = [
    ("ational", "ate"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("ization", "ize"),
    ("tional", "tion"),
    ("ations", "ate"),
    ("alities", "al"),
    ("ivity", "ive"),
    ("fully", "ful"),
    ("ingly", ""),
    ("edly", ""),
    ("ies", "y"),
    ("sses", "ss"),
    ("ing", ""),
    ("ed", ""),
    ("ly", ""),
    ("es", ""),
    ("s", ""),
]


def stem(word: str) -> str:
    """Light Porter-flavoured suffix stripper (longest matching suffix wins)."""
    for suffix, replacement in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stemmed = word[: len(word) - len(suffix)] + replacement
            return stemmed or word
    return word


def tokenize(text: str, *, stem_terms: bool = True, drop_stop_words: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stop_words:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if stem_terms:
        tokens = [stem(t) for t in tokens]
    return [t for t in tokens if t]


def raw_tokens(text: str) -> list[str]:
    """Un-stemmed, unfiltered tokens — used for snippet highlighting."""
    return [t for t in re.findall(r"\w+", text.lower())]
