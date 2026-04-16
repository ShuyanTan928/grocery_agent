# ============================================================
# tools/synonyms.py
# Lightweight grocery keyword normalization.
#
# Goal: turn user-facing phrases ("pork chop", "ground beef 80/20")
# into a set of candidate tokens we can match against either the
# category keys in mock_prices.json or the item_name field of
# scraped real items.
#
# Intentionally tiny and hand-curated. We prefer deterministic
# behavior over coverage here; the LLM already does the heavy
# lifting upstream for complex phrasings.
# ============================================================

from __future__ import annotations

import re
from typing import Iterable


# Map of canonical token -> list of surface forms that should all resolve to it.
# Keep singular forms; callers lowercase + strip trailing 's' before lookup.
SYNONYM_GROUPS: dict[str, list[str]] = {
    "milk":    ["milk", "whole milk", "2% milk", "skim milk", "dairy milk"],
    "egg":     ["egg", "eggs", "dozen eggs", "large egg"],
    "bread":   ["bread", "loaf", "sandwich bread", "sourdough", "baguette"],
    "butter":  ["butter", "unsalted butter", "salted butter"],
    "cheese":  ["cheese", "cheddar", "mozzarella", "parmesan"],
    "yogurt":  ["yogurt", "greek yogurt"],
    "pork":    ["pork", "pork chop", "pork chops", "pork loin", "pork loin chop",
                "pork tenderloin", "pork shoulder", "pork belly", "pork rib"],
    "chicken": ["chicken", "chicken breast", "chicken thigh", "chicken wing",
                "whole chicken", "chicken drumstick"],
    "beef":    ["beef", "ground beef", "steak", "ribeye", "sirloin", "beef chuck"],
    "fish":    ["fish", "salmon", "tuna", "cod", "tilapia"],
    "shrimp":  ["shrimp", "prawn", "prawns"],
    "apple":   ["apple", "apples", "gala apple", "honeycrisp"],
    "banana":  ["banana", "bananas"],
    "orange":  ["orange", "oranges"],
    "tomato":  ["tomato", "tomatoes", "cherry tomato", "roma tomato"],
    "onion":   ["onion", "onions", "yellow onion", "red onion"],
    "potato":  ["potato", "potatoes", "russet potato"],
    "lettuce": ["lettuce", "romaine", "iceberg", "mixed greens"],
    "pasta":   ["pasta", "spaghetti", "penne", "linguine", "macaroni"],
    "rice":    ["rice", "white rice", "brown rice", "jasmine rice", "basmati rice"],
}


# Reverse index: surface form -> canonical token.
_SURFACE_TO_CANON: dict[str, str] = {}
for canon, forms in SYNONYM_GROUPS.items():
    for f in forms:
        _SURFACE_TO_CANON[f.lower()] = canon


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def canonicalize(phrase: str) -> str | None:
    """Return the canonical token for a whole phrase, or None if unknown.

    Matches in this order:
      1. exact phrase hit in the surface map ("pork chop" -> "pork")
      2. any bigram in the phrase ("bone-in pork chops" -> "pork")
      3. any individual token ("chops" has no canon; "pork" does)
    """
    p = (phrase or "").strip().lower()
    if not p:
        return None

    if p in _SURFACE_TO_CANON:
        return _SURFACE_TO_CANON[p]

    toks = _tokens(p)
    for i in range(len(toks) - 1):
        bigram = f"{toks[i]} {toks[i + 1]}"
        if bigram in _SURFACE_TO_CANON:
            return _SURFACE_TO_CANON[bigram]

    for t in toks:
        if t in _SURFACE_TO_CANON:
            return _SURFACE_TO_CANON[t]
        # Trivial pluralization: "chops" -> "chop"
        if t.endswith("s") and t[:-1] in _SURFACE_TO_CANON:
            return _SURFACE_TO_CANON[t[:-1]]

    return None


def expand_query(query: str) -> list[str]:
    """Return a prioritized list of candidate substrings for matching.

    The first element is always the original query (lowercased and stripped).
    If a canonical token can be derived, it is appended, followed by the
    other surface forms in that group. Duplicates removed, order preserved.
    """
    base = (query or "").strip().lower()
    if not base:
        return []

    seen: set[str] = set()
    result: list[str] = []

    def _add(s: str) -> None:
        if s and s not in seen:
            seen.add(s)
            result.append(s)

    _add(base)

    canon = canonicalize(base)
    if canon:
        _add(canon)
        for form in SYNONYM_GROUPS.get(canon, []):
            _add(form.lower())

    return result


def matches_any(haystack: str, candidates: Iterable[str]) -> bool:
    """True iff any candidate is a substring of haystack (case-insensitive)."""
    h = (haystack or "").lower()
    return any(c and c in h for c in candidates)
