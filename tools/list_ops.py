"""
Leaf helpers that mutate a shopping list or a priced plan, shared
between the tool registry (`agent/tools.py`) and tests.

Extracted from the old agent.py so `tools/` stays the source of truth
for business logic and the agent layer only contains orchestration
glue.

Public surface:
    # query parsing
    items_to_query_strings(items)               -> list[str]
    # plan arithmetic
    recompute_plan_totals(plan_dict)            -> None   (in-place)
    # remove matching
    tokens(text)                                -> list[str]
    item_matches_target(name, target_tokens)    -> bool
    remove_match_level(name, target_norm, toks) -> int    (0 | 1 | 2)
    clean_remove_target(raw)                    -> str
    # preference application
    apply_preferred_stores(plan, preferred)     -> dict
    apply_avoid_stores(plan, avoid)             -> dict
"""

from __future__ import annotations

import re

from tools.price_optimizer import (
    find_at_store_in_cache,
    find_cheapest_in_cache_excluding,
    load_stores,
)
from tools.synonyms import expand_query, matches_any


# ------------------------- query shape --------------------------------

def items_to_query_strings(items: list[dict]) -> list[str]:
    """Convert structured item dicts to search strings for the price optimizer.
    E.g. {"name": "chicken breast", "quantity": 2, "unit": "lb"} -> "chicken breast 2 lb"
    """
    queries: list[str] = []
    for item in items:
        parts = [item["name"]]
        if item.get("quantity") and item.get("unit"):
            parts.append(f"{item['quantity']} {item['unit']}")
        elif item.get("unit"):
            parts.append(item["unit"])
        queries.append(" ".join(parts))
    return queries


# ------------------------- plan arithmetic ----------------------------

def recompute_plan_totals(shopping_plan: dict) -> None:
    """Rebuild total_cost, store_ids and prune empty stores in place."""
    plan = shopping_plan["plan"]
    all_stores = load_stores()

    for sid in list(plan.keys()):
        if not plan[sid]:
            del plan[sid]

    shopping_plan["store_ids"] = list(plan.keys())
    shopping_plan["stores_meta"] = {
        sid: all_stores[sid] for sid in shopping_plan["store_ids"] if sid in all_stores
    }
    shopping_plan["total_cost"] = round(
        sum(i["price"] for items in plan.values() for i in items), 2
    )


# ------------------------- remove matching ----------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _singularize(token: str) -> str:
    """Very small plural stripper for grocery nouns.
    Strips a trailing 's' when it looks like a simple plural marker so
    'oranges' matches 'orange' and 'eggs' matches 'egg'. Skips short
    tokens, 'ss' (class/pass), 'us' (asparagus/citrus), and 'is'
    (analysis) suffixes that aren't plurals."""
    if len(token) < 4:
        return token
    if token.endswith(("ss", "us", "is")):
        return token
    if token.endswith("s"):
        return token[:-1]
    return token


def _normalize_phrase(text: str) -> str:
    """Singular-normalized whitespace-joined phrase. Used for the
    strict level-2 match in remove_match_level — 'oranges' → 'orange',
    'orange juice' → 'orange juice'."""
    return " ".join(_singularize(t) for t in _TOKEN_RE.findall((text or "").lower()))

_REMOVE_TARGET_HEAD_STRIP = re.compile(
    r"^(?:the|that|those|my|our|a|an|any|some|all\s+(?:the|of))\s+",
    re.I,
)
_REMOVE_TARGET_TAIL_STRIP = re.compile(
    r"\s*(?:"
    r"from\s+(?:the\s+)?(?:list|plan|cart|order)|"
    r"in\s+(?:the\s+)?(?:list|plan|cart)|"
    r"off\s+(?:the\s+)?(?:list|plan|cart)|"
    r"out\s+of\s+(?:the\s+)?(?:list|plan|cart)|"
    r"please|pls|thanks|thank\s+you|thx|anymore|any\s+more|"
    r"right\s+now|for\s+now|this\s+time|today|tonight|tomorrow"
    r")\s*[.!?]*\s*$",
    re.I,
)

REMOVE_TARGET_STOPWORDS = {
    "", "the", "that", "this", "it", "them", "any", "anything",
    "more", "else", "other", "another", "one", "some", "all",
}


def tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens with simple plural stripping, so
    'eggs'/'egg' and 'oranges'/'orange' are treated as the same token
    during loose target matching."""
    return [_singularize(t) for t in _TOKEN_RE.findall((text or "").lower())]


def item_matches_target(name: str, target_tokens: list[str]) -> bool:
    """True if EVERY token in target_tokens appears (whole-word) in name.
    Empty target -> False."""
    if not target_tokens:
        return False
    name_tokens = set(tokens(name))
    return all(t in name_tokens for t in target_tokens)


def remove_match_level(name: str, target_norm: str, target_tokens: list[str]) -> int:
    """0 = no match, 1 = loose token-subset, 2 = exact normalized name.
    Exact matches take precedence to avoid `orange` nuking `orange juice`.
    The exact check is singular/plural-tolerant: removing 'orange' from a
    list containing 'oranges' is a level-2 hit (not level-1), so it wins
    over a loose match on 'orange juice'."""
    if not name:
        return 0
    name_norm = name.strip().lower()
    if name_norm == target_norm:
        return 2
    if _normalize_phrase(name) == _normalize_phrase(target_norm):
        return 2
    if item_matches_target(name, target_tokens):
        return 1
    return 0


def clean_remove_target(raw: str) -> str:
    """Trim articles / trailing prepositional phrases off a target.
    Returns an empty string if nothing substantive is left."""
    if not raw:
        return ""
    out = raw.strip().strip(",.!?")
    prev = None
    while prev != out:
        prev = out
        out = _REMOVE_TARGET_HEAD_STRIP.sub("", out).strip()
    out = _REMOVE_TARGET_TAIL_STRIP.sub("", out).strip()
    return out.strip(" .,!?\"'")


# ------------------------- store preferences --------------------------

def apply_preferred_stores(shopping_plan: dict, preferred_stores: dict) -> dict:
    """Force items whose name matches a preferred_stores key to be
    sourced from the user's preferred store (if that store carries the
    item). Any unhonored requests land in `unfulfilled_preferences`."""
    if not preferred_stores:
        shopping_plan.setdefault("unfulfilled_preferences", [])
        return shopping_plan

    all_stores = load_stores()
    unfulfilled: list[dict] = []

    for pref_item, preferred_ids in preferred_stores.items():
        if not preferred_ids:
            continue

        candidates = expand_query(pref_item) or [pref_item.lower()]

        current_sid = None
        current_entry = None
        for sid, entries in shopping_plan["plan"].items():
            for e in entries:
                if matches_any(e["item"], candidates):
                    current_sid = sid
                    current_entry = e
                    break
            if current_entry:
                break

        if current_entry is None:
            continue

        if current_sid in preferred_ids:
            continue

        placed = False
        for target_sid in preferred_ids:
            alt = find_at_store_in_cache(pref_item, target_sid, all_stores)
            if alt is None:
                continue

            shopping_plan["plan"][current_sid].remove(current_entry)
            shopping_plan["plan"].setdefault(target_sid, []).append({
                "item": alt["item_name"],
                "price": alt["item_price"],
                "store_display": alt.get("store", target_sid),
                "url": alt.get("url"),
                "source": alt.get("_source", "cache"),
            })
            placed = True
            break

        if not placed:
            unfulfilled.append({
                "item": pref_item,
                "preferred_stores": list(preferred_ids),
                "reason": "not available at any preferred store",
            })

    shopping_plan["unfulfilled_preferences"] = unfulfilled
    recompute_plan_totals(shopping_plan)
    return shopping_plan


def apply_avoid_stores(shopping_plan: dict, avoid_stores: dict) -> dict:
    """Move items away from stores the user wants to avoid, picking the
    next cheapest non-avoided store from the cached catalog."""
    if not avoid_stores:
        return shopping_plan

    all_stores = load_stores()
    pref_candidates: dict[str, list[str]] = {
        k: expand_query(k) or [k.lower()] for k in avoid_stores
    }

    for store_id in list(shopping_plan["plan"].keys()):
        items_in_store = shopping_plan["plan"][store_id]
        to_move = [
            e for e in items_in_store
            if any(
                store_id in avoided and matches_any(e["item"], pref_candidates[pref_item])
                for pref_item, avoided in avoid_stores.items()
            )
        ]

        for item_entry in to_move:
            items_in_store.remove(item_entry)

            matched_pref = next(
                (k for k in avoid_stores
                 if matches_any(item_entry["item"], pref_candidates[k])),
                None,
            )
            if matched_pref is None:
                continue
            avoided = list(avoid_stores.get(matched_pref) or [])
            if store_id not in avoided:
                avoided.append(store_id)

            alt = find_cheapest_in_cache_excluding(matched_pref, avoided, all_stores)
            if alt is None:
                continue

            target_sid = alt.get("_store_id")
            if not target_sid:
                continue
            shopping_plan["plan"].setdefault(target_sid, []).append({
                "item": alt["item_name"],
                "price": alt["item_price"],
                "store_display": alt.get("store", target_sid),
                "url": alt.get("url"),
                "source": alt.get("_source", "cache"),
            })

    recompute_plan_totals(shopping_plan)
    return shopping_plan
