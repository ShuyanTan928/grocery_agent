# ============================================================
# tools/price_optimizer.py
# Given a shopping list, finds the cheapest store for each item
# (from the real scraped caches in data/price_cache/) and returns
# an optimized buy plan that minimizes total cost.
#
# Public entry points used by the agent:
#   - optimize_shopping_list(items)   — main planner (cache-only; optional
#       per-item LLM pick when USE_LLM_MAIN_OPTIMIZER=true)
#   - find_cheapest_in_cache(query, stores)
#   - find_at_store_in_cache(query, store_id, stores)
#   - find_cheapest_in_cache_excluding(query, exclude, stores)
#
# Legacy mock-data helpers (still imported by some unit tests, not
# used in the live agent flow):
#   - find_cheapest(query, price_data)
#   - find_at_store(query, store_id, price_data)
#   - load_prices()
# ============================================================

import json
import re
from pathlib import Path
from config.settings import MOCK_DATA_DIR, USE_LLM_MAIN_OPTIMIZER, USE_MOCK_DATA
from tools.synonyms import expand_query
from tools.product_search import search_products_ranked


# Strip trailing "<qty> <unit>" or bare unit tokens from an item query so
# cache lookups can match on the item name portion. items_to_query_strings
# produces things like "pringles 2 bag" / "chicken breast 2 lb".
_UNIT_TOKENS = {
    "lb", "lbs", "pound", "pounds", "oz", "ounce", "ounces",
    "gallon", "gallons", "gal", "qt", "quart", "pint", "pt",
    "ml", "l", "liter", "liters", "kg", "g",
    "bag", "bags", "pack", "packs", "bottle", "bottles",
    "box", "boxes", "can", "cans", "jar", "jars",
    "dozen", "dz", "each", "ea", "piece", "pieces",
    "loaf", "loaves", "stick", "sticks",
}


def _strip_qty_unit(query: str) -> str:
    """Return a cache-friendly variant of `query` with trailing quantity
    and/or unit tokens removed. Falls back to the original if nothing can
    be stripped."""
    q = (query or "").strip()
    if not q:
        return q
    # repeatedly peel off trailing number / unit tokens
    changed = True
    while changed:
        changed = False
        m = re.match(r"^(.*?)[\s,]+(\d+(?:\.\d+)?)\s*$", q)
        if m:
            q = m.group(1).strip()
            changed = True
            continue
        m = re.match(r"^(.*?)[\s,]+([a-zA-Z]+)\s*$", q)
        if m and m.group(2).lower() in _UNIT_TOKENS:
            q = m.group(1).strip()
            changed = True
            continue
    return q or query


def load_prices() -> dict:
    """Load product price data from mock file or live scraper."""
    if USE_MOCK_DATA:
        path = Path(MOCK_DATA_DIR) / "mock_prices.json"
        with open(path) as f:
            return json.load(f)
    else:
        raise NotImplementedError("Live scraper not implemented yet")


def load_stores() -> dict:
    """
    Load store metadata indexed by store id.
    Also builds a display_name -> id lookup for matching price data.
    Returns: { store_id: store_dict }
    """
    path = Path(MOCK_DATA_DIR) / "mock_stores.json"
    with open(path) as f:
        data = json.load(f)
    return {s["id"]: s for s in data["stores"]}


def build_display_name_index(stores: dict) -> dict:
    """
    Build a reverse lookup: display_name (lowercase) -> store_id.
    e.g. "trader joe's" -> "trader_joes_shadyside"
    Used to map the store name strings in price data back to store IDs.
    """
    return {s["display_name"].lower(): sid for sid, s in stores.items()}


def find_cheapest(item_query: str, price_data: dict) -> dict | None:
    """
    Find the cheapest store entry for a given item query.
    Matches item_query against the category keys in price_data["items"]
    using simple substring matching, then re-ranks the matched entries
    by tier-relevance to the FULL user query and returns the cheapest
    tier-0/1 entry.

    Returning None when no entry is strictly relevant lets the caller
    fall back to the real-cache lookup (e.g. a query for "chicken wings"
    in mock matches the "chicken" category but only contains "chicken
    breast" entries — we want to defer to the cache, which has actual
    wings).
    """
    from tools.product_search import _relevance_tier

    items_db = price_data.get("items", {})
    matched_entries = _match_category(item_query, items_db)
    if not matched_entries:
        return None

    tier_query = _strip_qty_unit(item_query) or item_query
    scored = [
        (_relevance_tier(tier_query, e.get("item_name", "")), float(e["item_price"]), e)
        for e in matched_entries
    ]
    scored.sort(key=lambda x: (x[0], x[1]))
    best_tier = scored[0][0]
    if best_tier > 1:
        return None
    return scored[0][2]


# --- Real-cache fallback ---------------------------------------------------

def _cache_entry_to_mock_shape(cached: dict, stores: dict) -> dict | None:
    """Coerce a `search_products_ranked` result into the mock entry shape so
    it's a drop-in substitute for `find_cheapest` callers.

    The store display_name is resolved via `stores[store_id].display_name`
    so downstream `build_display_name_index` lookups continue to work.
    """
    store_id = cached.get("store_id")
    price = cached.get("item_price")
    if not store_id or price is None:
        return None
    store_meta = stores.get(store_id) or {}
    display = store_meta.get("display_name") or cached.get("store") or store_id
    return {
        "store": display,
        "location": store_meta.get("address", ""),
        "item_name": cached.get("item_name") or "",
        "item_price": float(price),
        "url": cached.get("url"),
        "_source": "cache",
        "_store_id": store_id,
    }


def _cache_strict_hits(search_query: str, tier_query: str,
                       stores_filter: list[str] | None = None) -> list[dict]:
    """Run a cache search and keep only tier-0/1 hits, where the tier is
    graded against `tier_query` (NOT `search_query`).

    This lets us widen the search (e.g. just "pringles") while still
    requiring the ORIGINAL multi-word query ("pringles chips") to match
    as a whole phrase or bag-of-words. That way "moon cheese" narrowed
    to "cheese" still fails the tier check (no "moon" in any cheese
    item), but "pringles chips" narrowed to "pringles" passes (Pringles
    items contain both "pringles" and "chips").
    """
    from tools.product_search import search_products, _relevance_tier
    raw = search_products(
        search_query,
        store_ids=stores_filter,
        include_mock=False,
        sort_by="none",
        expand_synonyms=True,
    )
    hits: list[dict] = []
    for it in raw:
        tier = _relevance_tier(tier_query, it.get("item_name") or "")
        if tier <= 1 and it.get("item_price") is not None:
            hits.append({**it, "_relevance_tier": tier})
    hits.sort(key=lambda x: (
        x.get("_relevance_tier", 2),
        x.get("item_price") or 0,
    ))
    return hits


def _cache_query_cascade(item_query: str) -> list[str]:
    """Progressively looser search-query variants. Each is used to pull
    candidates; the tier check always uses the stripped full query."""
    stripped = _strip_qty_unit(item_query)
    variants = [item_query, stripped]
    # Try each token >=4 chars individually, longest first, as a final widen.
    tokens = [t for t in re.split(r"\s+", stripped) if len(t) >= 4]
    tokens.sort(key=len, reverse=True)
    variants.extend(tokens)
    return _dedup(variants)


def find_cheapest_in_cache(item_query: str, stores: dict) -> dict | None:
    """Cache-backed price lookup. Searches the ~21k cached SKUs and
    returns the cheapest tier-0/1 (strict word-contains) match against
    the stripped full query, so spurious partial matches (e.g. "moon
    cheese" -> cheddar) are rejected while brand-plus-category queries
    (e.g. "pringles chips") still resolve.

    This is the primary lookup used by `optimize_shopping_list`."""
    tier_query = _strip_qty_unit(item_query) or item_query
    for search_q in _cache_query_cascade(item_query):
        hits = _cache_strict_hits(search_q, tier_query)
        if hits:
            return _cache_entry_to_mock_shape(hits[0], stores)
    return None


def find_at_store_in_cache(item_query: str, store_id: str,
                           stores: dict) -> dict | None:
    """Cache-backed lookup restricted to a single store. Same strict
    tier-0/1 acceptance rule as `find_cheapest_in_cache`."""
    tier_query = _strip_qty_unit(item_query) or item_query
    for search_q in _cache_query_cascade(item_query):
        hits = _cache_strict_hits(search_q, tier_query,
                                  stores_filter=[store_id])
        if hits:
            return _cache_entry_to_mock_shape(hits[0], stores)
    return None


def find_cheapest_in_cache_excluding(
    item_query: str, exclude_store_ids: list[str], stores: dict,
) -> dict | None:
    """Cheapest cache match for `item_query` from any store NOT in
    `exclude_store_ids`. Used by avoid-store rebalancing."""
    tier_query = _strip_qty_unit(item_query) or item_query
    excluded = set(exclude_store_ids or [])
    for search_q in _cache_query_cascade(item_query):
        hits = _cache_strict_hits(search_q, tier_query)
        for h in hits:
            if h.get("store_id") not in excluded:
                return _cache_entry_to_mock_shape(h, stores)
    return None


def _dedup(seq):
    """Order-preserving dedupe; drops falsy entries."""
    seen: set = set()
    out: list = []
    for x in seq:
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _match_category(item_query: str, items_db: dict) -> list | None:
    """
    Find the best-matching category in items_db for a query, using synonym
    expansion. Tries:
      1. direct substring match on the original query
      2. substring match against any synonym-expanded candidate (e.g.
         "pork chop" -> "pork")
    Returns the list of entries for the matched category, or None.
    """
    query_lower = (item_query or "").lower()

    for category, entries in items_db.items():
        if category in query_lower or query_lower in category:
            return entries

    for cand in expand_query(item_query):
        for category, entries in items_db.items():
            if category in cand or cand in category:
                return entries
    return None


def _llm_pick_for_item(item_query: str, stores: dict) -> dict | None:
    """Ask the recommender LLM to pick the single best cache SKU for this
    line item (relevance-first, then value). Returns the same shape as
    `_cache_entry_to_mock_shape` on success; None if no pick or on error.

    Used only when USE_LLM_MAIN_OPTIMIZER is true; callers should fall
    back to find_cheapest_in_cache when this returns None."""
    from tools.recommender import line_item_pick_hints, recommend_for_query

    q = _strip_qty_unit(item_query) or (item_query or "").strip()
    if not q:
        return None
    try:
        result = recommend_for_query(
            q,
            topk=1,
            max_candidates=40,
            extra_constraints=line_item_pick_hints(item_query),
        )
    except Exception:
        return None
    picks = result.get("picks") or []
    if not picks:
        return None
    c = (picks[0].get("candidate") or {})
    store_id = (c.get("store_id") or "").strip()
    price = c.get("price")
    name = (c.get("name") or "").strip()
    if not store_id or price is None or not name:
        return None
    try:
        price_f = float(price)
    except (TypeError, ValueError):
        return None
    store_meta = stores.get(store_id) or {}
    display = store_meta.get("display_name") or c.get("store") or store_id
    return {
        "store": display,
        "location": store_meta.get("address", ""),
        "item_name": name,
        "item_price": price_f,
        "url": c.get("url"),
        "_source": "llm",
        "_store_id": store_id,
    }


def optimize_shopping_list(items: list[str]) -> dict:
    """
    Main entry point for the price optimizer tool. Cache-only:
    queries the real scraped per-store caches (data/price_cache/*.json)
    and never touches mock_prices.json. mock data is no longer the
    primary source — we trust the ~21k SKUs from the live scrapers.

    When USE_LLM_MAIN_OPTIMIZER is true, each line item is first passed
    through the recommender LLM (same candidate list as ``recommend X``);
    if the LLM yields no valid pick, falls back to the deterministic
    find_cheapest_in_cache path for that item.

    Takes a list of item strings, finds the cheapest store for each,
    and groups them into a per-store buy plan.

    Returns:
    {
      "plan": {
        "aldi_greenfield": [
          {"item": "Whole Milk 1 Gallon", "price": 3.19, "store_display": "aldi", "url": "..."},
          ...
        ],
        ...
      },
      "total_cost": 12.45,
      "not_found": ["mystery item"],
      "store_ids": ["aldi_greenfield", "trader_joes_shadyside"],
      "stores_meta": { store_id: store_dict, ... }
    }
    """
    stores = load_stores()

    plan: dict[str, list] = {}
    not_found: list[str] = []
    total_cost = 0.0

    for item in items:
        result = None
        if USE_LLM_MAIN_OPTIMIZER:
            result = _llm_pick_for_item(item, stores)
        if result is None:
            result = find_cheapest_in_cache(item, stores)
        if result is None:
            not_found.append(item)
            continue

        store_id = result.get("_store_id")
        if not store_id:
            not_found.append(item)
            continue

        plan.setdefault(store_id, []).append({
            "item": result["item_name"],
            # Back-pointer to the original ingredient query so the agent
            # can delete SKUs by source (e.g. "remove the orange" when the
            # SKU was picked as "Navel Oranges") without relying on fuzzy
            # re-matching against SKU display names.
            "source_item": item,
            "price": result["item_price"],
            "store_display": result["store"],
            "url": result.get("url"),
            "source": result.get("_source", "cache"),
        })
        total_cost += result["item_price"]

    return {
        "plan": plan,
        "total_cost": round(total_cost, 2),
        "not_found": not_found,
        "store_ids": list(plan.keys()),
        "stores_meta": {sid: stores[sid] for sid in plan if sid in stores},
    }


def find_at_store(item_query: str, store_id: str, price_data: dict, stores: dict | None = None) -> dict | None:
    """
    Find the entry for a given item query at a specific store_id, if available.

    Same substring matching on category as find_cheapest, but restricted to
    entries whose store display_name maps to the target store_id. Falls back
    to the real scraped caches when mock has no matching category.
    Returns the price entry dict or None.
    """
    if stores is None:
        stores = load_stores()
    name_index = build_display_name_index(stores)

    from tools.product_search import _relevance_tier

    items_db = price_data.get("items", {})
    matched_entries = _match_category(item_query, items_db) or []

    candidates = [
        e for e in matched_entries
        if name_index.get(e.get("store", "").lower()) == store_id
    ]
    if candidates:
        tier_query = _strip_qty_unit(item_query) or item_query
        scored = [
            (_relevance_tier(tier_query, e.get("item_name", "")),
             float(e["item_price"]), e)
            for e in candidates
        ]
        scored.sort(key=lambda x: (x[0], x[1]))
        if scored[0][0] <= 1:
            return scored[0][2]

    # Cache fallback: strict tier-0/1 hit restricted to this store, using
    # the same query-cascade as find_cheapest_in_cache.
    tier_query = _strip_qty_unit(item_query) or item_query
    for search_q in _cache_query_cascade(item_query):
        hits = _cache_strict_hits(search_q, tier_query, stores_filter=[store_id])
        if hits:
            return _cache_entry_to_mock_shape(hits[0], stores)
    return None


def get_all_prices_for_item(item_query: str) -> list[dict]:
    """
    Return all store prices for a given item, sorted cheapest first.
    Useful for displaying price comparison to the user.
    """
    price_data = load_prices()
    items_db = price_data.get("items", {})
    matched = _match_category(item_query, items_db)
    if matched is None:
        return []
    return sorted(matched, key=lambda e: e["item_price"])