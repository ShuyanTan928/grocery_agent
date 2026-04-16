# ============================================================
# tools/product_search.py
# Substring search over cached (real) + mock product data.
#
# Used by the agent to answer "what foo options are there?" /
# "find all milk under $5" without re-scraping.
#
# CLI:
#   uv run python -m tools.product_search --q milk
#   uv run python -m tools.product_search --q "pork loin" --store trader_joes --limit 5
# ============================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from config.settings import MOCK_DATA_DIR, PRICE_CACHE_DIR
from tools.synonyms import expand_query


def _iter_cache_entries(store_ids: Iterable[str] | None = None) -> list[dict]:
    """Yield every normalized item across cached stores.

    Each item is enriched with: store_id, source ("cache"), scraped_date.
    Staleness is not checked — this is a lookup, not a refresh path.
    """
    cache_dir = Path(PRICE_CACHE_DIR)
    if not cache_dir.exists():
        return []

    wanted: set[str] | None = set(store_ids) if store_ids else None
    out: list[dict] = []

    for path in sorted(cache_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        store_id = data.get("store_id") or path.stem
        if wanted is not None and store_id not in wanted:
            continue

        scraped_date = data.get("scraped_date")
        for item in data.get("items") or []:
            if not isinstance(item, dict):
                continue
            out.append({
                **item,
                "store_id": store_id,
                "source": "cache",
                "scraped_date": scraped_date,
            })
    return out


def _iter_mock_entries() -> list[dict]:
    """Yield every item in data/mock_prices.json, with a synthetic store_id."""
    path = Path(MOCK_DATA_DIR) / "mock_prices.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    out: list[dict] = []
    for _category, rows in (data.get("items") or {}).items():
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            out.append({
                **row,
                "store_id": None,
                "source": "mock",
                "scraped_date": data.get("last_updated"),
            })
    return out


def search_products(
    query: str,
    *,
    store_ids: Iterable[str] | None = None,
    include_mock: bool = True,
    max_price: float | None = None,
    limit: int | None = None,
    sort_by: str = "price",
    expand_synonyms: bool = True,
) -> list[dict]:
    """Return items whose item_name contains `query` (case-insensitive).

    If `expand_synonyms` is True (default) we also accept surface forms
    defined in tools/synonyms.py — e.g. a query of "pork chop" will also
    match items containing "pork loin", "pork shoulder", etc.

    Args:
        query: substring to match against `item_name`.
        store_ids: restrict to these cached store_ids (None = all caches).
        include_mock: also search data/mock_prices.json (on by default).
        max_price: drop items with item_price above this.
        limit: max number of matches to return.
        sort_by: "price" (asc), "name" (asc), or "none".
        expand_synonyms: include synonym-group surface forms as candidates.
    """
    q = (query or "").strip().lower()
    if not q:
        return []

    candidates: list[str] = [q]
    if expand_synonyms:
        for c in expand_query(q):
            if c not in candidates:
                candidates.append(c)

    wanted: set[str] | None = set(store_ids) if store_ids else None

    entries: list[dict] = _iter_cache_entries(store_ids)
    if include_mock and wanted is None:
        # Mock entries have no store_id, so only include them when not
        # restricting by specific stores.
        entries += _iter_mock_entries()

    matches: list[dict] = []
    for it in entries:
        name = (it.get("item_name") or "").lower()
        if not any(c in name for c in candidates):
            continue
        price = it.get("item_price")
        if max_price is not None and price is not None and price > max_price:
            continue
        matches.append(it)

    if sort_by == "price":
        matches.sort(key=lambda x: (x.get("item_price") is None, x.get("item_price") or 0))
    elif sort_by == "name":
        matches.sort(key=lambda x: (x.get("item_name") or "").lower())
    # else: keep insertion order

    if limit is not None and limit >= 0:
        matches = matches[:limit]
    return matches


def format_results(results: list[dict]) -> str:
    """Human-friendly pretty-print for CLI output."""
    if not results:
        return "(no matches)"
    lines = []
    for r in results:
        price = r.get("item_price")
        price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "   n/a"
        store = r.get("store") or r.get("store_id") or "?"
        name = r.get("item_name") or ""
        url = r.get("url")
        extra = f"  {url}" if url else ""
        lines.append(f"{price_str:>8}  [{store:<14}]  {name}{extra}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Search cached/mock grocery products by name.")
    p.add_argument("--q", "--query", dest="query", required=True, help="substring to search in item_name")
    p.add_argument("--store", dest="stores", action="append", default=None,
                   help="restrict to store_id (repeatable)")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--max-price", type=float, default=None)
    p.add_argument("--sort", choices=["price", "name", "none"], default="price")
    p.add_argument("--no-mock", action="store_true", help="exclude mock data")
    p.add_argument("--no-synonyms", action="store_true", help="disable synonym expansion")
    p.add_argument("--json", action="store_true", help="emit JSON instead of pretty text")
    args = p.parse_args()

    results = search_products(
        args.query,
        store_ids=args.stores,
        include_mock=not args.no_mock,
        max_price=args.max_price,
        limit=args.limit,
        sort_by=args.sort,
        expand_synonyms=not args.no_synonyms,
    )

    if args.json:
        json.dump(results, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        print(format_results(results))


if __name__ == "__main__":
    main()
