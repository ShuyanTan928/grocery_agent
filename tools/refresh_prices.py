# ============================================================
# tools/refresh_prices.py
# CLI for refreshing per-store real-price caches.
#
# Examples:
#   uv run python -m tools.refresh_prices --store trader_joes
#   uv run python -m tools.refresh_prices --store trader_joes --force
# ============================================================

from __future__ import annotations

import argparse
import sys

from config.settings import GIANT_EAGLE_STORE_CODE, TRADER_JOES_STORE_CODE
from tools.price_cache import cache_info, load_cached, save_cache
from tools.price_optimizer import load_stores
from tools.scrapers.giant_eagle import (
    build_store_meta as build_giant_eagle_meta,
    fetch_giant_eagle,
)
from tools.scrapers.trader_joes import (
    build_store_meta as build_trader_joes_meta,
    fetch_trader_joes,
)

TRADER_JOES_STORE_ID = "trader_joes_shadyside"
GIANT_EAGLE_STORE_ID = "giant_eagle_squirrel_hill"


def _trader_joes_config() -> dict:
    stores = load_stores()
    base = stores.get(TRADER_JOES_STORE_ID, {})
    store_meta = build_trader_joes_meta(
        TRADER_JOES_STORE_CODE,
        store_id=TRADER_JOES_STORE_ID,
        branch=base.get("branch", "Shadyside"),
        address=base.get("address", "6343 Penn Ave, Pittsburgh, PA 15206"),
        lat=base.get("lat"),
        lng=base.get("lng"),
        hours=base.get("hours"),
    )
    return {
        "store_id": TRADER_JOES_STORE_ID,
        "store_meta": store_meta,
        "fetch": lambda: fetch_trader_joes(
            TRADER_JOES_STORE_CODE,
            store_meta["address"],
            store_meta=store_meta,
        ),
    }


def _giant_eagle_config() -> dict:
    stores = load_stores()
    base = stores.get(GIANT_EAGLE_STORE_ID, {})
    store_meta = build_giant_eagle_meta(
        GIANT_EAGLE_STORE_CODE,
        store_id=GIANT_EAGLE_STORE_ID,
        branch=base.get("branch", "Squirrel Hill"),
        address=base.get("address", "1901 Murray Ave, Pittsburgh, PA 15217"),
        lat=base.get("lat"),
        lng=base.get("lng"),
        hours=base.get("hours"),
    )
    return {
        "store_id": GIANT_EAGLE_STORE_ID,
        "store_meta": store_meta,
        "fetch": lambda: fetch_giant_eagle(
            GIANT_EAGLE_STORE_CODE,
            store_meta["address"],
            store_meta=store_meta,
        ),
    }


STORES: dict[str, dict] = {
    "trader_joes": _trader_joes_config(),
    "giant_eagle": _giant_eagle_config(),
}


def refresh(store_key: str, force: bool) -> int:
    if store_key not in STORES:
        print(f"[error] unknown store: {store_key}", file=sys.stderr)
        return 2

    cfg = STORES[store_key]
    store_id = cfg["store_id"]

    if not force:
        cached = load_cached(store_id)
        if cached:
            print(
                f"[cache hit] {store_id}: "
                f"{cached.get('item_count', 'n/a')} items "
                f"(scraped_date={cached.get('scraped_date')})"
            )
            return 0

    existing = cache_info(store_id)
    if existing:
        print(
            f"[cache stale] {store_id}: scraped_date={existing['scraped_date']}, "
            "refreshing ..."
        )
    else:
        print(f"[fetching] {store_id} (no cache) ...")

    payload = cfg["fetch"]()
    path = save_cache(store_id, payload)
    print(
        f"[saved] {path} "
        f"({payload['item_count']} items, store_code={payload['store_code']})"
    )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Refresh store price caches.")
    p.add_argument("--store", choices=list(STORES), required=True)
    p.add_argument("--force", action="store_true", help="Ignore today's cache.")
    args = p.parse_args()
    raise SystemExit(refresh(args.store, args.force))


if __name__ == "__main__":
    main()
