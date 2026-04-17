# ============================================================
# tools/scrapers/giant_eagle.py
# Fetch product + price data from Giant Eagle's public GraphQL API
# (https://core.shop.gianteagle.com/api/v2). Their robots.txt
# explicitly allows /api/v2 and the schema is introspectable.
#
# Strategy:
#   1. Walk top-level aisles via `categories(store, categoryContext: aisles)`
#   2. For each aisle, page through `products(store, filters: {categoryId})`
#      using Relay-style cursor pagination (after / pageInfo.endCursor)
#   3. Normalize to the same shape as the Trader Joe's scraper
#   4. Dedupe by SKU across aisles (a SKU can appear in multiple categories)
#
# Entry point: fetch_giant_eagle(store_code, location, *, store_meta=None)
# Returns a payload ready for tools.price_cache.save_cache().
# ============================================================

from __future__ import annotations

import time
from typing import Iterable

import requests

GRAPHQL_URL = "https://core.shop.gianteagle.com/api/v2"
HOME_URL = "https://www.gianteagle.com/"
GROCERY_URL = "https://www.gianteagle.com/grocery"
PDP_URL_TEMPLATE = "https://www.gianteagle.com/grocery/search/product/{sku}"

BRAND_INFO = {
    "name": "Giant Eagle",
    "website": "https://www.gianteagle.com",
    "products_url": GROCERY_URL,
    "api_url": GRAPHQL_URL,
    "store_finder_url": "https://www.gianteagle.com/stores",
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)

BASE_HEADERS = {
    "accept": "application/json,*/*",
    "accept-language": "en-US,en;q=0.9",
    "user-agent": USER_AGENT,
}

GRAPHQL_HEADERS = {
    **BASE_HEADERS,
    "content-type": "application/json",
    "origin": "https://www.gianteagle.com",
    "referer": "https://www.gianteagle.com/",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
}


# ---------- GraphQL queries ---------------------------------------------------

CATEGORIES_QUERY = """
query Aisles($store: StoreInput!) {
  categories(store: $store, categoryContext: aisles) {
    id name slug depth hasChildren
  }
}
""".strip()

PRODUCTS_QUERY = """
query CategoryProducts($store: StoreInput!, $filters: ProductFilters,
                       $first: Int, $after: String) {
  products(store: $store, filters: $filters, first: $first, after: $after) {
    totalCount
    pageInfo { hasNextPage endCursor }
    nodes {
      sku name brand
      displayItemSize displayPricePerUnit
      price unitPrice comparedPrice
      pricingModel inventoryStatus
      categoryNames
    }
  }
}
""".strip()


# ---------- Session helpers ---------------------------------------------------

def _build_session() -> requests.Session:
    """Visit the storefront once so we look like a normal browser tab."""
    session = requests.Session()
    session.headers.update(BASE_HEADERS)
    try:
        session.get(HOME_URL, timeout=15)
    except requests.RequestException:
        pass
    return session


def _post(session: requests.Session, query: str, variables: dict, *, timeout: float = 30.0) -> dict:
    resp = session.post(
        GRAPHQL_URL,
        json={"query": query, "variables": variables},
        headers=GRAPHQL_HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError(f"Giant Eagle GraphQL errors: {payload['errors']}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected Giant Eagle response: {payload}")
    return data


# ---------- Data fetchers -----------------------------------------------------

def fetch_top_aisles(store_code: str, *, session: requests.Session | None = None) -> list[dict]:
    """Return the top-level aisles for a store (depth=0)."""
    owned = session is None
    if owned:
        session = _build_session()
    try:
        data = _post(session, CATEGORIES_QUERY, {"store": {"storeCode": str(store_code)}})
        return data["categories"] or []
    finally:
        if owned:
            session.close()


def fetch_category_products(
    store_code: str,
    category_id: str,
    *,
    page_size: int = 100,
    max_pages: int = 50,
    sleep_s: float = 0.2,
    session: requests.Session | None = None,
) -> Iterable[dict]:
    """Yield raw product dicts for a single category, paginated by cursor."""
    owned = session is None
    if owned:
        session = _build_session()
    try:
        cursor: str | None = None
        for _ in range(max_pages):
            variables = {
                "store": {"storeCode": str(store_code)},
                "filters": {"categoryId": str(category_id)},
                "first": page_size,
                "after": cursor,
            }
            data = _post(session, PRODUCTS_QUERY, variables)
            products = data["products"]
            for node in products.get("nodes") or []:
                yield node
            page_info = products.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                return
            cursor = page_info.get("endCursor")
            if not cursor:
                return
            if sleep_s:
                time.sleep(sleep_s)
    finally:
        if owned:
            session.close()


# ---------- Normalization -----------------------------------------------------

def _coerce_price(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_pdp_url(sku: str | None) -> str | None:
    if not sku:
        return None
    return PDP_URL_TEMPLATE.format(sku=sku)


def normalize_item(raw: dict, location: str) -> dict:
    sku = raw.get("sku")
    name = (raw.get("name") or "").strip()
    size = raw.get("displayItemSize")
    display = name
    if name and size and size.lower() not in name.lower():
        display = f"{name} ({size})"

    return {
        "store": "giant eagle",
        "location": location,
        "item_name": display,
        "item_price": _coerce_price(raw.get("price")),
        "url": _build_pdp_url(sku),
        "_raw": {
            "sku": sku,
            "brand": raw.get("brand"),
            "size": size,
            "unit_price": raw.get("unitPrice"),
            "display_price_per_unit": raw.get("displayPricePerUnit"),
            "pricing_model": raw.get("pricingModel"),
            "availability": raw.get("inventoryStatus"),
            "category_names": raw.get("categoryNames"),
        },
    }


# ---------- Store metadata ---------------------------------------------------

def build_store_meta(
    store_code: str,
    *,
    store_id: str,
    branch: str,
    address: str,
    lat: float | None = None,
    lng: float | None = None,
    hours: str | None = None,
) -> dict:
    return {
        "store_id": store_id,
        "display_name": "giant eagle",
        "name": BRAND_INFO["name"],
        "branch": branch,
        "store_code": str(store_code),
        "address": address,
        "lat": lat,
        "lng": lng,
        "hours": hours,
        "website": BRAND_INFO["website"],
        "products_url": BRAND_INFO["products_url"],
        "api_url": BRAND_INFO["api_url"],
        "store_finder_url": BRAND_INFO["store_finder_url"],
    }


# ---------- Orchestration ----------------------------------------------------

# Aisles whose products are useful to the grocery agent. We skip Beauty,
# Personal Care, Pet, Floral, Gift Cards by default since they bloat the cache.
DEFAULT_INCLUDE_AISLES = {
    "meat", "produce", "dairy", "prepared-foods", "pantry", "deli", "seafood",
    "bakery", "snacks", "frozen", "beverages", "candy", "household-supplies",
}


def fetch_giant_eagle(
    store_code: str,
    location: str,
    *,
    page_size: int = 100,
    max_pages_per_aisle: int = 50,
    aisle_slugs: Iterable[str] | None = None,
    sleep_s: float = 0.2,
    store_meta: dict | None = None,
    progress: bool = True,
) -> dict:
    """
    Fetch + normalize Giant Eagle products for one store, by walking
    top-level aisles. Returns the *payload* portion of a cache entry.

    aisle_slugs: optional iterable of aisle slugs to restrict to. Defaults
                 to DEFAULT_INCLUDE_AISLES (food + household).
    """
    session = _build_session()
    try:
        all_aisles = fetch_top_aisles(store_code, session=session)

        wanted = set(aisle_slugs) if aisle_slugs else DEFAULT_INCLUDE_AISLES
        aisles = [a for a in all_aisles if a.get("slug") in wanted]

        seen_skus: set[str] = set()
        normalized: list[dict] = []

        for aisle in aisles:
            aisle_id = aisle["id"]
            aisle_name = aisle.get("name") or aisle.get("slug")
            count_before = len(normalized)
            for raw in fetch_category_products(
                store_code,
                aisle_id,
                page_size=page_size,
                max_pages=max_pages_per_aisle,
                sleep_s=sleep_s,
                session=session,
            ):
                sku = raw.get("sku")
                if not sku or sku in seen_skus:
                    continue
                seen_skus.add(sku)
                item = normalize_item(raw, location)
                if item["item_price"] is None or not item["item_name"]:
                    continue
                normalized.append(item)
            if progress:
                added = len(normalized) - count_before
                print(f"  [aisle] {aisle_name:<22} +{added} items "
                      f"(running total {len(normalized)})")

        payload: dict = {
            "store_code": str(store_code),
            "location": location,
            "source": "core.shop.gianteagle.com/api/v2",
            "item_count": len(normalized),
            "aisle_count": len(aisles),
            "items": normalized,
        }
        if store_meta is not None:
            payload["store"] = store_meta
        return payload
    finally:
        session.close()
