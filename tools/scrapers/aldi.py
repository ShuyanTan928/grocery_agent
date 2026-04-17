# ============================================================
# tools/scrapers/aldi.py
# Fetch product + price data from Aldi US (www.aldi.us).
#
# Aldi runs its e-commerce on the Instacart "Connect" platform
# (Maplebear), so all data comes from a single GraphQL endpoint
# at https://www.aldi.us/graphql, using Apollo persisted queries.
#
# Note: Aldi online prices are typically a few % higher than
# in-store (Instacart's third-party fulfillment markup), so treat
# these as "delivery prices" rather than literal in-store prices.
#
# Strategy:
#   1. Warm up a session against /store/aldi/storefront (sets
#      __Host-instacart_sid + ahoy_visit cookies).
#   2. Re-extract the persistedQuery sha256Hash for the two ops
#      we use (SearchResultsPlacements + Items) from the storefront
#      HTML, so a rotation of the bundle doesn't break us.
#   3. For each broad keyword in KEYWORDS, call
#      SearchResultsPlacements?first=146 to gather up to ~146
#      itemIds per query.
#   4. Batch-fetch full item details via the Items op (25 at a
#      time, well under the practical request size).
#   5. Dedupe by productId, normalize to the same schema as
#      trader_joes / target / giant_eagle.
#
# Entry point: fetch_aldi(store_code, location, *, store_meta=None)
# Returns a payload ready for tools.price_cache.save_cache().
# ============================================================

from __future__ import annotations

import json
import random
import re
import time
import urllib.parse
import uuid
from typing import Iterable

import requests

# ---- API constants ---------------------------------------------------------

ALDI_HOST = "https://www.aldi.us"
GRAPHQL_URL = f"{ALDI_HOST}/graphql"
STOREFRONT_URL = f"{ALDI_HOST}/store/aldi/storefront"
PDP_URL_TEMPLATE = f"{ALDI_HOST}/store/aldi/products/{{product_id}}/{{slug}}"

# Aldi's storefront on Instacart is a single shopId; the per-ZIP
# delivery zone (zoneId) is fetched dynamically from the storefront.
DEFAULT_SHOP_ID = "5260"

# Pittsburgh CMU coordinates (15213 maps to zoneId 1402 currently).
DEFAULT_POSTAL_CODE = "15213"
DEFAULT_LATITUDE = 40.450302
DEFAULT_LONGITUDE = -79.949303
DEFAULT_ZONE_ID = "1402"

# Persisted query hashes baked into the page bundle. Re-extracted at
# runtime by _refresh_persisted_hashes(); these are just fallbacks.
FALLBACK_SEARCH_HASH = (
    "faf431508136f42b7589e1650a8a26562e9c149cd3829992d36a31367c1e7f01"
)
FALLBACK_ITEMS_HASH = (
    "5116339819ff07f207fd38f949a8a7f58e52cc62223b535405b087e3076ebf2f"
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)

BASE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": ALDI_HOST,
    "Referer": STOREFRONT_URL,
    "x-client-identifier": "web",
}

BRAND_INFO = {
    "name": "Aldi",
    "website": ALDI_HOST,
    "products_url": f"{ALDI_HOST}/store/aldi/pages/explore-all-products",
    "api_url": GRAPHQL_URL,
    "store_finder_url": f"{ALDI_HOST}/store-locator/",
}

# Broad keywords spanning the typical Aldi grocery aisles. Each
# search returns up to ~146 itemIds; with dedupe we usually end up
# with ~1500-2500 unique SKUs for the whole catalog.
KEYWORDS: list[str] = [
    # produce
    "banana", "apple", "orange", "berry", "grape", "lemon", "lime",
    "avocado", "pineapple", "watermelon", "melon",
    "lettuce", "spinach", "kale", "salad", "broccoli", "cauliflower",
    "carrot", "celery", "onion", "garlic", "potato", "sweet potato",
    "tomato", "pepper", "cucumber", "mushroom", "corn", "squash",
    # dairy + eggs
    "milk", "cream", "butter", "cheese", "yogurt", "egg",
    "sour cream", "cottage cheese",
    # meat + seafood
    "chicken", "beef", "ground beef", "steak", "pork", "bacon",
    "sausage", "ham", "turkey", "deli", "salmon", "shrimp", "fish",
    "tuna", "lamb",
    # bakery
    "bread", "bagel", "tortilla", "muffin", "cake", "cookie",
    "pastry", "donut", "roll",
    # pantry / dry goods
    "rice", "pasta", "noodle", "cereal", "oats", "granola",
    "flour", "sugar", "salt", "oil", "olive oil", "vinegar",
    "sauce", "ketchup", "mustard", "mayo", "soup", "broth",
    "bean", "lentil", "peanut butter", "jelly", "jam", "honey",
    "nut", "almond", "walnut", "spice",
    # snacks + sweets
    "chip", "cracker", "popcorn", "pretzel", "candy", "chocolate",
    "ice cream",
    # beverages
    "coffee", "tea", "juice", "water", "soda", "sparkling",
    "energy drink", "wine", "beer",
    # frozen + prepared
    "pizza", "frozen", "frozen vegetable", "frozen fruit",
    "burrito", "dumpling",
    # household / baby (in case Aldi has them)
    "paper towel", "toilet paper", "detergent", "soap", "diaper",
]

ITEM_ID_RE = re.compile(r"items_24276-[0-9]+")
HASH_RE_TEMPLATE = (
    r'"operationName":"{op}","url":"/graphql\?[^"]*'
    r'sha256Hash%22%3A%22([0-9a-f]{{64}})'
)


# ---- Session helpers ------------------------------------------------------

def _build_session() -> requests.Session:
    """Build a session with warmed-up cookies for the Aldi storefront."""
    session = requests.Session()
    session.headers.update(BASE_HEADERS)
    try:
        session.get(STOREFRONT_URL, timeout=20)
    except requests.RequestException:
        pass
    return session


def _refresh_persisted_hashes(session: requests.Session) -> dict[str, str]:
    """
    Re-extract the current sha256Hash for the persisted queries we
    use, by parsing the storefront SSR HTML. Falls back to the
    hardcoded values if the parse fails.
    """
    hashes: dict[str, str] = {}
    try:
        r = session.get(STOREFRONT_URL, timeout=20)
        text = urllib.parse.unquote(r.text)
        for op in ("SearchResultsPlacements", "Items"):
            m = re.search(HASH_RE_TEMPLATE.format(op=op), text)
            if m:
                hashes[op] = m.group(1)
    except requests.RequestException:
        pass
    hashes.setdefault("SearchResultsPlacements", FALLBACK_SEARCH_HASH)
    hashes.setdefault("Items", FALLBACK_ITEMS_HASH)
    return hashes


def _pace(base: float = 0.4) -> None:
    time.sleep(base + random.uniform(0.0, 0.3))


# ---- Raw fetch ------------------------------------------------------------

class AldiBlockedError(RuntimeError):
    """Raised when Aldi/Instacart returns a non-recoverable block."""


def _gql_get(
    session: requests.Session,
    *,
    operation: str,
    variables: dict,
    sha256_hash: str,
    referer: str,
    timeout: float = 30.0,
) -> dict:
    params = {
        "operationName": operation,
        "variables": json.dumps(variables, separators=(",", ":")),
        "extensions": json.dumps(
            {"persistedQuery": {"version": 1, "sha256Hash": sha256_hash}},
            separators=(",", ":"),
        ),
    }
    resp = session.get(
        GRAPHQL_URL, params=params, timeout=timeout,
        headers={"Referer": referer},
    )
    if resp.status_code == 403:
        raise AldiBlockedError(f"403 from Aldi GraphQL ({operation})")
    resp.raise_for_status()
    body = resp.json()
    # Aldi sometimes returns top-level "errors" alongside partial data;
    # treat as fatal only when no usable data is present.
    if "errors" in body and not body.get("data"):
        raise RuntimeError(
            f"Aldi GraphQL errors for {operation}: {body['errors'][:2]}"
        )
    return body


def _search_item_ids(
    session: requests.Session,
    *,
    query: str,
    sha256_hash: str,
    shop_id: str = DEFAULT_SHOP_ID,
    postal_code: str = DEFAULT_POSTAL_CODE,
    zone_id: str = DEFAULT_ZONE_ID,
    first: int = 146,
    page_view_id: str | None = None,
) -> list[str]:
    pvid = page_view_id or str(uuid.uuid4())
    variables = {
        "filters": [],
        "action": None,
        "query": query,
        "pageViewId": pvid,
        "elevatedProductId": None,
        "searchSource": "search",
        "disableReformulation": False,
        "disableLlm": False,
        "forceInspiration": False,
        "orderBy": "bestMatch",
        "clusterId": None,
        "includeDebugInfo": False,
        "clusteringStrategy": None,
        "contentManagementSearchParams": {"itemGridColumnCount": 5},
        "shopId": shop_id,
        "postalCode": postal_code,
        "zoneId": zone_id,
        "first": first,
    }
    body = _gql_get(
        session,
        operation="SearchResultsPlacements",
        variables=variables,
        sha256_hash=sha256_hash,
        referer=f"{ALDI_HOST}/store/aldi/search?k={urllib.parse.quote(query)}",
    )
    raw = json.dumps(body)
    return sorted(set(ITEM_ID_RE.findall(raw)))


def _fetch_items(
    session: requests.Session,
    *,
    item_ids: Iterable[str],
    sha256_hash: str,
    shop_id: str = DEFAULT_SHOP_ID,
    postal_code: str = DEFAULT_POSTAL_CODE,
    zone_id: str = DEFAULT_ZONE_ID,
) -> list[dict]:
    ids = list(item_ids)
    if not ids:
        return []
    variables = {
        "ids": ids,
        "shopId": shop_id,
        "zoneId": zone_id,
        "postalCode": postal_code,
    }
    body = _gql_get(
        session,
        operation="Items",
        variables=variables,
        sha256_hash=sha256_hash,
        referer=STOREFRONT_URL,
    )
    return list((body.get("data") or {}).get("items") or [])


# ---- Normalization --------------------------------------------------------

_PRICE_NUM_RE = re.compile(r"\$?(\d+(?:\.\d{1,2})?)")


def _coerce_price(text: str | None) -> float | None:
    if not text:
        return None
    m = _PRICE_NUM_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def _build_pdp_url(product_id: str | None, slug: str | None) -> str | None:
    if not product_id:
        return None
    return PDP_URL_TEMPLATE.format(
        product_id=product_id, slug=slug or product_id
    )


def normalize_item(raw: dict, location: str) -> dict | None:
    """Extract our common shape from an Aldi GraphQL item node."""
    product_id = raw.get("productId")
    name = (raw.get("name") or "").strip()
    if not name:
        return None

    # Price lives at price.viewSection.itemCard.{priceString,plainFullPriceString}.
    price_section = (raw.get("price") or {}).get("viewSection") or {}
    item_card = price_section.get("itemCard") or {}
    price_string = item_card.get("priceString")
    full_price_string = item_card.get("plainFullPriceString")
    per_unit_string = item_card.get("pricePerUnitString")
    pricing_unit = item_card.get("pricingUnitString")

    current_price = _coerce_price(price_string)
    regular_price = _coerce_price(full_price_string)
    if current_price is None:
        return None

    return {
        "store": "aldi",
        "location": location,
        "item_name": name,
        "item_price": current_price,
        "url": _build_pdp_url(product_id, raw.get("evergreenUrl")),
        "_raw": {
            "product_id": product_id,
            "legacy_id": raw.get("legacyId"),
            "size": raw.get("size"),
            "brand": raw.get("brandName"),
            "regular_price": regular_price,
            "price_string": price_string,
            "price_per_unit_string": per_unit_string,
            "pricing_unit_string": pricing_unit,
        },
    }


# ---- Store metadata -------------------------------------------------------

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
        "display_name": "aldi",
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


# ---- Orchestration --------------------------------------------------------

def fetch_aldi(
    store_code: str,
    location: str,
    *,
    keywords: Iterable[str] | None = None,
    items_batch_size: int = 25,
    sleep_s: float = 0.4,
    shop_id: str = DEFAULT_SHOP_ID,
    postal_code: str = DEFAULT_POSTAL_CODE,
    zone_id: str = DEFAULT_ZONE_ID,
    store_meta: dict | None = None,
    progress: bool = True,
) -> dict:
    """
    Fetch + normalize Aldi grocery products via search-driven crawl.

    `store_code` is currently informational; the real per-ZIP store
    selection is driven by (postal_code, zone_id, shop_id), which all
    default to the Pittsburgh CMU area.

    Returns the payload portion of a cache entry (with `store` meta
    attached when `store_meta` is provided).
    """
    kw_list = list(keywords) if keywords is not None else KEYWORDS
    session = _build_session()
    try:
        hashes = _refresh_persisted_hashes(session)
        if progress:
            print(
                f"  [aldi] persisted hashes: search={hashes['SearchResultsPlacements'][:8]}…  "
                f"items={hashes['Items'][:8]}…"
            )

        # 1) collect itemIds across all keywords
        all_ids: set[str] = set()
        for kw in kw_list:
            try:
                ids = _search_item_ids(
                    session,
                    query=kw,
                    sha256_hash=hashes["SearchResultsPlacements"],
                    shop_id=shop_id,
                    postal_code=postal_code,
                    zone_id=zone_id,
                )
            except (AldiBlockedError, requests.RequestException) as exc:
                if progress:
                    print(f"  [search] {kw:<22} skipped ({exc})")
                continue
            new_ids = set(ids) - all_ids
            all_ids.update(ids)
            if progress:
                print(
                    f"  [search] {kw:<22} +{len(new_ids):>3} new "
                    f"(running total {len(all_ids)})"
                )
            _pace(sleep_s)

        # 2) batch fetch full item data
        normalized: list[dict] = []
        seen_product_ids: set[str] = set()
        ids_list = sorted(all_ids)
        if progress:
            print(
                f"  [items ] fetching {len(ids_list)} items in batches of "
                f"{items_batch_size}…"
            )
        for i in range(0, len(ids_list), items_batch_size):
            batch = ids_list[i : i + items_batch_size]
            try:
                items = _fetch_items(
                    session,
                    item_ids=batch,
                    sha256_hash=hashes["Items"],
                    shop_id=shop_id,
                    postal_code=postal_code,
                    zone_id=zone_id,
                )
            except (AldiBlockedError, requests.RequestException) as exc:
                if progress:
                    print(f"  [items ] batch starting {i} skipped ({exc})")
                continue
            for raw in items:
                pid = raw.get("productId")
                if not pid or pid in seen_product_ids:
                    continue
                norm = normalize_item(raw, location)
                if norm is None:
                    continue
                seen_product_ids.add(pid)
                normalized.append(norm)
            _pace(sleep_s)
            if progress and (i // items_batch_size) % 10 == 9:
                print(
                    f"  [items ] processed {min(i + items_batch_size, len(ids_list))}/"
                    f"{len(ids_list)} ids → {len(normalized)} normalized"
                )

        payload: dict = {
            "store_code": str(store_code),
            "location": location,
            "source": (
                "www.aldi.us/graphql "
                "(SearchResultsPlacements + Items persisted queries)"
            ),
            "item_count": len(normalized),
            "keyword_count": len(kw_list),
            "items": normalized,
        }
        if store_meta is not None:
            payload["store"] = store_meta
        return payload
    finally:
        session.close()
