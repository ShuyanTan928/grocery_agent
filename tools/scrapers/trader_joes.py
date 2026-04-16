# ============================================================
# tools/scrapers/trader_joes.py
# Fetch product + price data from Trader Joe's internal GraphQL
# endpoint (the same one the website calls from the browser).
#
# Entry point: fetch_trader_joes(store_code, location)
# Returns a payload ready for tools.price_cache.save_cache().
# ============================================================

from __future__ import annotations

import time

import requests

GRAPHQL_URL = "https://www.traderjoes.com/api/graphql"
HOME_URL = "https://www.traderjoes.com/home"
PRODUCTS_URL = "https://www.traderjoes.com/home/products"
PDP_URL_TEMPLATE = "https://www.traderjoes.com/home/products/pdp/{url_key}-{sku}"

# Top-level brand info reused when building cache store_meta.
BRAND_INFO = {
    "name": "Trader Joe's",
    "website": "https://www.traderjoes.com",
    "products_url": PRODUCTS_URL,
    "api_url": GRAPHQL_URL,
    "store_finder_url": "https://www.traderjoes.com/home/stores",
}

# Minimal subset of fields needed to build a price listing.
QUERY = """
query SearchProducts($pageSize: Int, $currentPage: Int, $storeCode: String, $published: String = "1") {
  products(
    filter: {store_code: {eq: $storeCode}, published: {eq: $published}}
    pageSize: $pageSize
    currentPage: $currentPage
  ) {
    items {
      sku
      url_key
      name
      item_title
      sales_size
      sales_uom_description
      availability
      retail_price
      price_range { minimum_price { final_price { currency value } } }
    }
    total_count
    page_info { current_page page_size total_pages }
  }
}
""".strip()

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)

BASE_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "user-agent": USER_AGENT,
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="136", "Google Chrome";v="136"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
}

GRAPHQL_HEADERS = {
    **BASE_HEADERS,
    "content-type": "application/json",
    "origin": "https://www.traderjoes.com",
    "referer": PRODUCTS_URL,
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
}


def _build_session() -> requests.Session:
    """Session pre-warmed with cookies by visiting the storefront."""
    session = requests.Session()
    session.headers.update(BASE_HEADERS)
    try:
        session.get(HOME_URL, timeout=15)
        session.get(PRODUCTS_URL, timeout=15)
    except requests.RequestException:
        # Non-fatal: some environments still succeed without warm-up.
        pass
    return session


def _post(
    session: requests.Session,
    store_code: str,
    page: int,
    page_size: int,
    timeout: float,
) -> dict:
    body = {
        "operationName": "SearchProducts",
        "variables": {
            "storeCode": str(store_code),
            "published": "1",
            "currentPage": page,
            "pageSize": page_size,
        },
        "query": QUERY,
    }
    resp = session.post(
        GRAPHQL_URL,
        json=body,
        headers=GRAPHQL_HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError(f"Trader Joe's GraphQL errors: {payload['errors']}")
    data = (payload.get("data") or {}).get("products")
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected Trader Joe's response: {payload}")
    return data


def fetch_all_products(
    store_code: str,
    *,
    page_size: int = 100,
    max_pages: int = 200,
    sleep_s: float = 0.2,
    timeout: float = 20.0,
    session: requests.Session | None = None,
) -> list[dict]:
    """Paginate SearchProducts and return raw item dicts."""
    owned_session = False
    if session is None:
        session = _build_session()
        owned_session = True

    try:
        items: list[dict] = []
        page = 1
        while page <= max_pages:
            products = _post(session, store_code, page, page_size, timeout)
            batch = products.get("items") or []
            items.extend(batch)

            info = products.get("page_info") or {}
            total_pages = int(info.get("total_pages") or 1)
            if page >= total_pages or not batch:
                break
            page += 1
            if sleep_s:
                time.sleep(sleep_s)
        return items
    finally:
        if owned_session:
            session.close()


def _coerce_price(raw: dict) -> float | None:
    v = raw.get("retail_price")
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    # Fallback: price_range.minimum_price.final_price.value
    pr = (raw.get("price_range") or {}).get("minimum_price") or {}
    fp = pr.get("final_price") or {}
    try:
        return float(fp.get("value"))
    except (TypeError, ValueError):
        return None


def _build_pdp_url(sku: str | None, url_key: str | None) -> str | None:
    if not sku or not url_key:
        return None
    return PDP_URL_TEMPLATE.format(url_key=url_key, sku=sku)


def normalize_item(raw: dict, location: str) -> dict:
    price = _coerce_price(raw)
    name = (raw.get("item_title") or raw.get("name") or "").strip()
    size = raw.get("sales_size")
    unit = raw.get("sales_uom_description")
    sku = raw.get("sku")
    url_key = raw.get("url_key")

    display = name
    if name and size and unit:
        display = f"{name} {size} {unit}".strip()

    return {
        "store": "trader joe's",
        "location": location,
        "item_name": display,
        "item_price": price,
        "url": _build_pdp_url(sku, url_key),
        "_raw": {
            "sku": sku,
            "url_key": url_key,
            "availability": raw.get("availability"),
            "sales_size": size,
            "sales_uom": unit,
        },
    }


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
    """Bundle store-level info for inclusion in the cache payload."""
    return {
        "store_id": store_id,
        "display_name": "trader joe's",
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


def fetch_trader_joes(
    store_code: str,
    location: str,
    *,
    page_size: int = 100,
    store_meta: dict | None = None,
) -> dict:
    """
    Fetch + normalize Trader Joe's products for one store.
    Result is the *payload* portion of the cache entry (the
    wrapper adds scraped_date / scraped_at).
    """
    raw_items = fetch_all_products(store_code, page_size=page_size)
    normalized = [normalize_item(r, location) for r in raw_items]
    normalized = [
        i for i in normalized
        if i["item_price"] is not None and i["item_name"]
    ]

    payload: dict = {
        "store_code": str(store_code),
        "location": location,
        "source": "traderjoes.com/api/graphql",
        "item_count": len(normalized),
        "items": normalized,
    }
    if store_meta is not None:
        payload["store"] = store_meta
    return payload
