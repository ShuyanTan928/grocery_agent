# ============================================================
# tools/scrapers/target.py
# Fetch product + price data from Target's Redsky aggregations API
# (https://redsky.target.com/redsky_aggregations/v1/web/...).
#
# Notes on anti-bot:
#   Target's web UA (desktop Chrome) triggers Akamai bot manager
#   after ~10 quick requests, resulting in HTTP 403 + captcha body.
#   An iOS mobile UA plus `X-Requested-With: XMLHttpRequest` is
#   accepted by the same endpoints without captcha, so we use that.
#   The public Redsky "key" is baked into the site HTML and
#   rotates rarely; we hardcode the current value and document
#   how to re-extract it if it ever flips.
#
# Strategy:
#   1. Walk a fixed list of grocery sub-categories (their tcin IDs)
#   2. For each, page through `plp_search_v2` with offset pagination
#   3. API caps: count<=28 per page, offset<=1199 per category
#   4. Dedupe by tcin across categories
#   5. Normalize to the same shape as trader_joes.py / giant_eagle.py
#
# Entry point: fetch_target(store_code, location, *, store_meta=None)
# Returns a payload ready for tools.price_cache.save_cache().
# ============================================================

from __future__ import annotations

import html
import random
import time
import uuid
from typing import Iterable

import requests

# ---- API constants ---------------------------------------------------------

REDSKY_BASE = "https://redsky.target.com/redsky_aggregations/v1/web"
# Public "API key" baked into target.com — not secret, but may rotate.
# To re-extract: GET https://www.target.com/c/grocery/-/N-5xt1a and
# grep for `key=[0-9a-f]{40}` in the HTML.
DEFAULT_REDSKY_KEY = "9f36aeafbe60771e321a7cc95a78140772ab3e96"

HOME_URL = "https://www.target.com/"
GROCERY_URL = "https://www.target.com/c/grocery/-/N-5xt1a"
PDP_URL_TEMPLATE = "https://www.target.com/p/-/A-{tcin}"

BRAND_INFO = {
    "name": "Target",
    "website": "https://www.target.com",
    "products_url": GROCERY_URL,
    "api_url": REDSKY_BASE,
    "store_finder_url": "https://www.target.com/store-locator/find-stores",
}

# iOS Safari UA + XHR header avoids the Akamai challenge that triggers
# on desktop UAs after ~10 consecutive Redsky requests.
USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
    "Mobile/15E148 Safari/604.1"
)

BASE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.target.com",
    "Referer": "https://www.target.com/",
    "X-Requested-With": "XMLHttpRequest",
}


# ---- Grocery sub-categories (tcin -> slug) --------------------------------
# Extracted from the /c/grocery nav (2025-). If Target re-organizes, re-run:
#   curl https://www.target.com/c/grocery/-/N-5xt1a | rg -o '/c/([a-z0-9-]+-grocery)/-/N-([0-9a-z]+)'
GROCERY_SUBCATEGORIES: list[tuple[str, str]] = [
    ("fresh-meat-seafood-grocery", "5xsyh"),
    ("fresh-flowers-plants-produce-grocery", "2dei4"),
    ("dairy-eggs-cheese-grocery", "5xszm"),
    ("deli-grocery", "5hp74"),
    ("bakery-bread-grocery", "5xt19"),
    ("frozen-foods-grocery", "5xszd"),
    ("baking-staples-pantry-grocery", "4u9lv"),
    ("breakfast-cereal-grocery", "wo2mp"),
    ("snacks-grocery", "5xt0n"),
    ("candy-grocery", "5xt0d"),
    ("beverages-grocery", "5xt0r"),
    ("coffee-beverages-grocery", "4yi5p"),
    ("water-beverages-grocery", "5xt0k"),
]


# ---- Session helpers ------------------------------------------------------

def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(BASE_HEADERS)
    try:
        session.get(HOME_URL, timeout=15)
        session.get(GROCERY_URL, timeout=15)
    except requests.RequestException:
        pass
    return session


def _pace(base: float = 0.6) -> None:
    """Small randomized sleep to avoid rate-limit traps."""
    time.sleep(base + random.uniform(0, 0.4))


# ---- Raw fetch ------------------------------------------------------------

class TargetCaptchaError(RuntimeError):
    """Raised when Target's Akamai layer returns a captcha challenge."""


def _plp_search(
    session: requests.Session,
    *,
    store_id: str,
    visitor_id: str,
    category: str | None = None,
    keyword: str | None = None,
    offset: int = 0,
    count: int = 28,
    zipcode: str = "15206",
    redsky_key: str = DEFAULT_REDSKY_KEY,
    timeout: float = 25.0,
) -> dict:
    if not category and not keyword:
        raise ValueError("Must provide category or keyword")

    params = {
        "key": redsky_key,
        "channel": "WEB",
        "count": count,
        "default_purchasability_filter": "true",
        "new_search": "true",
        "offset": offset,
        "platform": "desktop",
        "pricing_store_id": store_id,
        "store_ids": store_id,
        "visitor_id": visitor_id,
        "zip": zipcode,
        "useragent": USER_AGENT,
    }
    if category:
        params["category"] = category
        params["page"] = f"/c/-/N-{category}"
    else:
        params["keyword"] = keyword
        params["page"] = f"/s?searchTerm={keyword}"

    resp = session.get(f"{REDSKY_BASE}/plp_search_v2", params=params, timeout=timeout)

    # Akamai captcha wall returns 403 with a JSON body containing
    # `captchaRelativeURL`.
    if resp.status_code == 403:
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if "captchaRelativeURL" in body or "captcha" in resp.text.lower():
            raise TargetCaptchaError(
                f"Captcha challenge from Redsky (tracking={body.get('captchaRelativeURL')})"
            )

    resp.raise_for_status()
    body = resp.json()
    # Redsky returns per-tcin `errors` alongside a valid `data.search` when
    # some SKUs in the page fail to resolve. Treat those as non-fatal; only
    # raise if there's no usable data at all.
    search = (body.get("data") or {}).get("search")
    if not isinstance(search, dict):
        raise RuntimeError(f"Unexpected Redsky response: {body}")
    return search


def fetch_category_products(
    store_id: str,
    category_tcin: str,
    *,
    page_size: int = 28,
    max_items: int = 1200,
    sleep_s: float = 0.6,
    session: requests.Session | None = None,
    visitor_id: str | None = None,
) -> Iterable[dict]:
    """Yield raw product dicts for a single category via offset pagination.
    Target caps count<=28 and offset<=1199, so max_items <= 1200.
    """
    owned = session is None
    if owned:
        session = _build_session()
    vid = visitor_id or uuid.uuid4().hex.upper()

    try:
        offset = 0
        while offset <= max_items - 1:
            batch_size = min(page_size, max_items - offset, 28)
            if batch_size <= 0:
                return
            search = _plp_search(
                session,
                store_id=store_id,
                visitor_id=vid,
                category=category_tcin,
                offset=offset,
                count=batch_size,
            )
            products = search.get("products") or []
            if not products:
                return
            for p in products:
                yield p
            offset += len(products)
            if len(products) < batch_size:
                return
            if sleep_s:
                time.sleep(sleep_s + random.uniform(0, 0.4))
    finally:
        if owned:
            session.close()


# ---- Normalization --------------------------------------------------------

def _coerce_price(price_obj: dict | None) -> float | None:
    if not price_obj:
        return None
    for key in ("current_retail", "reg_retail"):
        v = price_obj.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _clean_title(raw: str | None) -> str:
    if not raw:
        return ""
    return html.unescape(raw).strip()


def _build_pdp_url(tcin: str | None) -> str | None:
    if not tcin:
        return None
    return PDP_URL_TEMPLATE.format(tcin=tcin)


def normalize_item(raw: dict, location: str) -> dict:
    item = raw.get("item") or {}
    tcin = raw.get("tcin") or item.get("tcin")
    desc = item.get("product_description") or {}
    name = _clean_title(desc.get("title"))
    price = _coerce_price(raw.get("price"))
    brand = (item.get("primary_brand") or {}).get("name")
    promotions = raw.get("promotions") or []

    return {
        "store": "target",
        "location": location,
        "item_name": name,
        "item_price": price,
        "url": _build_pdp_url(tcin),
        "_raw": {
            "tcin": tcin,
            "dpci": item.get("dpci"),
            "brand": brand,
            "unit_price": (raw.get("price") or {}).get("formatted_unit_price"),
            "unit_price_suffix": (raw.get("price") or {}).get("formatted_unit_price_suffix"),
            "reg_retail": (raw.get("price") or {}).get("reg_retail"),
            "promo_count": len(promotions),
            "merchandise_classification": item.get("merchandise_classification"),
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
        "display_name": "target",
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

def fetch_target(
    store_code: str,
    location: str,
    *,
    page_size: int = 28,
    max_items_per_category: int = 1200,
    categories: Iterable[tuple[str, str]] | None = None,
    sleep_s: float = 0.6,
    store_meta: dict | None = None,
    progress: bool = True,
) -> dict:
    """
    Fetch + normalize Target grocery products for one store by walking
    GROCERY_SUBCATEGORIES. Returns the payload portion of a cache entry.

    categories: optional iterable of (slug, tcin). Defaults to
                GROCERY_SUBCATEGORIES (13 grocery aisles).
    """
    cats = list(categories) if categories else GROCERY_SUBCATEGORIES
    session = _build_session()
    try:
        visitor_id = uuid.uuid4().hex.upper()
        seen_tcins: set[str] = set()
        normalized: list[dict] = []

        for slug, tcin in cats:
            count_before = len(normalized)
            try:
                for raw in fetch_category_products(
                    store_code,
                    tcin,
                    page_size=page_size,
                    max_items=max_items_per_category,
                    sleep_s=sleep_s,
                    session=session,
                    visitor_id=visitor_id,
                ):
                    tc = raw.get("tcin") or (raw.get("item") or {}).get("tcin")
                    if not tc or tc in seen_tcins:
                        continue
                    seen_tcins.add(tc)
                    item = normalize_item(raw, location)
                    if item["item_price"] is None or not item["item_name"]:
                        continue
                    normalized.append(item)
            except TargetCaptchaError as exc:
                # Back off hard, rotate visitor_id, then bail on this aisle.
                if progress:
                    print(f"  [aisle] {slug:<40}  hit captcha, backing off 30s")
                time.sleep(30.0)
                visitor_id = uuid.uuid4().hex.upper()
            if progress:
                added = len(normalized) - count_before
                print(f"  [aisle] {slug:<40} +{added} items "
                      f"(running total {len(normalized)})")

        payload: dict = {
            "store_code": str(store_code),
            "location": location,
            "source": "redsky.target.com/redsky_aggregations/v1/web/plp_search_v2",
            "item_count": len(normalized),
            "category_count": len(cats),
            "items": normalized,
        }
        if store_meta is not None:
            payload["store"] = store_meta
        return payload
    finally:
        session.close()
