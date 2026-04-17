# ============================================================
# tools/price_cache.py
# Per-store, date-based JSON cache for scraped grocery prices.
#
# Layout: <PRICE_CACHE_DIR>/<store_id>.json
# Hit rule: cache is considered fresh when scraped_date == today
#           (override by passing today= or by deleting the file).
# ============================================================

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from config.settings import PRICE_CACHE_DIR


def _cache_path(store_id: str) -> Path:
    return Path(PRICE_CACHE_DIR) / f"{store_id}.json"


def _today_utc() -> date:
    """All freshness comparisons use UTC date so save/load agree across tz."""
    return datetime.now(timezone.utc).date()


def load_cached(store_id: str, *, today: date | None = None) -> dict | None:
    """Return cached payload if its scraped_date matches today (UTC), else None."""
    path = _cache_path(store_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    today_str = (today or _today_utc()).isoformat()
    if data.get("scraped_date") != today_str:
        return None
    return data


def save_cache(store_id: str, payload: dict, *, now: datetime | None = None) -> Path:
    """Persist payload with scraped_date / scraped_at stamps."""
    stamp = now or datetime.now(timezone.utc)
    path = _cache_path(store_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "store_id": store_id,
        "scraped_date": stamp.date().isoformat(),
        "scraped_at": stamp.isoformat(),
        **payload,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def cache_info(store_id: str) -> dict | None:
    """Lightweight peek at a cache file (no freshness check)."""
    path = _cache_path(store_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return {
        "path": str(path),
        "scraped_date": data.get("scraped_date"),
        "scraped_at": data.get("scraped_at"),
        "item_count": data.get("item_count") or len(data.get("items") or []),
    }
