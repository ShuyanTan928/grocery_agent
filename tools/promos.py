"""
Promo (daily-deal) extraction + cache.

Reads the existing per-store price caches in ``data/price_cache/*.json``
and extracts SKUs that look like promos. Writes a compact file at
``data/promos.json`` for fast lookup by the agent loop.

Entry points:
  build_all_promos()      -> dict         # extract from caches
  save_promos(data)       -> Path         # atomic write to promos.json
  load_promos()           -> dict         # read promos.json (or empty)
  get_daily_promos(...)   -> dict         # compact rows for tool callers

Per-store extractors:
  * Target: uses ``_raw.reg_retail`` vs ``item_price`` plus
    ``_raw.promo_count`` from the Target redsky scraper.
  * Giant Eagle / Trader Joe's / Aldi: no explicit promo fields today,
    so we return [] for them. Hooks are in place for future parsers
    (weekly-ad PDFs, circular JSON, etc.).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from config.settings import MOCK_DATA_DIR, PRICE_CACHE_DIR


PROMOS_CACHE_PATH = Path(MOCK_DATA_DIR) / "promos.json"


# Keep the greeting focused on staples. Catalog noise (gift boxes, candy
# novelty, craft kits, etc.) is extracted as "promos" because Target
# marks them as on-sale, but they're not useful to surface in an opening
# grocery greeting. Matched case-insensitively against the item name.
_GREETING_NOVELTY_TOKENS: frozenset[str] = frozenset({
    "gift box", "gift set", "gift card", "gift pack", "novelty",
    "lollipop", "lip shaped", "heart shaped", "starter culture",
    "party favor", "figurine", "collectible", "plush",
    "bouquet", "wreath", "ornament", "stationery", "stickers",
    "nail polish", "essential oil", "supplement", "mushroom coffee",
    "decor", "candle", "soap bar", "shampoo", "tumbler",
    "paper goods", "toys",
    "cotton candy", "flossugar", "candy mix", "gummies", "gummy mix",
    "case of", "pre-mixed",
})


def _is_grocery_like(item_name: str) -> bool:
    name = (item_name or "").lower()
    return not any(tok in name for tok in _GREETING_NOVELTY_TOKENS)


# --------------- helpers ----------------------------------------------------

def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _discount_pct(sale: float, reg: float) -> float:
    if reg <= 0:
        return 0.0
    return round(max(0.0, (reg - sale) / reg) * 100.0, 1)


# --------------- per-store extractors ---------------------------------------

def extract_target_promos(items: list[dict]) -> list[dict]:
    """Target: reg_retail > item_price OR promo_count > 0."""
    out: list[dict] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        raw = it.get("_raw") or {}
        price = _as_float(it.get("item_price"))
        reg = _as_float(raw.get("reg_retail"))
        promo_count = int(raw.get("promo_count") or 0)
        if price is None:
            continue

        discount_reason: str | None = None
        if reg is not None and reg > price + 1e-9:
            discount_reason = "reg_retail_drop"
        elif promo_count > 0:
            discount_reason = "promo_flag"

        if discount_reason is None:
            continue

        out.append({
            "item_name": it.get("item_name") or "",
            "sale_price": price,
            "reg_price": reg,
            "discount_pct": _discount_pct(price, reg) if reg else None,
            "reason": discount_reason,
            "brand": (raw.get("brand") or None),
            "url": it.get("url"),
            "store": "target",
        })
    # biggest % off first; fall back to absolute $ off
    out.sort(
        key=lambda r: (
            -(r["discount_pct"] or 0.0),
            -((r["reg_price"] or r["sale_price"]) - r["sale_price"]),
        )
    )
    return out


def extract_noop(items: list[dict]) -> list[dict]:
    """Placeholder for stores without an explicit promo field in cache."""
    return []


# store_id (exact) → extractor. Unlisted stores → extract_noop.
STORE_EXTRACTORS: dict[str, Callable[[list[dict]], list[dict]]] = {
    "target_east_liberty": extract_target_promos,
    # Hooks (return [] today; wire parsers as you add them):
    "giant_eagle_squirrel_hill": extract_noop,
    "trader_joes_shadyside": extract_noop,
    "aldi_greenfield": extract_noop,
}


# --------------- build + persist --------------------------------------------

def _iter_cache_files() -> list[Path]:
    cache_dir = Path(PRICE_CACHE_DIR)
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.glob("*.json"))


def build_all_promos() -> dict:
    """Scan every per-store cache file and extract promo rows."""
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stores: dict[str, list[dict]] = {}
    source_snapshots: dict[str, str | None] = {}
    total = 0

    for path in _iter_cache_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        store_id = data.get("store_id") or path.stem
        items = data.get("items") or []
        extractor = STORE_EXTRACTORS.get(store_id, extract_noop)
        rows = extractor(items)
        stores[store_id] = rows
        source_snapshots[store_id] = data.get("scraped_date")
        total += len(rows)

    return {
        "generated_at": generated_at,
        "total_promos": total,
        "source_snapshots": source_snapshots,
        "stores": stores,
    }


def save_promos(data: dict, path: Path | None = None) -> Path:
    """Atomic write to ``data/promos.json``. Returns the resolved path."""
    target = path or PROMOS_CACHE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_promos(path: Path | None = None) -> dict:
    """Read the cached promos file. Returns an empty dict if missing or
    malformed."""
    target = path or PROMOS_CACHE_PATH
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# --------------- tool-facing API --------------------------------------------

def get_daily_promos(
    *,
    topk_per_store: int = 5,
    stores: list[str] | None = None,
    min_discount_pct: float | None = None,
) -> dict:
    """Return a compact promo digest suitable for the LLM.

    Shape:
        {
          "generated_at": str,
          "total": int,
          "per_store": {
            "<store_id>": [ {item_name, sale_price, reg_price,
                             discount_pct, reason, url}, ... ],
            ...
          },
          "empty": bool,          # true when nothing cached
        }
    """
    cached = load_promos()
    if not cached:
        return {"generated_at": None, "total": 0, "per_store": {}, "empty": True}

    store_map = cached.get("stores") or {}
    if stores:
        store_map = {sid: rows for sid, rows in store_map.items() if sid in set(stores)}

    per_store: dict[str, list[dict]] = {}
    total = 0
    for sid, rows in store_map.items():
        pruned: list[dict] = []
        for row in rows or []:
            if (
                min_discount_pct is not None
                and (row.get("discount_pct") or 0.0) < min_discount_pct
            ):
                continue
            pruned.append({
                "item_name": row.get("item_name"),
                "sale_price": row.get("sale_price"),
                "reg_price": row.get("reg_price"),
                "discount_pct": row.get("discount_pct"),
                "reason": row.get("reason"),
                "url": row.get("url"),
            })
            if len(pruned) >= topk_per_store:
                break
        if pruned:
            per_store[sid] = pruned
            total += len(pruned)

    return {
        "generated_at": cached.get("generated_at"),
        "total": total,
        "per_store": per_store,
        "empty": total == 0,
    }


def get_greeting_promos(
    *,
    limit: int = 3,
    min_discount_pct: float = 15.0,
    path: Path | None = None,
) -> dict:
    """Return a compact, grocery-only top-deals digest for the opening
    greeting.

    Flattens across stores, drops novelty/gift items, and keeps the top
    ``limit`` rows by discount percentage. Shape:
        {
          "generated_at": str | None,
          "items": [ {item_name, sale_price, reg_price, discount_pct,
                      url, store_id}, ... ],
          "empty": bool,
        }
    """
    cached = load_promos(path=path)
    if not cached:
        return {"generated_at": None, "items": [], "empty": True}

    flat: list[dict] = []
    for sid, rows in (cached.get("stores") or {}).items():
        for row in rows or []:
            name = row.get("item_name") or ""
            pct = row.get("discount_pct") or 0.0
            if pct < min_discount_pct:
                continue
            if not _is_grocery_like(name):
                continue
            flat.append({
                "item_name": name,
                "sale_price": row.get("sale_price"),
                "reg_price": row.get("reg_price"),
                "discount_pct": pct,
                "reason": row.get("reason"),
                "url": row.get("url"),
                "store_id": sid,
            })

    flat.sort(key=lambda r: -(r["discount_pct"] or 0.0))
    top = flat[: max(0, int(limit))]
    return {
        "generated_at": cached.get("generated_at"),
        "items": top,
        "empty": len(top) == 0,
    }
