# ============================================================
# tests/conftest.py
# Shared pytest fixtures.
#
# `cache_from_mock` rebuilds data/mock_prices.json (category-grouped)
# into per-store cache files (item-list per store) in a tmp dir, then
# monkeypatches tools.product_search.PRICE_CACHE_DIR to point there.
# This lets us keep the well-loved hand-curated test prices while the
# agent's hot path is cache-only.
#
# Tests that depend on deterministic prices opt in via:
#     pytestmark = pytest.mark.usefixtures("cache_from_mock")
# at the top of the test module (or class).
# ============================================================

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.settings import MOCK_DATA_DIR


# Mock display_name -> real store_id (matches mock_stores.json).
_DISPLAY_TO_STORE_ID = {
    "giant eagle":  "giant_eagle_squirrel_hill",
    "aldi":         "aldi_greenfield",
    "walmart":      "walmart_crafton",
    "trader joe's": "trader_joes_shadyside",
    "whole foods":  "whole_foods_east_liberty",
    "target":       "target_east_liberty",
}


def _build_cache_files(target_dir: Path) -> None:
    """Group mock_prices.json by store and emit one cache file per store."""
    src = Path(MOCK_DATA_DIR) / "mock_prices.json"
    if not src.exists():
        return
    data = json.loads(src.read_text())
    last_updated = data.get("last_updated", "")[:10] or "2026-04-14"

    grouped: dict[str, list[dict]] = {}
    for _category, rows in (data.get("items") or {}).items():
        for row in rows or []:
            disp = (row.get("store") or "").lower()
            sid = _DISPLAY_TO_STORE_ID.get(disp)
            if not sid:
                continue
            grouped.setdefault(sid, []).append({
                "store": row.get("store"),
                "item_name": row.get("item_name"),
                "item_price": row.get("item_price"),
                "url": row.get("url"),
            })

    target_dir.mkdir(parents=True, exist_ok=True)
    for sid, items in grouped.items():
        (target_dir / f"{sid}.json").write_text(json.dumps({
            "store_id": sid,
            "scraped_date": last_updated,
            "items": items,
        }))


@pytest.fixture
def cache_from_mock(tmp_path, monkeypatch):
    """Materialize mock_prices.json into per-store cache files in tmp,
    then point PRICE_CACHE_DIR at it. Function-scoped so each test
    gets a clean isolated cache.
    """
    cache_dir = tmp_path / "cache"
    _build_cache_files(cache_dir)

    from tools import product_search as ps
    monkeypatch.setattr(ps, "PRICE_CACHE_DIR", str(cache_dir))
    return cache_dir
