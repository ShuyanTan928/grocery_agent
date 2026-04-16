from __future__ import annotations

import json
from datetime import date, datetime, timezone, timedelta

import pytest

from tools import price_cache


@pytest.fixture(autouse=True)
def tmp_cache_dir(tmp_path, monkeypatch):
    """Redirect PRICE_CACHE_DIR to a tmp dir for every test."""
    monkeypatch.setattr(price_cache, "PRICE_CACHE_DIR", str(tmp_path))
    return tmp_path


def _sample_payload() -> dict:
    return {
        "store_code": "638",
        "location": "Somewhere",
        "source": "test",
        "item_count": 2,
        "items": [
            {"store": "trader joe's", "item_name": "Milk", "item_price": 3.99},
            {"store": "trader joe's", "item_name": "Eggs", "item_price": 2.49},
        ],
    }


def test_save_and_load_roundtrip(tmp_cache_dir):
    payload = _sample_payload()
    path = price_cache.save_cache("trader_joes_shadyside", payload)

    assert path.exists()
    data = json.loads(path.read_text())
    assert data["store_id"] == "trader_joes_shadyside"
    assert data["scraped_date"] == date.today().isoformat()
    assert data["item_count"] == 2

    cached = price_cache.load_cached("trader_joes_shadyside")
    assert cached is not None
    assert cached["item_count"] == 2
    assert cached["items"][0]["item_name"] == "Milk"


def test_load_returns_none_when_stale(tmp_cache_dir):
    # Save with a scraped_at one day in the past.
    past = datetime.now(timezone.utc) - timedelta(days=1)
    price_cache.save_cache("stale_store", _sample_payload(), now=past)

    assert price_cache.load_cached("stale_store") is None


def test_load_hits_with_explicit_today(tmp_cache_dir):
    past = datetime.now(timezone.utc) - timedelta(days=3)
    price_cache.save_cache("backdated", _sample_payload(), now=past)

    assert price_cache.load_cached("backdated", today=past.date()) is not None


def test_load_missing_returns_none(tmp_cache_dir):
    assert price_cache.load_cached("never_saved") is None


def test_cache_info_exposes_metadata(tmp_cache_dir):
    price_cache.save_cache("peek", _sample_payload())
    info = price_cache.cache_info("peek")

    assert info is not None
    assert info["item_count"] == 2
    assert info["scraped_date"] == date.today().isoformat()
