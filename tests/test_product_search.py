from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import product_search as ps


@pytest.fixture
def fake_caches(tmp_path, monkeypatch):
    """Point PRICE_CACHE_DIR and MOCK_DATA_DIR at a tmp dir with known fixtures."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    mock_dir = tmp_path / "mock"
    mock_dir.mkdir()

    (cache_dir / "trader_joes_shadyside.json").write_text(json.dumps({
        "store_id": "trader_joes_shadyside",
        "scraped_date": "2026-04-16",
        "items": [
            {"store": "trader joe's", "item_name": "Organic Whole Milk 1 Gallon",
             "item_price": 4.49, "url": "https://tj/milk"},
            {"store": "trader joe's", "item_name": "Bananas 1 lb",
             "item_price": 0.29, "url": "https://tj/bananas"},
            {"store": "trader joe's", "item_name": "Pork Loin Tenderloin 1 lb",
             "item_price": 7.49, "url": "https://tj/pork"},
        ],
    }))
    (cache_dir / "aldi_greenfield.json").write_text(json.dumps({
        "store_id": "aldi_greenfield",
        "scraped_date": "2026-04-16",
        "items": [
            {"store": "aldi", "item_name": "Whole Milk 1 Gallon", "item_price": 3.19},
            {"store": "aldi", "item_name": "Bananas 1 lb", "item_price": 0.45},
        ],
    }))
    (mock_dir / "mock_prices.json").write_text(json.dumps({
        "last_updated": "2026-04-01",
        "items": {
            "milk": [
                {"store": "walmart", "location": "x", "item_name": "GV Whole Milk 1 Gallon",
                 "item_price": 3.48},
            ],
        },
    }))

    monkeypatch.setattr(ps, "PRICE_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(ps, "MOCK_DATA_DIR", str(mock_dir))
    return tmp_path


def test_search_finds_all_milk(fake_caches):
    res = ps.search_products("milk")
    names = [r["item_name"] for r in res]
    assert "Whole Milk 1 Gallon" in names
    assert "Organic Whole Milk 1 Gallon" in names
    assert "GV Whole Milk 1 Gallon" in names


def test_search_is_case_insensitive(fake_caches):
    assert ps.search_products("MILK") == ps.search_products("milk")


def test_search_sorted_by_price_asc(fake_caches):
    res = ps.search_products("milk")
    prices = [r["item_price"] for r in res]
    assert prices == sorted(prices)


def test_search_restrict_to_store(fake_caches):
    res = ps.search_products("milk", store_ids=["aldi_greenfield"])
    assert len(res) == 1
    assert res[0]["store_id"] == "aldi_greenfield"


def test_search_respects_max_price_and_limit(fake_caches):
    res = ps.search_products("milk", max_price=3.50, limit=1)
    assert len(res) == 1
    assert res[0]["item_price"] <= 3.50


def test_search_exclude_mock(fake_caches):
    res = ps.search_products("milk", include_mock=False)
    sources = {r["source"] for r in res}
    assert sources == {"cache"}


def test_search_empty_query_returns_empty(fake_caches):
    assert ps.search_products("") == []


def test_search_no_matches(fake_caches):
    assert ps.search_products("unicornburger") == []


def test_cache_entries_annotate_source_and_store(fake_caches):
    res = ps.search_products("bananas", include_mock=False)
    assert {r["source"] for r in res} == {"cache"}
    assert {r["store_id"] for r in res} == {"trader_joes_shadyside", "aldi_greenfield"}
