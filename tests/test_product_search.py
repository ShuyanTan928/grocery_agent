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


# ---------- ranked search --------------------------------------------------

def test_relevance_tier_classification():
    # exact phrase
    assert ps._relevance_tier("ground beef", "Fresh Ground Beef 80/20") == 0
    # all words present (different order)
    assert ps._relevance_tier("ground beef", "Beef Patties Ground 80/20") == 1
    # neither phrase nor all tokens
    assert ps._relevance_tier("ground beef", "Beef Broth 32 oz") == 2
    # empty inputs degrade gracefully
    assert ps._relevance_tier("", "anything") == 2
    assert ps._relevance_tier("foo", "") == 2


def test_relevance_tier_demotes_prepared_category_skus():
    """Regression: real bacon must outrank frozen burritos listing bacon
    as an ingredient, and real eggs must outrank Marshmallow Eggs candy."""
    # bacon query — burrito SKU has "burritos" (prepared) and should be demoted
    real_bacon = ps._relevance_tier("bacon", "Sugardale Hickory Smoked Bacon (12 oz)")
    burrito = ps._relevance_tier(
        "bacon",
        "El Monterey Burritos, Egg, Applewood Smoked Bacon & Cheese (1 each)",
    )
    ravioli = ps._relevance_tier("bacon", "Brussels Sprouts & Uncured Bacon Ravioli 8 Oz")
    salad_kit = ps._relevance_tier("bacon", "Reggano Ranch and Bacon Pasta Salad Kit")
    bits = ps._relevance_tier("bacon", "Tuscan Garden Real Bacon Bits")
    assert real_bacon == 0
    assert burrito > real_bacon
    assert ravioli > real_bacon
    assert salad_kit > real_bacon
    assert bits > real_bacon

    # eggs query — marshmallow candy demoted, real eggs stay tier 0
    real_eggs = ps._relevance_tier("eggs", "Crystal Spring Large White Eggs, 12 Ct")
    marshmallow = ps._relevance_tier("eggs", "Marshmallow Eggs 1.5 Oz")
    chocolate = ps._relevance_tier("eggs", "Chocolate Truffle Eggs 2.22 Oz")
    assert real_eggs == 0
    assert marshmallow > real_eggs
    assert chocolate > real_eggs


def test_relevance_tier_no_demotion_when_query_is_category():
    """If the user explicitly asks for the prepared category (e.g. "yogurt",
    "chocolate"), don't demote matching SKUs."""
    # "yogurt" query should still tier-0 Greek yogurt even though "yogurt" is in the blocklist
    assert ps._relevance_tier("yogurt", "FAGE Total Greek Yogurt 6 Oz") == 0
    # "marshmallow" query keeps Marshmallow Eggs at tier 0
    assert ps._relevance_tier("marshmallow", "Marshmallow Eggs 1.5 Oz") == 0
    # "chocolate chips" keeps chocolate chip SKUs at tier 0
    assert ps._relevance_tier(
        "chocolate chips", "Nestle Toll House Semi-Sweet Chocolate Chips 12 oz"
    ) == 0


def test_relevance_tier_demotion_caps_at_two():
    """Demotion must never produce an invalid tier (≤ 2)."""
    # tier-1 base + demotion should cap at 2, not become 3
    t = ps._relevance_tier(
        "bacon eggs", "Frozen Breakfast Burritos with Bacon and Eggs"
    )
    assert t in (1, 2)


def test_search_ranked_puts_exact_matches_first(fake_caches, monkeypatch):
    # Add a noisy item to verify ordering
    extra = {
        "store_id": "noise_store",
        "scraped_date": "2026-04-16",
        "items": [
            {"store": "noise", "item_name": "Pork Rinds Snack 4 oz", "item_price": 1.99},
            {"store": "noise", "item_name": "Pork Loin Roast 2 lb", "item_price": 9.99},
        ],
    }
    cache_dir = Path(ps.PRICE_CACHE_DIR)
    (cache_dir / "noise_store.json").write_text(json.dumps(extra))

    res = ps.search_products_ranked("pork loin", include_mock=False)
    # tier 0 (contains "pork loin") items should appear before tier 1 ("pork rinds" only has "pork")
    tiers = [r["_relevance_tier"] for r in res]
    assert tiers == sorted(tiers), f"tiers should be non-decreasing, got {tiers}"
    # first item must be a tier-0 match
    assert res[0]["_relevance_tier"] == 0
    assert "pork loin" in res[0]["item_name"].lower()
