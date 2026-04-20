# ============================================================
# tests/test_price_optimizer.py
# Unit tests for the price optimizer using the new data format:
# { "items": { "category": [ {store, location, item_name, item_price} ] } }
# Run with: pytest tests/test_price_optimizer.py -v
# ============================================================

import pytest
from tools.price_optimizer import (
    optimize_shopping_list,
    find_cheapest,
    load_prices,
    get_all_prices_for_item,
)


pytestmark = pytest.mark.usefixtures("cache_from_mock")


class TestFindCheapest:
    """Tests for single-item price lookup."""

    def test_finds_cheapest_milk(self):
        price_data = load_prices()
        result = find_cheapest("milk", price_data)
        assert result is not None
        # Aldi should be cheapest for milk ($3.19) in mock data
        assert result["store"].lower() == "aldi"
        assert result["item_price"] == 3.19

    def test_finds_cheapest_eggs(self):
        price_data = load_prices()
        result = find_cheapest("eggs", price_data)
        assert result is not None
        assert result["store"].lower() == "aldi"
        assert result["item_price"] == 2.89

    def test_finds_cheapest_bananas(self):
        price_data = load_prices()
        result = find_cheapest("bananas", price_data)
        assert result is not None
        # Trader Joe's has cheapest bananas ($0.29) in mock data
        assert "trader joe" in result["store"].lower()
        assert result["item_price"] == 0.29

    def test_result_has_required_fields(self):
        price_data = load_prices()
        result = find_cheapest("pasta", price_data)
        assert result is not None
        assert "store" in result
        assert "location" in result
        assert "item_name" in result
        assert "item_price" in result

    def test_returns_none_for_unknown_item(self):
        price_data = load_prices()
        result = find_cheapest("dragon fruit", price_data)
        assert result is None

    def test_partial_match_works(self):
        # "pork chops" should match the "pork" category
        price_data = load_prices()
        result = find_cheapest("pork chops", price_data)
        assert result is not None


class TestGetAllPricesForItem:
    """Tests for full price comparison across all stores."""

    def test_returns_all_stores(self):
        results = get_all_prices_for_item("milk")
        assert len(results) == 5  # 5 stores in mock data

    def test_sorted_cheapest_first(self):
        results = get_all_prices_for_item("milk")
        prices = [r["item_price"] for r in results]
        assert prices == sorted(prices)

    def test_empty_for_unknown_item(self):
        results = get_all_prices_for_item("moon cheese")
        assert results == []


class TestOptimizeShoppingList:
    """Tests for the full shopping list optimizer."""

    def test_basic_list(self):
        result = optimize_shopping_list(["milk", "eggs", "bread"])
        assert result["total_cost"] > 0
        assert len(result["plan"]) > 0
        assert result["not_found"] == []

    def test_unknown_items_go_to_not_found(self):
        result = optimize_shopping_list(["milk", "unicorn food"])
        assert "unicorn food" in result["not_found"]
        assert result["total_cost"] > 0

    def test_store_ids_matches_plan_keys(self):
        result = optimize_shopping_list(["milk", "eggs", "bread", "chicken"])
        assert set(result["store_ids"]) == set(result["plan"].keys())

    def test_total_cost_is_sum_of_items(self):
        result = optimize_shopping_list(["milk", "eggs"])
        expected = sum(
            item["price"]
            for items in result["plan"].values()
            for item in items
        )
        assert abs(result["total_cost"] - expected) < 0.01

    def test_empty_list_returns_zero_cost(self):
        result = optimize_shopping_list([])
        assert result["total_cost"] == 0.0
        assert result["plan"] == {}
        assert result["not_found"] == []

    def test_stores_meta_has_address_and_coords(self):
        result = optimize_shopping_list(["milk"])
        for store_id in result["store_ids"]:
            meta = result["stores_meta"][store_id]
            assert "address" in meta
            assert "lat" in meta
            assert "lng" in meta

    def test_plan_items_have_required_fields(self):
        result = optimize_shopping_list(["pork", "chicken"])
        for store_id, items in result["plan"].items():
            for item in items:
                assert "item" in item        # item_name from price data
                assert "price" in item       # item_price from price data
                assert "store_display" in item  # store display name


class TestLlmMainOptimizer:
    """USE_LLM_MAIN_OPTIMIZER: per-item LLM pick with cache fallback."""

    def test_llm_pick_used_when_flag_on(self, monkeypatch):
        monkeypatch.setattr("tools.price_optimizer.USE_LLM_MAIN_OPTIMIZER", True)

        def fake_recommend(q, **kw):
            assert q  # stripped query
            return {
                "picks": [{
                    "rank": 1,
                    "candidate": {
                        "name": "Test Milk LLM Pick",
                        "price": 1.11,
                        "store_id": "aldi_greenfield",
                        "store": "aldi",
                        "url": "https://example.test/milk",
                    },
                    "reason": "fixture",
                }],
            }

        monkeypatch.setattr("tools.recommender.recommend_for_query", fake_recommend)
        result = optimize_shopping_list(["milk"])
        assert result["not_found"] == []
        flat = [it for row in result["plan"].values() for it in row]
        assert len(flat) == 1
        assert flat[0]["item"] == "Test Milk LLM Pick"
        assert flat[0]["price"] == 1.11
        assert flat[0]["source"] == "llm"

    def test_empty_llm_pick_falls_back_to_cache(self, monkeypatch):
        monkeypatch.setattr("tools.price_optimizer.USE_LLM_MAIN_OPTIMIZER", True)
        monkeypatch.setattr(
            "tools.recommender.recommend_for_query",
            lambda *a, **k: {"picks": [], "candidates": []},
        )
        result = optimize_shopping_list(["milk"])
        assert result["not_found"] == []
        flat = [it for row in result["plan"].values() for it in row]
        assert len(flat) == 1
        assert flat[0]["price"] == 3.19  # mock-cache Aldi milk
        assert flat[0]["source"] == "cache"

    def test_llm_exception_falls_back_to_cache(self, monkeypatch):
        monkeypatch.setattr("tools.price_optimizer.USE_LLM_MAIN_OPTIMIZER", True)

        def boom(*a, **k):
            raise RuntimeError("no API")

        monkeypatch.setattr("tools.recommender.recommend_for_query", boom)
        result = optimize_shopping_list(["milk"])
        assert result["not_found"] == []
        flat = [it for row in result["plan"].values() for it in row]
        assert flat[0]["price"] == 3.19
        assert flat[0]["source"] == "cache"