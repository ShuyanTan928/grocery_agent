# ============================================================
# tests/test_preferences.py
# Tests for preferred_stores (require) and avoid_stores (forbid)
# semantics in the shopping-plan post-processing pipeline.
# These tests use the mock price data; no LLM calls are made.
# ============================================================

import pytest

from tools.price_optimizer import optimize_shopping_list, find_at_store, load_prices
from agent.agent import apply_preferred_stores, apply_avoid_stores


pytestmark = pytest.mark.usefixtures("cache_from_mock")


def _find_item_store(plan: dict, item_keyword: str) -> str | None:
    for sid, entries in plan["plan"].items():
        for e in entries:
            if item_keyword.lower() in e["item"].lower():
                return sid
    return None


class TestFindAtStore:
    def test_finds_pork_at_trader_joes(self):
        data = load_prices()
        result = find_at_store("pork", "trader_joes_shadyside", data)
        assert result is not None
        assert "trader joe" in result["store"].lower()

    def test_none_when_store_has_no_item(self):
        data = load_prices()
        # walmart_crafton doesn't have bananas in the mock produce list
        # (use a category we know is limited); any miss is fine here.
        # Use a fake store id to deterministically hit the None path.
        result = find_at_store("milk", "nonexistent_store_id", data)
        assert result is None


class TestApplyPreferredStores:
    def test_moves_pork_to_trader_joes(self):
        # Baseline: with just "pork", the cheapest store in mock is Aldi.
        plan = optimize_shopping_list(["pork"])
        baseline_store = _find_item_store(plan, "pork")
        assert baseline_store is not None

        # Now force it to Trader Joe's
        plan = optimize_shopping_list(["pork"])
        plan = apply_preferred_stores(
            plan, {"pork": ["trader_joes_shadyside"]}
        )
        assert _find_item_store(plan, "pork") == "trader_joes_shadyside"
        assert plan.get("unfulfilled_preferences") == []

    def test_noop_when_already_at_preferred(self):
        # Bananas in mock are cheapest at Trader Joe's already
        plan = optimize_shopping_list(["bananas"])
        assert _find_item_store(plan, "banana") == "trader_joes_shadyside"
        plan = apply_preferred_stores(
            plan, {"bananas": ["trader_joes_shadyside"]}
        )
        assert _find_item_store(plan, "banana") == "trader_joes_shadyside"
        assert plan["unfulfilled_preferences"] == []

    def test_records_unfulfilled_when_store_lacks_item(self):
        plan = optimize_shopping_list(["milk"])
        plan = apply_preferred_stores(
            plan, {"milk": ["nonexistent_store_id"]}
        )
        assert len(plan["unfulfilled_preferences"]) == 1
        entry = plan["unfulfilled_preferences"][0]
        assert entry["item"] == "milk"
        assert entry["preferred_stores"] == ["nonexistent_store_id"]

    def test_recomputes_total_and_store_ids(self):
        plan = optimize_shopping_list(["pork", "milk"])
        plan = apply_preferred_stores(
            plan, {"pork": ["trader_joes_shadyside"]}
        )
        # store_ids should match plan keys
        assert set(plan["store_ids"]) == set(plan["plan"].keys())
        # total_cost equals the sum of every line item
        computed = round(
            sum(i["price"] for items in plan["plan"].values() for i in items), 2
        )
        assert plan["total_cost"] == computed
        # stores_meta should cover every active store
        for sid in plan["store_ids"]:
            assert sid in plan["stores_meta"]


class TestApplyAvoidStores:
    def test_moves_item_away_from_avoided_store(self):
        # Bananas baseline = Trader Joe's in mock
        plan = optimize_shopping_list(["bananas"])
        assert _find_item_store(plan, "banana") == "trader_joes_shadyside"
        plan = apply_avoid_stores(
            plan, {"bananas": ["trader_joes_shadyside"]}
        )
        new_store = _find_item_store(plan, "banana")
        assert new_store is not None
        assert new_store != "trader_joes_shadyside"


class TestPreferredStoresWithSynonyms:
    """Regression: 'pork chops' (user phrase) must match 'Pork Loin Chops' in the plan."""

    def test_pork_chops_phrase_reassigns_to_tj(self):
        plan = optimize_shopping_list(["pork chops"])
        before_store = _find_item_store(plan, "pork")
        assert before_store is not None

        plan = apply_preferred_stores(
            plan, {"pork chops": ["trader_joes_shadyside"]}
        )
        after_store = _find_item_store(plan, "pork")
        assert after_store == "trader_joes_shadyside", (
            f"pork item should have moved to TJ but ended at {after_store!r}; "
            f"plan={plan['plan']}"
        )
        assert plan.get("unfulfilled_preferences") == []


class TestCombinedPreferAndAvoid:
    def test_prefer_wins_then_avoid_runs(self):
        # Put pork at TJ (prefer), then say "avoid aldi for milk"
        plan = optimize_shopping_list(["pork", "milk"])
        plan = apply_preferred_stores(plan, {"pork": ["trader_joes_shadyside"]})
        plan = apply_avoid_stores(plan, {"milk": ["aldi_greenfield"]})

        pork_store = _find_item_store(plan, "pork")
        milk_store = _find_item_store(plan, "milk")

        assert pork_store == "trader_joes_shadyside"
        assert milk_store is not None
        assert milk_store != "aldi_greenfield"
