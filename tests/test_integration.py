# ============================================================
# tests/test_integration.py
# End-to-end integration test: simulates a full agent run
# without calling any real APIs (mock data only).
#
# Run with: pytest tests/test_integration.py -v
# ============================================================

import pytest
from tools.price_optimizer import optimize_shopping_list
from tools.route_planner import plan_route
from tools.errand_runner import generate_errand_quote


pytestmark = pytest.mark.usefixtures("cache_from_mock")


class TestFullShoppingPipeline:
    """
    Integration tests that chain optimizer -> route planner -> errand runner
    the same way agent.py does at runtime.
    """

    SAMPLE_LIST = ["milk", "eggs", "bread", "chicken", "bananas"]

    def test_pipeline_produces_complete_plan(self):
        # Step 1: optimize
        shopping = optimize_shopping_list(self.SAMPLE_LIST)
        assert shopping["total_cost"] > 0
        assert len(shopping["store_ids"]) > 0

        # Step 2: route
        route = plan_route(shopping["store_ids"], shopping["stores_meta"])
        assert len(route["ordered_stops"]) == len(shopping["store_ids"])

        # Step 3: errand quote
        quote = generate_errand_quote(shopping, route)
        assert quote["grand_total"] > shopping["total_cost"]  # fee added
        assert quote["num_stores"] == len(shopping["store_ids"])

    def test_all_items_accounted_for(self):
        shopping = optimize_shopping_list(self.SAMPLE_LIST)
        found_items = [
            item["item"]
            for items in shopping["plan"].values()
            for item in items
        ]
        # All sample items exist in mock data, so none should be missing
        assert len(found_items) == len(self.SAMPLE_LIST)
        assert shopping["not_found"] == []

    def test_errand_quote_total_is_correct(self):
        shopping = optimize_shopping_list(["milk", "eggs"])
        route = plan_route(shopping["store_ids"], shopping["stores_meta"])
        quote = generate_errand_quote(shopping, route, tip_pct=0.0)

        # With 0% tip, grand total = groceries + service fee only
        expected = shopping["total_cost"] + quote["service_fee"]
        assert abs(quote["grand_total"] - expected) < 0.01

    def test_single_item_list_works(self):
        shopping = optimize_shopping_list(["eggs"])
        route = plan_route(shopping["store_ids"], shopping["stores_meta"])
        assert len(route["ordered_stops"]) == 1

    def test_all_unknown_items(self):
        shopping = optimize_shopping_list(["flying pig", "moon cheese"])
        assert shopping["total_cost"] == 0.0
        assert shopping["store_ids"] == []
        assert len(shopping["not_found"]) == 2
