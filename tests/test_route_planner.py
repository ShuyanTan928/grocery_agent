# ============================================================
# tests/test_route_planner.py
# Unit tests for the route planner tool using mock distance data.
# Run with: pytest tests/test_route_planner.py -v
# ============================================================

import pytest
from tools.route_planner import (
    plan_route,
    solve_tsp_brute_force,
    solve_tsp_greedy,
    get_mock_distance_matrix,
)

# Reusable mock stores_meta for testing
MOCK_META = {
    "aldi_greenfield": {
        "name": "Aldi", "branch": "Greenfield",
        "address": "3850 Bigelow Blvd, Pittsburgh, PA 15213",
        "lat": 40.4442, "lng": -79.9558,
    },
    "walmart_crafton": {
        "name": "Walmart", "branch": "Crafton",
        "address": "200 Bower Hill Rd, Pittsburgh, PA 15243",
        "lat": 40.4051, "lng": -80.0611,
    },
    "trader_joes_shadyside": {
        "name": "Trader Joe's", "branch": "Shadyside",
        "address": "6343 Penn Ave, Pittsburgh, PA 15206",
        "lat": 40.4583, "lng": -79.9256,
    },
}


class TestTSPSolvers:
    """Tests for the TSP solving algorithms."""

    def test_brute_force_single_node(self):
        matrix = [[0]]
        result = solve_tsp_brute_force(matrix, 1)
        assert result == [0]

    def test_brute_force_two_nodes(self):
        # Both orders have same cost (symmetric), just check valid output
        matrix = [[0, 10], [10, 0]]
        result = solve_tsp_brute_force(matrix, 2)
        assert set(result) == {0, 1}

    def test_brute_force_picks_shorter_path(self):
        # A->B->C costs 1+1=2, A->C->B costs 10+1=11
        matrix = [
            [0, 1, 10],
            [1, 0, 1],
            [10, 1, 0],
        ]
        result = solve_tsp_brute_force(matrix, 3)
        assert result == [0, 1, 2]

    def test_greedy_visits_all_nodes(self):
        matrix = [[0, 5, 10], [5, 0, 3], [10, 3, 0]]
        result = solve_tsp_greedy(matrix, 3)
        assert sorted(result) == [0, 1, 2]

    def test_greedy_starts_at_zero(self):
        matrix = [[0, 5, 10], [5, 0, 3], [10, 3, 0]]
        result = solve_tsp_greedy(matrix, 3)
        assert result[0] == 0


class TestMockDistanceMatrix:
    """Tests for mock data loading and slicing."""

    def test_loads_two_stores(self):
        store_ids = ["aldi_greenfield", "walmart_crafton"]
        result = get_mock_distance_matrix(store_ids)
        assert len(result["durations"]) == 2
        assert len(result["durations"][0]) == 2
        # Diagonal must be 0 (same store to itself)
        assert result["durations"][0][0] == 0

    def test_all_five_stores(self):
        store_ids = [
            "giant_eagle_squirrel_hill",
            "aldi_greenfield",
            "walmart_crafton",
            "trader_joes_shadyside",
            "whole_foods_east_liberty",
        ]
        result = get_mock_distance_matrix(store_ids)
        assert len(result["durations"]) == 5


class TestPlanRoute:
    """Integration tests for the full route planner."""

    def test_single_store_returns_immediately(self):
        result = plan_route(
            store_ids=["aldi_greenfield"],
            stores_meta=MOCK_META,
        )
        assert len(result["ordered_stops"]) == 1
        assert result["ordered_stops"][0]["store_id"] == "aldi_greenfield"

    def test_two_stores_returns_both(self):
        result = plan_route(
            store_ids=["aldi_greenfield", "walmart_crafton"],
            stores_meta=MOCK_META,
        )
        assert len(result["ordered_stops"]) == 2
        visited = {s["store_id"] for s in result["ordered_stops"]}
        assert visited == {"aldi_greenfield", "walmart_crafton"}

    def test_total_duration_is_positive(self):
        result = plan_route(
            store_ids=["aldi_greenfield", "walmart_crafton"],
            stores_meta=MOCK_META,
        )
        assert result["total_duration_min"] > 0
        assert result["total_distance_km"] > 0

    def test_empty_store_list(self):
        result = plan_route(store_ids=[], stores_meta={})
        assert result["ordered_stops"] == []

    def test_ors_url_is_generated(self):
        result = plan_route(
            store_ids=["aldi_greenfield", "trader_joes_shadyside"],
            stores_meta=MOCK_META,
        )
        assert result["ors_directions_url"] is not None
        assert "openrouteservice.org" in result["ors_directions_url"]
