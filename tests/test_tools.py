"""Unit tests for the tool registry (agent/tools.py).

Each tool gets a good-args path + a bad-args path. Tools that need
priced cache data opt into the shared `cache_from_mock` fixture.
"""
from __future__ import annotations

import pytest

from agent.state import AgentState
from agent import tools as tools_mod
from agent.tools import ReplySignal, run_tool


# ────────────────────────── add_items ─────────────────────────────

class TestAddItems:
    def test_appends_and_dedupes(self):
        s = AgentState()
        obs = run_tool(s, "add_items", {"items": [
            {"name": "milk", "quantity": 1, "unit": "gallon", "ambiguous": False},
            "eggs",
            {"name": "milk"},  # duplicate
        ]})
        assert obs["added"] == ["milk", "eggs"]
        assert len(s.raw_items) == 2

    def test_coerces_string_items(self):
        s = AgentState()
        obs = run_tool(s, "add_items", {"items": ["rice"]})
        assert obs["added"] == ["rice"]
        assert s.raw_items[0]["ambiguous"] is False

    def test_empty_list_is_error(self):
        s = AgentState()
        obs = run_tool(s, "add_items", {"items": []})
        assert "error" in obs


# ────────────────────────── remove_items ──────────────────────────

class TestRemoveItems:
    def test_exact_match_preserves_similar_items(self):
        s = AgentState()
        s.raw_items = [
            {"name": "orange", "quantity": 1, "unit": None, "ambiguous": False},
            {"name": "orange juice", "quantity": 1, "unit": None, "ambiguous": False},
        ]
        s.shopping_plan = {
            "plan": {
                "aldi_greenfield": [{"item": "Navel Oranges",
                                     "source_item": "orange", "price": 3.99}],
                "giant_eagle_squirrel_hill": [{"item": "Simply Orange Juice",
                                               "source_item": "orange juice", "price": 2.59}],
            },
            "total_cost": 6.58,
            "not_found": [],
            "store_ids": ["aldi_greenfield", "giant_eagle_squirrel_hill"],
            "stores_meta": {},
        }
        obs = run_tool(s, "remove_items", {"target": "orange"})
        assert "orange" in (obs["removed_items"] or [])
        # Juice should survive.
        remaining_raw = [i["name"] for i in s.raw_items]
        assert remaining_raw == ["orange juice"]
        assert len(s.shopping_plan["plan"]) == 1
        assert "giant_eagle_squirrel_hill" in s.shopping_plan["plan"]

    def test_plural_target_removes_plural_not_juice(self):
        """Regression: remove('orange') on ['orange juice', 'oranges']
        must hit the plural 'oranges' (exact-after-singularize) and
        leave 'orange juice' alone."""
        s = AgentState()
        s.raw_items = [
            {"name": "orange juice", "quantity": 1, "unit": None, "ambiguous": False},
            {"name": "oranges", "quantity": None, "unit": None, "ambiguous": False},
        ]
        obs = run_tool(s, "remove_items", {"target": "orange"})
        assert obs.get("ambiguous") is not True, f"expected unambiguous hit, got {obs}"
        remaining = [it["name"] for it in s.raw_items]
        assert "orange juice" in remaining
        assert "oranges" not in remaining
        assert obs["removed_items"] == ["oranges"]

    def test_singular_target_removes_singular_not_juice(self):
        """Symmetric: remove('oranges') on ['orange juice', 'orange']
        must hit the singular 'orange', not the juice."""
        s = AgentState()
        s.raw_items = [
            {"name": "orange juice", "quantity": 1, "unit": None, "ambiguous": False},
            {"name": "orange", "quantity": None, "unit": None, "ambiguous": False},
        ]
        run_tool(s, "remove_items", {"target": "oranges"})
        remaining = [it["name"] for it in s.raw_items]
        assert remaining == ["orange juice"]

    def test_ambiguous_returns_matches_without_mutating(self):
        s = AgentState()
        s.raw_items = [
            {"name": "orange mango", "quantity": 1, "unit": None, "ambiguous": False},
            {"name": "orange juice", "quantity": 1, "unit": None, "ambiguous": False},
        ]
        obs = run_tool(s, "remove_items", {"target": "orange"})
        assert obs.get("ambiguous") is True
        assert set(obs["matches"]) == {"orange mango", "orange juice"}
        # State is untouched.
        assert len(s.raw_items) == 2

    def test_emptying_clears_plan(self):
        s = AgentState()
        s.raw_items = [{"name": "water", "quantity": 1, "unit": "bottle", "ambiguous": False}]
        s.shopping_plan = {
            "plan": {"aldi_greenfield": [{"item": "Sparkling Water",
                                          "source_item": "water 1 bottle", "price": 0.65}]},
            "total_cost": 0.65, "not_found": [], "store_ids": ["aldi_greenfield"],
            "stores_meta": {},
        }
        run_tool(s, "remove_items", {"target": "water"})
        assert s.raw_items == []
        assert s.shopping_plan is None

    def test_blank_target_is_error(self):
        s = AgentState()
        obs = run_tool(s, "remove_items", {"target": "   "})
        assert "error" in obs


# ────────────────────────── update_quantity ───────────────────────

class TestUpdateQuantity:
    def test_updates_matching_row(self):
        s = AgentState()
        s.raw_items = [{"name": "pork chops", "quantity": None, "unit": None, "ambiguous": True}]
        obs = run_tool(s, "update_quantity", {"name": "pork chops", "quantity": 2, "unit": "lb"})
        assert obs["updated"] == 1
        assert s.raw_items[0]["quantity"] == 2
        assert s.raw_items[0]["ambiguous"] is False

    def test_noop_when_no_match(self):
        s = AgentState()
        obs = run_tool(s, "update_quantity", {"name": "ghost", "quantity": 1})
        assert obs["updated"] == 0

    def test_requires_qty_or_unit(self):
        s = AgentState()
        s.raw_items = [{"name": "milk"}]
        obs = run_tool(s, "update_quantity", {"name": "milk"})
        assert "error" in obs


# ────────────────────────── clear_list ────────────────────────────

class TestClearList:
    def test_wipes_everything(self):
        s = AgentState()
        s.raw_items = [{"name": "milk"}]
        s.shopping_plan = {"plan": {"x": []}, "total_cost": 0, "not_found": [],
                            "store_ids": ["x"], "stores_meta": {}}
        s.preferences = {"chicken": ["x"]}
        s.pending_dish = {"name": "carbonara", "ingredients": []}
        run_tool(s, "clear_list", {})
        assert s.raw_items == []
        assert s.shopping_plan is None
        assert s.preferences == {}
        assert s.pending_dish is None


# ────────────────────────── preferences ───────────────────────────

class TestSetPreference:
    def test_avoid_merges(self):
        s = AgentState()
        run_tool(s, "set_preference", {"item": "chicken",
                                         "store_id": "trader_joes_shadyside", "kind": "avoid"})
        run_tool(s, "set_preference", {"item": "chicken",
                                         "store_id": "walmart_crafton", "kind": "avoid"})
        assert s.preferences["chicken"] == ["trader_joes_shadyside", "walmart_crafton"]

    def test_prefer_uses_separate_bucket(self):
        s = AgentState()
        run_tool(s, "set_preference", {"item": "pork",
                                         "store_id": "trader_joes_shadyside", "kind": "prefer"})
        assert s.preferred_stores["pork"] == ["trader_joes_shadyside"]
        assert s.preferences == {}

    def test_invalid_kind_is_error(self):
        s = AgentState()
        obs = run_tool(s, "set_preference", {"item": "pork",
                                              "store_id": "trader_joes_shadyside",
                                              "kind": "something"})
        assert "error" in obs

    def test_unknown_store_id_handled(self):
        s = AgentState()
        obs = run_tool(s, "set_preference", {"item": "pork",
                                              "store_id": "mars_hypermart",
                                              "kind": "prefer"})
        assert obs["ok"] is False
        assert "error" in obs


class TestUnsetPreference:
    def test_removes_specific_store(self):
        s = AgentState()
        s.preferences = {"chicken": ["trader_joes_shadyside", "walmart_crafton"]}
        run_tool(s, "unset_preference", {"item": "chicken",
                                           "store_id": "walmart_crafton"})
        assert s.preferences["chicken"] == ["trader_joes_shadyside"]

    def test_removes_all_when_store_omitted(self):
        s = AgentState()
        s.preferences = {"chicken": ["a", "b"]}
        s.preferred_stores = {"chicken": ["c"]}
        run_tool(s, "unset_preference", {"item": "chicken"})
        assert "chicken" not in s.preferences
        assert "chicken" not in s.preferred_stores


# ────────────────────────── destinations ──────────────────────────

class TestDestinations:
    def test_add_by_label_hits_landmark_dict(self):
        s = AgentState()
        obs = run_tool(s, "add_destination", {"label": "CMU"})
        assert obs["ok"] is True
        assert obs["source"] == "landmark"
        assert len(s.destinations) == 1
        assert s.destinations[0]["label"] == "CMU"
        assert 40 < s.destinations[0]["lat"] < 41
        assert -80.5 < s.destinations[0]["lng"] < -79.5

    def test_add_with_explicit_coords_skips_geocode(self):
        s = AgentState()
        obs = run_tool(s, "add_destination", {
            "label": "Secret Lab",
            "address": "10 Unknown Rd, Pittsburgh, PA",
            "lat": 40.4500, "lng": -79.9500,
        })
        assert obs["ok"] is True
        assert obs["source"] == "user_coords"
        assert s.destinations[0]["address"] == "10 Unknown Rd, Pittsburgh, PA"

    def test_add_unknown_label_without_coords_returns_error(self):
        s = AgentState()
        obs = run_tool(s, "add_destination", {"label": "zzz-nonexistent-place-xyz"})
        assert obs.get("ok") is False
        assert "error" in obs
        assert s.destinations == []

    def test_add_twice_deduplicates_by_label(self):
        s = AgentState()
        run_tool(s, "add_destination", {"label": "CMU"})
        run_tool(s, "add_destination", {"label": "cmu"})  # case-insensitive
        assert len(s.destinations) == 1

    def test_add_invalidates_route_plan(self):
        s = AgentState()
        s.route_plan = {"ordered_stops": [{"store_id": "x"}]}
        s.errand_quote = {"total": 10}
        run_tool(s, "add_destination", {"label": "CMU"})
        assert s.route_plan is None
        assert s.errand_quote is None

    def test_remove_by_label(self):
        s = AgentState()
        run_tool(s, "add_destination", {"label": "CMU"})
        run_tool(s, "add_destination", {"label": "Shadyside"})
        obs = run_tool(s, "remove_destination", {"label": "cmu"})
        assert obs["removed"] == 1
        assert [d["label"] for d in s.destinations] == ["Shadyside"]

    def test_remove_missing_is_noop(self):
        s = AgentState()
        obs = run_tool(s, "remove_destination", {"label": "Nowhere"})
        assert obs["removed"] == 0

    def test_clear_wipes_all(self):
        s = AgentState()
        run_tool(s, "add_destination", {"label": "CMU"})
        run_tool(s, "add_destination", {"label": "Pitt"})
        obs = run_tool(s, "clear_destinations", {})
        assert obs["cleared"] == 2
        assert s.destinations == []

    def test_destinations_appear_in_llm_view(self):
        s = AgentState()
        run_tool(s, "add_destination", {"label": "CMU"})
        view = s.to_llm_view()
        assert view["destinations"] == [{"label": "CMU", "address": "CMU"}]


# ────────────────────────── errand ─────────────────────────────────

class TestSetErrand:
    def test_toggles_flag(self):
        s = AgentState()
        obs = run_tool(s, "set_errand", {"want_errand": True})
        assert s.want_errand is True and obs["want_errand"] is True
        run_tool(s, "set_errand", {"want_errand": False})
        assert s.want_errand is False


# ────────────────────────── reply terminator ─────────────────────

class TestReply:
    def test_raises_signal(self):
        s = AgentState()
        with pytest.raises(ReplySignal) as exc:
            tools_mod.tool_reply(s, {"text": "hi"})
        assert exc.value.text == "hi"

    def test_blank_text_is_error(self):
        s = AgentState()
        obs = run_tool(s, "reply", {"text": ""})
        assert "error" in obs


# ────────────────────────── dispatcher ─────────────────────────────

class TestDispatcher:
    def test_unknown_tool_returns_error(self):
        s = AgentState()
        obs = run_tool(s, "does_not_exist", {})
        assert "error" in obs
        assert "available" in obs

    def test_pick_option_without_staged(self):
        s = AgentState()
        obs = run_tool(s, "pick_option", {"n": 1})
        assert "error" in obs


# ────────────────────────── cache-backed tools ───────────────────

pytestmark_cache = pytest.mark.usefixtures("cache_from_mock")


class TestCacheBackedTools:
    pytestmark = pytestmark_cache

    def test_search_products_returns_hits(self):
        s = AgentState()
        obs = run_tool(s, "search_products", {"query": "milk", "topk": 3})
        assert "results" in obs
        assert len(obs["results"]) > 0

    def test_find_at_store_hits(self):
        s = AgentState()
        obs = run_tool(s, "find_at_store", {"item": "pork",
                                             "store_id": "trader_joes_shadyside"})
        assert obs["found"] is True

    def test_find_at_store_misses(self):
        s = AgentState()
        obs = run_tool(s, "find_at_store", {"item": "ghost-ingredient-xyz",
                                             "store_id": "trader_joes_shadyside"})
        assert obs["found"] is False

    def test_list_options_stages_and_pick_locks_in(self):
        s = AgentState()
        run_tool(s, "list_options", {"query": "milk", "topk": 2})
        assert len(s.last_options) > 0
        obs = run_tool(s, "pick_option", {"n": 1})
        assert obs.get("item_name")
        assert s.last_options == []
        assert s.shopping_plan is not None
        assert s.shopping_plan["total_cost"] > 0


# ────────────────────────── optimize_and_route ───────────────────

class TestOptimizeAndRoute:
    pytestmark = pytestmark_cache

    def test_empty_list_is_error(self):
        s = AgentState()
        obs = run_tool(s, "optimize_and_route", {})
        assert "error" in obs

    def test_builds_plan_and_route(self, monkeypatch):
        # Stub route_planner to avoid ORS calls.
        monkeypatch.setattr(
            "agent.tools.plan_route",
            lambda **kw: {"ordered_stops": kw.get("store_ids"), "total_minutes": 10},
        )
        s = AgentState()
        s.raw_items = [
            {"name": "milk", "quantity": 1, "unit": "gallon", "ambiguous": False},
            {"name": "eggs", "quantity": 1, "unit": "dozen", "ambiguous": False},
        ]
        obs = run_tool(s, "optimize_and_route", {})
        assert obs["ok"] is True
        assert obs["total_cost"] > 0
        assert s.shopping_plan and s.shopping_plan["plan"]


# ────────────────────────── dish flow ─────────────────────────────

class TestDishFlow:
    def test_propose_then_apply(self):
        s = AgentState()
        obs = run_tool(s, "propose_dish", {"name": "carbonara"})
        assert obs["found"] is True
        assert s.pending_dish is not None
        applied = run_tool(s, "apply_pending_dish", {})
        assert s.pending_dish is None
        assert applied["added"]
        # pantry staples skipped by default.
        assert all("salt" != (i["name"] or "").lower() for i in s.raw_items)

    def test_apply_only_cherry_picks(self):
        s = AgentState()
        run_tool(s, "propose_dish", {"name": "carbonara"})
        applied = run_tool(s, "apply_pending_dish", {"only": [1]})
        assert len(applied["added"]) == 1

    def test_propose_unknown_dish(self, monkeypatch):
        monkeypatch.setattr("config.settings.USE_LLM_DISH_FALLBACK", False)
        s = AgentState()
        obs = run_tool(s, "propose_dish", {"name": "definitely-not-a-dish-xyz"})
        assert obs["found"] is False

    def test_cancel_clears(self):
        s = AgentState()
        run_tool(s, "propose_dish", {"name": "carbonara"})
        run_tool(s, "cancel_pending_dish", {})
        assert s.pending_dish is None


# ────────────────────────── justify ─────────────────────────────

class TestJustify:
    def test_no_plan_returns_note(self):
        s = AgentState()
        obs = run_tool(s, "justify_pick", {"target": "milk"})
        assert obs["hits"] == []
        assert "note" in obs

    def test_matches_and_traces(self):
        s = AgentState()
        s.raw_items = [{"name": "milk"}]
        s.shopping_plan = {
            "plan": {"aldi_greenfield": [{"item": "Organic Whole Milk 1 Gallon",
                                          "source_item": "milk", "price": 3.19}]},
            "total_cost": 3.19, "not_found": [], "store_ids": ["aldi_greenfield"],
            "stores_meta": {},
        }
        obs = run_tool(s, "justify_pick", {"target": "milk"})
        assert len(obs["hits"]) == 1
        assert obs["hits"][0]["item"].lower().startswith("organic whole milk")
