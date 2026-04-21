"""Tests for recommender prompt helpers (no LLM calls)."""

from __future__ import annotations

from tools.recommender import line_item_pick_hints


class TestLineItemPickHints:
    def test_milk_line_includes_beverage_constraint(self):
        h = line_item_pick_hints("1 gallon milk")
        assert "beverage" in h.lower() or "cow" in h.lower()
        assert "milk chocolate" in h.lower()

    def test_milk_chocolate_line_skips_milk_beverage_block(self):
        h = line_item_pick_hints("milk chocolate chips for cookies")
        assert h == "" or "beverage" not in h.lower()

    def test_eggs_line(self):
        h = line_item_pick_hints("dozen eggs")
        assert "shell eggs" in h.lower() or "cooking" in h.lower()

    def test_empty(self):
        assert line_item_pick_hints("") == ""
        assert line_item_pick_hints("   ") == ""

    def test_unrelated_line(self):
        assert line_item_pick_hints("paper towels") == ""
