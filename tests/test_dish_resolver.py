"""Unit tests for tools.dish_resolver: seed lookups, alias matching,
fuzzy substring match, LLM fallback (stubbed), caching, and the
ingredients→raw_items projection used by the agent."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import dish_resolver as dr


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Redirect dishes_cache.json to a temp file and reset the in-memory
    seed cache between tests so state doesn't leak."""
    monkeypatch.setattr(dr, "CACHE_PATH", tmp_path / "dishes_cache.json")
    monkeypatch.setattr(dr, "_SEED_CACHE", None)
    yield
    monkeypatch.setattr(dr, "_SEED_CACHE", None)


def test_normalize_basic():
    assert dr._normalize("  Spaghetti Carbonara!  ") == "spaghetti carbonara"
    assert dr._normalize("MA-PO tofu?") == "ma po tofu"
    assert dr._normalize("") == ""


def test_resolve_exact_seed_key():
    d = dr.resolve_dish("spaghetti carbonara")
    assert d is not None
    assert d["source"] == "seed"
    assert d["name"] == "spaghetti carbonara"
    assert any(i["name"] == "spaghetti" for i in d["ingredients"])


def test_resolve_alias():
    d = dr.resolve_dish("carbonara")
    assert d is not None
    assert d["source"] == "seed"
    assert d["name"] == "spaghetti carbonara"


def test_resolve_case_insensitive_punctuation():
    d = dr.resolve_dish("Mapo Tofu!")
    assert d is not None
    assert d["name"] == "mapo tofu"


def test_resolve_fuzzy_substring_prefers_longest():
    # "carbonara" should still resolve to "spaghetti carbonara".
    d = dr.resolve_dish("pasta carbonara")
    assert d is not None
    assert d["name"] == "spaghetti carbonara"


def test_resolve_unknown_without_llm_returns_none(monkeypatch):
    monkeypatch.setattr("config.settings.USE_LLM_DISH_FALLBACK", False)
    assert dr.resolve_dish("zzz not a real dish") is None


def test_llm_fallback_caches(monkeypatch, tmp_path):
    monkeypatch.setattr("config.settings.USE_LLM_DISH_FALLBACK", True)

    fake_response = {
        "dish": "ropa vieja",
        "cuisine": "cuban",
        "servings": 2,
        "ingredients": [
            {"name": "flank steak", "quantity": 1, "unit": "lb", "pantry": False},
            {"name": "canned tomato sauce", "quantity": 400, "unit": "g", "pantry": False},
            {"name": "onion", "quantity": 1, "unit": None, "pantry": False},
            {"name": "salt", "quantity": None, "unit": None, "pantry": True},
        ],
    }
    monkeypatch.setattr(dr, "_call_llm_for_dish", lambda name: fake_response)

    d = dr.resolve_dish("ropa vieja")
    assert d is not None
    assert d["source"] == "llm"
    assert d["cuisine"] == "cuban"
    assert any(i["name"] == "flank steak" for i in d["ingredients"])

    # Next call should read from the persistent cache, not re-call the LLM.
    def boom(name):
        raise AssertionError("LLM should not be re-queried")

    monkeypatch.setattr(dr, "_call_llm_for_dish", boom)
    d2 = dr.resolve_dish("ropa vieja")
    assert d2 is not None
    assert d2["source"] == "cache"
    assert any(i["name"] == "flank steak" for i in d2["ingredients"])


def test_llm_fallback_rejects_malformed(monkeypatch):
    monkeypatch.setattr("config.settings.USE_LLM_DISH_FALLBACK", True)
    # Missing ingredients list
    monkeypatch.setattr(dr, "_call_llm_for_dish", lambda name: {"dish": "x"})
    assert dr.resolve_dish("mystery thing") is None


def test_ingredients_to_raw_items_skips_pantry():
    ings = [
        {"name": "spaghetti", "quantity": 200, "unit": "g"},
        {"name": "salt", "quantity": None, "unit": None, "pantry": True},
        {"name": "eggs", "quantity": 2, "unit": None},
    ]
    out = dr.ingredients_to_raw_items(ings)
    names = [r["name"] for r in out]
    assert "spaghetti" in names
    assert "eggs" in names
    assert "salt" not in names


def test_ingredients_to_raw_items_include_pantry_flag():
    ings = [
        {"name": "salt", "quantity": None, "unit": None, "pantry": True},
        {"name": "eggs", "quantity": 2, "unit": None},
    ]
    out = dr.ingredients_to_raw_items(ings, include_pantry=True)
    names = [r["name"] for r in out]
    assert names == ["salt", "eggs"]
    for row in out:
        assert row["ambiguous"] is False


def test_list_seed_dishes_nonempty():
    names = dr.list_seed_dishes()
    assert "spaghetti carbonara" in names
    assert "mapo tofu" in names
    assert len(names) >= 10
