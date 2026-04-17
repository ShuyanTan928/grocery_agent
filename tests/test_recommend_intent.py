"""Unit tests for the recommend side-flow: intent detection (regex only)
and recommender preference rendering. These do not call the LLM."""
from __future__ import annotations

import pytest

from agent.agent import detect_recommend_intent
from tools.recommender import _expand_preferences, format_recommendation


# ---------- intent detection ----------------------------------------------

@pytest.mark.parametrize("msg, query", [
    ("recommend pork", "pork"),
    ("recommend me some milk", "milk"),
    ("what's the best ground beef?", "ground beef"),
    ("what is the best chicken to buy", "chicken"),
    ("best yogurt", "yogurt"),
    ("which pasta should I buy?", "pasta"),
    ("suggest some snacks", "snacks"),
    ("top 5 olive oil", "olive oil"),
])
def test_intent_detected(msg, query):
    intent = detect_recommend_intent(msg)
    assert intent is not None, f"failed to detect: {msg!r}"
    assert intent["query"] == query


@pytest.mark.parametrize("msg", [
    "I need pork and milk",
    "add bananas to my list",
    "yes",
    "no preferences",
    "",
])
def test_intent_not_detected(msg):
    assert detect_recommend_intent(msg) is None


def test_topk_extracted():
    intent = detect_recommend_intent("top 5 ice cream")
    assert intent["topk"] == 5
    intent = detect_recommend_intent("recommend best 7 yogurt")
    assert intent["topk"] == 7


def test_topk_default_is_3():
    intent = detect_recommend_intent("recommend pork")
    assert intent["topk"] == 3


def test_topk_clamped():
    # > 10 should clamp to 10
    intent = detect_recommend_intent("top 99 things")
    assert intent["topk"] == 10


def test_preferences_inferred():
    intent = detect_recommend_intent("recommend organic milk")
    assert "organic" in intent["preferences"]

    intent = detect_recommend_intent("what's the cheapest pasta")
    assert "cheapest" in intent["preferences"]

    intent = detect_recommend_intent("best premium ice cream")
    assert "premium" in intent["preferences"]

    intent = detect_recommend_intent("recommend grass-fed ground beef")
    assert "organic" in intent["preferences"]  # grass-fed maps to organic flag


def test_preferences_default_empty():
    intent = detect_recommend_intent("recommend milk")
    assert intent["preferences"] == []


# ---------- preference expansion ------------------------------------------

def test_expand_preferences_known_flags():
    out = _expand_preferences(["organic", "cheapest"])
    assert "organic" in out.lower() or "ORGANIC" in out
    assert "lowest absolute price" in out.lower()


def test_expand_preferences_brand_flag():
    out = _expand_preferences(["brand:Just Bare"])
    assert "Just Bare" in out


def test_expand_preferences_freeform_passthrough():
    out = _expand_preferences(["I hate pickles"])
    assert "I hate pickles" in out


def test_expand_preferences_empty():
    assert _expand_preferences(None) == ""
    assert _expand_preferences([]) == ""


# ---------- format_recommendation rendering -------------------------------

def test_format_recommendation_with_picks():
    result = {
        "query": "milk",
        "topk": 2,
        "candidates": [],
        "picks": [
            {"rank": 1, "candidate": {
                "store": "aldi", "price": 3.19, "name": "Whole Milk 1 Gal",
                "size": "1 gal", "brand": "", "unit_price": "$3.19/gal"
            }, "reason": "cheapest"},
        ],
        "summary": "Aldi wins.",
        "timings": {"search_ms": 5.0, "llm_ms": 800.0},
        "raw_llm": "",
    }
    out = format_recommendation(result)
    assert "Aldi wins." in out
    assert "Whole Milk 1 Gal" in out
    assert "cheapest" in out


def test_format_recommendation_no_picks():
    result = {"picks": [], "summary": "no matches"}
    assert format_recommendation(result) == "no matches"
