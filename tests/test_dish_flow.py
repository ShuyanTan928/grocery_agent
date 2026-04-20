"""End-to-end tests for the dish → ingredients side-flow wired into
agent.agent.chat(). Covers:
  - detect_dish_intent on common phrasings (and negatives)
  - handle_dish_request stages a proposal on session.pending_dish
  - dish-confirm: yes / no / numeric-pick / include-pantry paths
  - integration with chat(): proposal → confirm → CONFIRM state

All LLM calls are monkeypatched.
"""
from __future__ import annotations

import pytest

from agent import agent as a
from agent.agent import (
    ShoppingSession,
    chat,
    detect_dish_intent,
    handle_dish_request,
    handle_dish_confirm,
)


@pytest.fixture
def patched(monkeypatch):
    """Neutralize all LLM + execute hooks so dish-flow tests don't
    touch the network. Returns a call counter so assertions can check
    what fired."""
    calls = {"call_llm": 0, "execute": 0, "update_qty": 0, "extract_prefs": 0}

    def fake_call_llm(prompt, **kw):
        calls["call_llm"] += 1
        return "[confirm-template]"

    def fake_execute(session):
        calls["execute"] += 1
        return "[EXECUTED]"

    def fake_update_qty(msg, items):
        calls["update_qty"] += 1
        return items

    def fake_extract_prefs(msg, avoid, prefer):
        calls["extract_prefs"] += 1
        return avoid, prefer

    monkeypatch.setattr(a, "call_llm", fake_call_llm)
    monkeypatch.setattr(a, "session_execute", fake_execute)
    monkeypatch.setattr(a, "update_quantities_from_reply", fake_update_qty)
    monkeypatch.setattr(a, "extract_preferences_from_reply", fake_extract_prefs)
    return calls


# ---------- detect_dish_intent --------------------------------------------

@pytest.mark.parametrize("msg, expected_contains", [
    ("I want to make carbonara", "carbonara"),
    ("i wanna cook mapo tofu", "mapo tofu"),
    ("recipe for butter chicken", "butter chicken"),
    ("ingredients for pad thai", "pad thai"),
    ("what do i need to make tacos", "tacos"),
    ("let's make pancakes", "pancakes"),
    ("i feel like guacamole", "guacamole"),
    ("i'd like to cook miso soup tonight", "miso soup"),
])
def test_detect_dish_intent_positive(msg, expected_contains):
    got = detect_dish_intent(msg)
    assert got is not None
    assert expected_contains in got.lower()


@pytest.mark.parametrize("msg", [
    "I need milk and eggs",
    "yes",
    "no thanks",
    "recommend carbonara sauce",       # recommend side-flow, not dish
    "",
    "i want to cook you dinner",        # stopword first-token
])
def test_detect_dish_intent_negative(msg):
    assert detect_dish_intent(msg) is None


# ---------- handle_dish_request -------------------------------------------

def test_handle_dish_request_stages_proposal():
    s = ShoppingSession()
    reply = handle_dish_request(s, "carbonara")
    assert s.pending_dish is not None
    assert s.pending_dish["name"] == "spaghetti carbonara"
    assert "Add all" in reply
    # Non-pantry ingredients are numbered in the prompt
    assert "1." in reply


def test_handle_dish_request_unknown_dish(monkeypatch):
    monkeypatch.setattr("config.settings.USE_LLM_DISH_FALLBACK", False)
    s = ShoppingSession()
    reply = handle_dish_request(s, "zzz not a real dish")
    assert s.pending_dish is None
    assert "don't have a recipe" in reply.lower()


# ---------- handle_dish_confirm -------------------------------------------

def _staged_session() -> ShoppingSession:
    s = ShoppingSession()
    handle_dish_request(s, "carbonara")
    assert s.pending_dish is not None
    return s


def test_confirm_yes_adds_non_pantry(patched):
    s = _staged_session()
    reply = handle_dish_confirm(s, "yes")
    # salt / black pepper / garlic are pantry and must be skipped
    names = [i["name"] for i in s.raw_items]
    assert "spaghetti" in names
    assert "bacon" in names
    assert "salt" not in names
    assert "black pepper" not in names
    assert s.pending_dish is None
    assert s.state == "CONFIRM"
    assert reply == "[confirm-template]"  # our fake CONFIRM prompt
    assert patched["call_llm"] == 1


def test_confirm_no_drops(patched):
    s = _staged_session()
    reply = handle_dish_confirm(s, "no thanks")
    assert s.pending_dish is None
    assert s.raw_items == []
    assert s.state == "CLARIFY"
    assert "dropped" in reply.lower()
    assert patched["call_llm"] == 0  # no CONFIRM template


def test_confirm_numeric_selection(patched):
    s = _staged_session()
    # carbonara non-pantry order: spaghetti(1) bacon(2) eggs(3) parmesan(4)
    handle_dish_confirm(s, "1 3")
    names = [i["name"] for i in s.raw_items]
    assert names == ["spaghetti", "eggs"]
    assert s.state == "CONFIRM"


def test_confirm_include_pantry(patched):
    s = _staged_session()
    handle_dish_confirm(s, "yes, with pantry")
    names = [i["name"] for i in s.raw_items]
    # include_pantry should pull in garlic + black pepper + salt
    assert "salt" in names
    assert "garlic" in names


def test_confirm_dedupes_against_existing_list(patched):
    s = _staged_session()
    s.raw_items = [
        {"name": "spaghetti", "quantity": 1, "unit": "box", "ambiguous": False},
    ]
    handle_dish_confirm(s, "yes")
    # spaghetti was already there — should NOT duplicate
    assert sum(1 for i in s.raw_items if i["name"] == "spaghetti") == 1
    # other ingredients still spliced in
    assert any(i["name"] == "bacon" for i in s.raw_items)


def test_confirm_ambiguous_reply_keeps_proposal(patched):
    s = _staged_session()
    reply = handle_dish_confirm(s, "hmmm maybe")
    # proposal stays staged, state not advanced
    assert s.pending_dish is not None
    assert s.state == "CLARIFY"
    assert "yes" in reply.lower() and "no" in reply.lower()


# ---------- chat() integration --------------------------------------------

def test_chat_end_to_end_dish_flow(patched):
    s = ShoppingSession()

    reply1 = chat(s, "I want to make carbonara tonight")
    assert s.pending_dish is not None
    assert "Add all" in reply1
    assert s.state == "CLARIFY"          # state not advanced yet
    assert patched["execute"] == 0

    reply2 = chat(s, "yes")
    assert s.pending_dish is None
    assert s.state == "CONFIRM"           # confirm template rendered
    assert any(i["name"] == "spaghetti" for i in s.raw_items)
    # Dish-confirm runs before pick/recommend/list_options branches.
    assert patched["call_llm"] == 1


def test_chat_dish_confirm_takes_priority_over_pick(patched):
    """Even if last_options is non-empty, a pending_dish must consume the
    next turn so 'yes' applies to the dish instead of being mis-routed."""
    s = ShoppingSession()
    s.last_options = [{
        "item_name": "fake", "item_price": 1.0,
        "store_id": "aldi_greenfield", "store_display": "Aldi", "url": None,
    }]
    chat(s, "I want to make pancakes")
    assert s.pending_dish is not None
    reply = chat(s, "yes")
    # Dish branch consumed it → state is CONFIRM; pick-N was not triggered.
    assert s.state == "CONFIRM"
    assert any(i["name"] == "pancake mix" for i in s.raw_items)
