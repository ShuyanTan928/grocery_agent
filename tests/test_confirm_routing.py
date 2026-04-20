"""Unit tests for the CONFIRM-state routing in chat().

Covers the three branches we care about:
  - bare "no" → ask-what-to-change, no LLM re-parse
  - "no, <refinement>" (or any reply with substantive content) → apply
    refinement and auto-execute (no second yes/no round-trip)
  - "yes" → execute directly

These tests monkeypatch the LLM-backed helpers so no API calls are made.
"""
from __future__ import annotations

import pytest

from agent import agent as a
from agent.agent import ShoppingSession, chat


@pytest.fixture
def patched_confirm(monkeypatch):
    """Replace LLM + execute with deterministic stand-ins and a call log."""
    calls: dict[str, int] = {
        "call_llm": 0,
        "update_qty": 0,
        "extract_prefs": 0,
        "execute": 0,
    }

    def fake_call_llm(prompt):
        calls["call_llm"] += 1
        return "[confirm-ish text]"

    def fake_update_qty(msg, items):
        calls["update_qty"] += 1
        return items

    def fake_extract_prefs(msg, avoid, prefer):
        calls["extract_prefs"] += 1
        return avoid, prefer

    def fake_execute(session):
        calls["execute"] += 1
        return "[EXECUTED]"

    monkeypatch.setattr(a, "call_llm", fake_call_llm)
    monkeypatch.setattr(a, "update_quantities_from_reply", fake_update_qty)
    monkeypatch.setattr(a, "extract_preferences_from_reply", fake_extract_prefs)
    monkeypatch.setattr(a, "session_execute", fake_execute)

    # Force list_options down its deterministic (non-LLM) branch so these
    # tests don't need to stub the recommender pipeline. The dedicated
    # test_list_options_uses_llm_filter test re-enables + stubs it.
    import config.settings as cs
    monkeypatch.setattr(cs, "USE_LLM_LIST_OPTIONS", False)

    return calls


def _session_in_confirm() -> ShoppingSession:
    s = ShoppingSession()
    s.state = "CONFIRM"
    s.raw_items = [{"name": "ketchup", "quantity": 1, "unit": None, "ambiguous": False}]
    return s


def test_bare_no_asks_what_to_change(patched_confirm):
    s = _session_in_confirm()
    reply = chat(s, "no")
    assert "what would you like to change" in reply.lower()
    assert patched_confirm["execute"] == 0
    assert patched_confirm["call_llm"] == 0      # no LLM roundtrip
    assert patched_confirm["update_qty"] == 0
    assert s.state == "CONFIRM"                   # stay parked


def test_no_plus_refinement_auto_executes(patched_confirm):
    s = _session_in_confirm()
    reply = chat(s, "no, find the heinz one")
    assert reply == "[EXECUTED]"
    assert patched_confirm["execute"] == 1
    assert patched_confirm["update_qty"] == 1     # refinement applied
    assert s.state == "EXECUTE"


def test_plain_yes_executes(patched_confirm):
    s = _session_in_confirm()
    reply = chat(s, "yes")
    assert reply == "[EXECUTED]"
    assert patched_confirm["execute"] == 1
    assert patched_confirm["update_qty"] == 0     # no refinement applied
    assert s.state == "EXECUTE"


def test_long_non_yes_reply_is_treated_as_refinement(patched_confirm):
    # Even without a leading "no", a multi-word reply counts as refinement.
    s = _session_in_confirm()
    reply = chat(s, "actually add organic bananas too")
    assert reply == "[EXECUTED]"
    assert patched_confirm["execute"] == 1
    assert patched_confirm["update_qty"] == 1


def test_bare_cancel_variants(patched_confirm):
    for phrase in ["no", "nope", "nah", "cancel", "stop", "wait"]:
        patched_confirm.update({k: 0 for k in patched_confirm})
        s = _session_in_confirm()
        reply = chat(s, phrase)
        assert patched_confirm["execute"] == 0, f"{phrase!r} should not execute"
        assert "what would you like to change" in reply.lower()


# ---------- post-execute (state == "EXECUTE") routing --------------------

def _session_after_execute() -> ShoppingSession:
    s = ShoppingSession()
    s.state = "EXECUTE"
    # pretend a plan was produced in the previous turn
    s.shopping_plan = {"plan": {"x": [{"item": "x", "price": 1}]}, "total_cost": 1.0}
    return s


@pytest.mark.parametrize("phrase", [
    "no thanks",
    "No thanks",
    "nope",
    "thanks",
    "thank you",
    "nothing",
    "that's all",
    "we're done",
    "bye",
    "goodbye",
])
def test_post_execute_closer_ends_session(patched_confirm, phrase):
    s = _session_after_execute()
    reply = chat(s, phrase)
    # No LLM call, no new-list parsing, no new CONFIRM render.
    assert patched_confirm["call_llm"] == 0
    assert patched_confirm["execute"] == 0
    assert s.state == "DONE"
    assert "happy shopping" in reply.lower() or "all set" in reply.lower()


def test_post_execute_new_list_starts_fresh(patched_confirm, monkeypatch):
    # Stub out the list parser so we don't hit the real LLM.
    monkeypatch.setattr(
        a, "parse_items_from_message",
        lambda msg: [{"name": "milk", "quantity": 1, "unit": "gallon", "ambiguous": False}],
    )
    s = _session_after_execute()
    reply = chat(s, "I also need milk")
    # LLM was called to render CONFIRM for the fresh list, but execute
    # wasn't auto-triggered (we still wait for yes/no on the new list).
    assert patched_confirm["call_llm"] == 1
    assert patched_confirm["execute"] == 0
    assert s.state == "CONFIRM"
    assert s.raw_items and s.raw_items[0]["name"] == "milk"


def test_post_execute_unparseable_asks_for_real_list(patched_confirm, monkeypatch):
    # Parser returns empty (LLM couldn't find items) — we should NOT render
    # an empty CONFIRM template; just ask politely.
    monkeypatch.setattr(a, "parse_items_from_message", lambda msg: [])
    s = _session_after_execute()
    reply = chat(s, "maybe something fun")
    assert patched_confirm["call_llm"] == 0
    assert patched_confirm["execute"] == 0
    assert "didn't catch" in reply.lower() or "what would you like" in reply.lower()
    assert s.state == "CLARIFY"


# ---------- list-options (question) intent -------------------------------

from agent.agent import detect_list_options_intent, _clean_plan_item_name


@pytest.mark.parametrize("msg", [
    "can you list the options?",
    "do you have more options?",
    "what other options do you have",
    "show me more",
    "any alternatives?",
    "what else?",
    "give me some alternatives",
    "other choices?",
    "show me some options",
    "more options",
])
def test_list_options_intent_detected(msg):
    assert detect_list_options_intent(msg), f"failed to detect: {msg!r}"


@pytest.mark.parametrize("msg", [
    "yes",
    "no",
    "I need milk",
    "2 gallons",
    "no preferences",
    "recommend pork",
    "",
])
def test_list_options_intent_negative(msg):
    assert not detect_list_options_intent(msg), f"false positive: {msg!r}"


def test_clean_plan_item_name_strips_brand_size():
    assert _clean_plan_item_name("Mini Avocados (Bag)") == "avocados"
    assert _clean_plan_item_name("Heinz Tomato Ketchup, 38 oz Bottle").endswith("ketchup")
    assert _clean_plan_item_name("Organic Bananas 1 lb") == "bananas"


def test_list_options_does_not_advance_state(patched_confirm, monkeypatch):
    """The user's complaint: mid-CLARIFY question shouldn't promote to
    CONFIRM; just answer the question and stay parked."""
    # Stub the cache search so we don't depend on real data.
    from tools import product_search as ps
    monkeypatch.setattr(
        ps, "search_products_ranked",
        lambda *a_, **kw: [
            {"store_id": "aldi_greenfield", "store": "aldi",
             "item_name": "Mini Avocados (Bag)", "item_price": 3.25, "url": None},
            {"store_id": "trader_joes_shadyside", "store": "trader joe's",
             "item_name": "Organic Hass Avocados 4 Ct", "item_price": 5.99, "url": None},
        ],
    )
    s = ShoppingSession()
    s.state = "CLARIFY"
    s.raw_items = [{"name": "avocado", "quantity": None, "unit": None, "ambiguous": False}]

    reply = chat(s, "can you list the options?")
    assert s.state == "CLARIFY"                       # did NOT advance
    assert patched_confirm["call_llm"] == 0           # no LLM roundtrip
    assert patched_confirm["execute"] == 0
    assert "Mini Avocados" in reply
    assert "Organic Hass Avocados" in reply
    # raw_items should be untouched — we didn't consume the quantity turn
    assert s.raw_items[0]["name"] == "avocado"


def test_list_options_post_execute_uses_plan_item(patched_confirm, monkeypatch):
    from tools import product_search as ps
    monkeypatch.setattr(
        ps, "search_products_ranked",
        lambda *a_, **kw: [
            {"store_id": "aldi_greenfield", "store": "aldi",
             "item_name": "Mini Avocados (Bag)", "item_price": 3.25, "url": None},
        ],
    )
    s = _session_after_execute()
    s.raw_items = []  # simulate they were cleared
    s.shopping_plan = {
        "plan": {"aldi_greenfield": [{"item": "Mini Avocados (Bag)", "price": 3.25}]},
        "total_cost": 3.25,
    }

    reply = chat(s, "do you have more options?")
    assert s.state == "EXECUTE"                       # still parked at DONE-ish
    assert patched_confirm["execute"] == 0
    assert "Mini Avocados" in reply


def test_list_options_with_no_context_asks_user(patched_confirm):
    s = ShoppingSession()  # fresh, no items yet
    reply = chat(s, "show me some options")
    assert s.state == "CLARIFY"
    assert "which item" in reply.lower()
    assert patched_confirm["call_llm"] == 0
    assert patched_confirm["execute"] == 0


def test_list_options_uses_llm_filter_and_feeds_pick(patched_confirm, monkeypatch):
    """When USE_LLM_LIST_OPTIONS is on, handle_list_options_request should
    call recommend_for_query, show only the LLM-approved picks (dropping
    brand-name drift like 'Lamb Weston fries'), and stash them so a
    subsequent 'pick 1' still resolves to a real SKU."""
    # Re-enable the LLM filter just for this test (patched_confirm turns
    # it off by default for determinism).
    import config.settings as cs
    monkeypatch.setattr(cs, "USE_LLM_LIST_OPTIONS", True)

    from tools import recommender as rec

    calls = {"recommend": 0}

    def fake_recommend(query, *, topk=3, **kw):
        calls["recommend"] += 1
        assert query  # non-empty
        # Simulate the LLM correctly filtering out "Lamb Weston fries"
        # (a brand-name match that's actually potato fries) and keeping
        # only real lamb meat.
        return {
            "query": query,
            "topk": topk,
            "picks": [
                {
                    "rank": 1,
                    "candidate": {
                        "name": "Catelli Ground Lamb",
                        "price": 9.99,
                        "store_id": "giant_eagle_squirrel_hill",
                        "store": "giant_eagle",
                        "url": "https://example.com/ground-lamb",
                    },
                    "reason": "real lamb meat, reasonably priced",
                },
            ],
            "summary": "Catelli ground lamb is the best real-lamb option.",
        }

    monkeypatch.setattr(rec, "recommend_for_query", fake_recommend)
    # Stub load_stores so we show a pretty store name + can route later.
    monkeypatch.setattr(
        a, "load_stores",
        lambda: {"giant_eagle_squirrel_hill": {
            "name": "Giant Eagle (Squirrel Hill)",
            "address": "1900 Murray Ave",
        }},
    )
    monkeypatch.setattr(a, "plan_route", lambda **kw: {"stops": kw["store_ids"]})

    s = ShoppingSession()
    s.state = "CLARIFY"
    s.raw_items = [{"name": "lamb", "quantity": None, "unit": None, "ambiguous": False}]

    reply1 = chat(s, "can you list the options?")

    assert calls["recommend"] == 1                   # LLM filter ran
    assert "Catelli Ground Lamb" in reply1
    assert "Lamb Weston" not in reply1                # drift filtered
    assert "real lamb meat" in reply1                 # LLM reason rendered
    assert len(s.last_options) == 1                   # staged for pick N
    assert s.last_options[0]["store_id"] == "giant_eagle_squirrel_hill"
    assert s.last_options[0]["url"]                   # URL carried through

    # Pick N must resolve to the same SKU without re-running search
    # (this is the end-to-end guarantee we're buying with this change).
    reply2 = chat(s, "pick 1")
    assert s.state == "EXECUTE"
    assert "Catelli Ground Lamb" in reply2
    assert s.shopping_plan["total_cost"] == 9.99


# ---------- "pick N" side-flow -------------------------------------------

from agent.agent import detect_pick_intent, handle_pick_request


def _session_with_options() -> ShoppingSession:
    s = ShoppingSession()
    s.state = "CLARIFY"
    s.raw_items = [{"name": "lamb", "quantity": None, "unit": None, "ambiguous": False}]
    s.last_options = [
        {"item_name": "Lamb Weston Fries", "item_price": 5.99,
         "store_id": "giant_eagle_squirrel_hill", "store_display": "Giant Eagle", "url": None},
        {"item_name": "Lamb Weston Waffle Fries", "item_price": 5.99,
         "store_id": "giant_eagle_squirrel_hill", "store_display": "Giant Eagle", "url": None},
        {"item_name": "Catelli Ground Lamb", "item_price": 9.99,
         "store_id": "giant_eagle_squirrel_hill", "store_display": "Giant Eagle",
         "url": "https://example.com/ground-lamb"},
    ]
    return s


@pytest.mark.parametrize("msg,expected", [
    ("3", 3),
    ("#3", 3),
    ("pick 3", 3),
    ("pick #3", 3),
    ("I'll take 2", 2),
    ("choose 1", 1),
    ("I prefer 3", 3),
    ("I want option 3", 3),
    ("option 3", 3),
    ("number 2", 2),
    ("go with 2", 2),
    ("no, I prefer lamb of option 3", 3),
    ("hmm, I think #2 is best", 2),
])
def test_detect_pick_intent_positive(msg, expected):
    s = _session_with_options()
    assert detect_pick_intent(msg, s) == expected


@pytest.mark.parametrize("msg", [
    "I need 3 bananas",        # 3 is a quantity, not a pick index
    "add 3 eggs",
    "yes",
    "no thanks",
    "4",                       # out of range (only 3 options staged)
    "pick 99",                 # out of range
    "list some options",       # question, not a pick
    "",
])
def test_detect_pick_intent_negative(msg):
    s = _session_with_options()
    # Allow one ambiguity: "3 bananas" shouldn't pick — our bare-number
    # regex anchors to ^\s*#?\s*(\d+)$ so trailing words cause a miss.
    assert detect_pick_intent(msg, s) is None


def test_detect_pick_intent_needs_staged_options():
    s = ShoppingSession()  # no last_options
    assert detect_pick_intent("pick 2", s) is None
    assert detect_pick_intent("3", s) is None


def test_handle_pick_builds_single_sku_plan(monkeypatch):
    s = _session_with_options()
    # Stub load_stores so we don't hit real config.
    monkeypatch.setattr(
        a, "load_stores",
        lambda: {"giant_eagle_squirrel_hill": {
            "name": "Giant Eagle (Squirrel Hill)",
            "address": "1900 Murray Ave, Pittsburgh, PA 15217",
        }},
    )
    # Stub plan_route to avoid maps dep.
    monkeypatch.setattr(a, "plan_route", lambda **kw: {"stops": kw["store_ids"]})

    reply = handle_pick_request(s, 3)

    assert s.state == "EXECUTE"
    assert s.shopping_plan is not None
    plan = s.shopping_plan["plan"]
    assert "giant_eagle_squirrel_hill" in plan
    only_item = plan["giant_eagle_squirrel_hill"][0]
    assert only_item["item"] == "Catelli Ground Lamb"
    assert only_item["price"] == 9.99
    assert only_item["source"] == "user_pick"
    assert s.shopping_plan["total_cost"] == 9.99
    # last_options is consumed so a later bare number doesn't re-pick.
    assert s.last_options == []
    # User-facing copy mentions the item + price + store.
    assert "Catelli Ground Lamb" in reply
    assert "9.99" in reply
    assert "Giant Eagle" in reply


def test_pick_via_chat_skips_optimize(patched_confirm, monkeypatch):
    """E2E through chat(): 'I prefer 3' while options are staged must NOT
    go through session_execute (which would re-run optimize_shopping_list
    and potentially drift back to 'Lamb Weston fries')."""
    s = _session_with_options()
    monkeypatch.setattr(
        a, "load_stores",
        lambda: {"giant_eagle_squirrel_hill": {
            "name": "Giant Eagle (Squirrel Hill)",
            "address": "1900 Murray Ave",
        }},
    )
    monkeypatch.setattr(a, "plan_route", lambda **kw: {"stops": kw["store_ids"]})

    reply = chat(s, "I prefer 3")

    assert patched_confirm["execute"] == 0         # no optimize pass
    assert patched_confirm["call_llm"] == 0        # no LLM summary either
    assert s.state == "EXECUTE"
    assert "Catelli Ground Lamb" in reply
    # And a follow-up "no thanks" cleanly closes the session via the
    # existing EXECUTE-state closer handling.
    reply2 = chat(s, "no thanks")
    assert s.state == "DONE"
    assert "happy shopping" in reply2.lower()
