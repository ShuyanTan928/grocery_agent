"""Tests for the hybrid intent router: the USE_LLM_INTENT_ROUTER
on/off switch, the classifier reply parser, and end-to-end routing
behavior with the LLM stubbed out.

All tests mock the LLM call. No network traffic.
"""
from __future__ import annotations

import pytest

from agent import agent as a
from agent.agent import ShoppingSession, chat
from tools import intent_classifier as ic
from tools.intent_classifier import ALLOWED_LABELS, parse_classifier_reply


# ---------- parse_classifier_reply ----------------------------------------

def test_parse_reply_valid_json():
    out = parse_classifier_reply(
        '{"label": "list_options", "query": "avocado", "confidence": 0.9}'
    )
    assert out["label"] == "list_options"
    assert out["query"] == "avocado"
    assert out["confidence"] == pytest.approx(0.9)


def test_parse_reply_strips_code_fence():
    raw = "```json\n{\"label\": \"recommend\", \"query\": \"milk\", \"confidence\": 0.8}\n```"
    assert parse_classifier_reply(raw)["label"] == "recommend"


def test_parse_reply_tolerates_preamble():
    raw = "Sure! Here you go:\n{\"label\": \"closer\", \"query\": \"\", \"confidence\": 0.95}"
    assert parse_classifier_reply(raw)["label"] == "closer"


def test_parse_reply_unknown_label_falls_back():
    out = parse_classifier_reply('{"label": "dance_party", "confidence": 1.0}')
    assert out["label"] == "passthrough"


def test_parse_reply_bad_json_falls_back():
    assert parse_classifier_reply("not json at all")["label"] == "passthrough"
    assert parse_classifier_reply("")["label"] == "passthrough"


def test_parse_reply_clamps_confidence():
    hi = parse_classifier_reply('{"label": "closer", "confidence": 7.0}')
    lo = parse_classifier_reply('{"label": "closer", "confidence": -3}')
    assert hi["confidence"] == 1.0
    assert lo["confidence"] == 0.0


def test_all_allowed_labels_round_trip():
    for lbl in ALLOWED_LABELS:
        if lbl == "passthrough":
            continue
        out = parse_classifier_reply(f'{{"label": "{lbl}", "confidence": 0.9}}')
        assert out["label"] == lbl


# ---------- small-model override ------------------------------------------

def test_classifier_uses_router_model_when_configured(monkeypatch):
    """When LLM_ROUTER_MODEL is set, classify_intent must thread the
    small-model override and temperature=0 into call_llm."""
    import config.settings as settings
    from tools.intent_classifier import classify_intent

    monkeypatch.setattr(settings, "LLM_ROUTER_MODEL", "google/gemma-3-1b-it")
    monkeypatch.setattr(settings, "LLM_ROUTER_TEMPERATURE", 0.0)

    captured: dict = {}

    def fake_call_llm(prompt, *, model=None, temperature=None):
        captured["model"] = model
        captured["temperature"] = temperature
        return '{"label": "closer", "confidence": 0.9}'

    monkeypatch.setattr(a, "call_llm", fake_call_llm)

    s = ShoppingSession()
    s.state = "CONFIRM"
    out = classify_intent("that's enough for now", s)

    assert captured["model"] == "google/gemma-3-1b-it"
    assert captured["temperature"] == 0.0
    assert out["label"] == "closer"


def test_classifier_falls_back_to_main_model_when_unset(monkeypatch):
    """Empty LLM_ROUTER_MODEL → model=None → call_llm uses LLM_MODEL."""
    import config.settings as settings
    from tools.intent_classifier import classify_intent

    monkeypatch.setattr(settings, "LLM_ROUTER_MODEL", "")
    monkeypatch.setattr(settings, "LLM_ROUTER_TEMPERATURE", 0.0)

    captured: dict = {}

    def fake_call_llm(prompt, *, model=None, temperature=None):
        captured["model"] = model
        return '{"label": "passthrough", "confidence": 0.9}'

    monkeypatch.setattr(a, "call_llm", fake_call_llm)

    classify_intent("hmm", ShoppingSession())
    assert captured["model"] is None        # caller passed None → main model


# ---------- hybrid routing in chat() --------------------------------------

@pytest.fixture
def stub_handlers(monkeypatch):
    """Replace the LLM-hitting agent helpers with stubs so we can
    observe routing without making network calls."""
    calls: dict[str, int] = {
        "call_llm": 0,
        "classify_intent": 0,
        "execute": 0,
        "handle_list_options": 0,
        "handle_recommend": 0,
        "update_qty": 0,
        "extract_prefs": 0,
    }

    def fake_call_llm(prompt):
        calls["call_llm"] += 1
        return "(stubbed llm reply)"

    def fake_execute(session):
        calls["execute"] += 1
        return "[EXECUTED]"

    def fake_list_options(session, msg):
        calls["handle_list_options"] += 1
        return "[LIST_OPTIONS]"

    def fake_recommend(intent):
        calls["handle_recommend"] += 1
        return f"[RECOMMEND:{intent.get('query','')}]"

    def fake_update_qty(msg, items):
        calls["update_qty"] += 1
        return items

    def fake_extract_prefs(msg, avoid, prefer):
        calls["extract_prefs"] += 1
        return avoid, prefer

    monkeypatch.setattr(a, "call_llm", fake_call_llm)
    monkeypatch.setattr(a, "session_execute", fake_execute)
    monkeypatch.setattr(a, "handle_list_options_request", fake_list_options)
    monkeypatch.setattr(a, "handle_recommend_request", fake_recommend)
    monkeypatch.setattr(a, "update_quantities_from_reply", fake_update_qty)
    monkeypatch.setattr(a, "extract_preferences_from_reply", fake_extract_prefs)
    return calls


def _patch_switch(monkeypatch, on: bool):
    """Patch the flag where agent.py reads it (inline import)."""
    import config.settings as settings
    monkeypatch.setattr(settings, "USE_LLM_INTENT_ROUTER", on)


def _patch_classifier(monkeypatch, label: str, query: str = "", conf: float = 0.9, target: str = ""):
    def fake_classify(message, session):
        return {
            "label": label, "query": query, "target": target,
            "confidence": conf, "raw": "",
        }
    monkeypatch.setattr(ic, "classify_intent", fake_classify)


def test_switch_off_skips_classifier(stub_handlers, monkeypatch):
    _patch_switch(monkeypatch, False)

    # Sentinel: if classifier is accidentally called, blow up loudly.
    def boom(*args, **kwargs):
        raise AssertionError("classifier should NOT run when switch is off")
    monkeypatch.setattr(ic, "classify_intent", boom)

    s = ShoppingSession()
    s.state = "CONFIRM"
    s.raw_items = [{"name": "milk", "quantity": 1, "unit": None, "ambiguous": False}]
    # "hmm" is ambiguous: no regex match, not a yes/no/cancel word, not long
    # enough to be a refinement. Without the switch this goes through the
    # "ambiguous short reply" LLM path (one CONFIRM re-render).
    chat(s, "hmm")
    assert stub_handlers["call_llm"] == 1


def test_switch_on_routes_confirm_yes(stub_handlers, monkeypatch):
    _patch_switch(monkeypatch, True)
    _patch_classifier(monkeypatch, "confirm_yes", conf=0.95)
    s = ShoppingSession()
    s.state = "CONFIRM"
    s.raw_items = [{"name": "milk", "quantity": 1, "unit": None, "ambiguous": False}]
    # "hmm" isn't a yes-word in the regex path, so confirmation would never
    # happen there. With the switch on, the LLM label takes over.
    reply = chat(s, "hmm")
    assert reply == "[EXECUTED]"
    assert stub_handlers["execute"] == 1
    assert s.state == "EXECUTE"


def test_switch_on_routes_closer(stub_handlers, monkeypatch):
    _patch_switch(monkeypatch, True)
    _patch_classifier(monkeypatch, "closer", conf=0.9)
    s = ShoppingSession()
    s.state = "EXECUTE"
    s.shopping_plan = {"plan": {"x": [{"item": "x", "price": 1}]}, "total_cost": 1.0}
    reply = chat(s, "actually we're all set for today")
    assert "happy shopping" in reply.lower()
    assert s.state == "DONE"
    assert stub_handlers["execute"] == 0


def test_switch_on_routes_list_options(stub_handlers, monkeypatch):
    _patch_switch(monkeypatch, True)
    _patch_classifier(monkeypatch, "list_options", query="avocado", conf=0.85)
    s = ShoppingSession()
    s.state = "CLARIFY"
    s.raw_items = [{"name": "avocado", "quantity": None, "unit": None, "ambiguous": False}]
    reply = chat(s, "其他牌子呢")                     # LLM handles the long tail
    assert reply == "[LIST_OPTIONS]"
    assert stub_handlers["handle_list_options"] == 1
    assert s.state == "CLARIFY"                       # did NOT advance


def test_switch_on_routes_recommend(stub_handlers, monkeypatch):
    _patch_switch(monkeypatch, True)
    _patch_classifier(monkeypatch, "recommend", query="ground beef", conf=0.9)
    s = ShoppingSession()
    # Phrasing the recommend regex does NOT cover ("what X should I get"
    # vs the supported "which X should I get") — regex miss, LLM hit.
    reply = chat(s, "what beef should I get for dinner")
    assert reply == "[RECOMMEND:ground beef]"
    assert stub_handlers["handle_recommend"] == 1


def test_switch_on_low_confidence_falls_through(stub_handlers, monkeypatch):
    _patch_switch(monkeypatch, True)
    _patch_classifier(monkeypatch, "list_options", query="", conf=0.2)
    s = ShoppingSession()
    s.state = "CONFIRM"
    s.raw_items = [{"name": "milk", "quantity": 1, "unit": None, "ambiguous": False}]
    # Low confidence → classifier bails → state machine's own "ambiguous
    # short reply" branch runs, which calls the LLM for the CONFIRM
    # prompt. We observe that rather than the list-options handler.
    chat(s, "hmm")
    assert stub_handlers["handle_list_options"] == 0
    assert stub_handlers["call_llm"] == 1


def test_switch_on_refinement_falls_through_to_state_machine(stub_handlers, monkeypatch):
    _patch_switch(monkeypatch, True)
    _patch_classifier(monkeypatch, "refinement", query="", conf=0.9)
    s = ShoppingSession()
    s.state = "CONFIRM"
    s.raw_items = [{"name": "milk", "quantity": 1, "unit": None, "ambiguous": False}]
    # For refinement, _llm_route returns None and the existing CONFIRM
    # branch takes over (auto-execute path for multi-word replies).
    reply = chat(s, "actually 2 gallons of whole milk")
    assert reply == "[EXECUTED]"
    assert stub_handlers["execute"] == 1
    assert stub_handlers["update_qty"] == 1


# ---------- deterministic regex side-flows (no LLM) -----------------------

class TestRemoveIntentRegex:
    """The deterministic regex must catch all phrasings from the live
    transcript without needing the LLM router, since small routing models
    (e.g. gemma-3-1b) can't reliably label this intent."""

    def _make_session(self):
        s = ShoppingSession()
        s.raw_items = [
            {"name": "orange juice", "quantity": 240, "unit": "ml", "ambiguous": False},
            {"name": "sparkling water", "quantity": 1, "unit": None, "ambiguous": False},
        ]
        s.shopping_plan = {"plan": {"aldi": [{"item": "Sparkling Water", "price": 0.65}]}}
        return s

    @pytest.mark.parametrize("msg, expected", [
        ("No i don't want the water", "water"),
        ("I don't need water", "water"),
        ("remove the sparkling water", "sparkling water"),
        ("drop the ginger", "ginger"),
        ("skip the bacon", "bacon"),
        ("please remove water", "water"),
        ("no sparkling water please", "sparkling water"),
        ("take out the ginger ale", "ginger ale"),
        ("get rid of the water", "water"),
        ("without the sugar", "sugar"),
        ("delete sparkling water from the list", "sparkling water"),
    ])
    def test_positive_cases(self, msg, expected):
        from agent.agent import detect_remove_intent
        assert detect_remove_intent(msg, self._make_session()) == expected

    @pytest.mark.parametrize("msg", [
        "no",                          # bare cancel
        "no thanks",                   # close-out, not remove
        "no thank you",
        "actually, remove that",       # target is "that" — too vague
        "yes remove something",        # starts wrong
        "",                            # empty
    ])
    def test_false_positives_rejected(self, msg):
        from agent.agent import detect_remove_intent
        assert detect_remove_intent(msg, self._make_session()) is None

    def test_requires_active_list_or_plan(self):
        """With nothing to remove from, the detector bails out so the
        request can be routed to a fresh-list flow instead."""
        from agent.agent import detect_remove_intent
        empty = ShoppingSession()
        assert detect_remove_intent("remove the water", empty) is None


class TestJustifyIntentRegex:
    def _make_session(self):
        s = ShoppingSession()
        s.raw_items = [{"name": "ginger", "quantity": 1, "unit": "piece", "ambiguous": False}]
        s.shopping_plan = {"plan": {"ga": [{"item": "Ginger Ale", "price": 0.89}]}}
        return s

    @pytest.mark.parametrize("msg, expected_token", [
        ("why did you provide me sparkling water?", "sparkling water"),
        ("why did you pick the sparkling water", "sparkling water"),
        ("why is ginger ale in there?", "ginger ale"),
        ("why is the orange juice on the list", "orange juice"),
    ])
    def test_positive_cases(self, msg, expected_token):
        from agent.agent import detect_justify_intent
        assert detect_justify_intent(msg, self._make_session()) == expected_token

    @pytest.mark.parametrize("msg", [
        "how are you?",
        "why not add more?",
        "what is in the plan?",        # not a "why" question
    ])
    def test_false_positives_rejected(self, msg):
        from agent.agent import detect_justify_intent
        assert detect_justify_intent(msg, self._make_session()) is None

    def test_detector_fires_without_plan_handler_gates(self):
        """The detector is permissive — it fires on the phrasing alone.
        The HANDLER is responsible for responding politely when there's
        no plan yet. This beats silently falling through to the state
        machine's refinement branch (which would mis-route the question
        as a change request)."""
        from agent.agent import detect_justify_intent, handle_justify_request
        s = ShoppingSession()
        s.raw_items = [{"name": "milk"}]
        # No shopping_plan yet.
        assert detect_justify_intent("why did you pick milk", s) == "milk"
        reply = handle_justify_request(s, "milk")
        assert "no active plan" in reply.lower()


def test_regex_remove_runs_before_llm_router(stub_handlers, monkeypatch):
    """chat() priority: deterministic remove regex beats the LLM router.
    Even if the classifier wanted to return a different (wrong) label,
    we never get there."""
    _patch_switch(monkeypatch, True)

    def should_not_run(*args, **kwargs):
        raise AssertionError("LLM classifier should not fire when regex matches")
    monkeypatch.setattr(ic, "classify_intent", should_not_run)

    s = ShoppingSession()
    s.state = "EXECUTE"
    s.raw_items = [
        {"name": "sparkling water", "quantity": 1, "unit": None, "ambiguous": False},
    ]
    s.shopping_plan = {
        "plan": {"aldi": [{"item": "PurAqua Sparkling Water", "price": 0.65,
                           "store_display": "aldi", "url": None}]},
        "total_cost": 0.65, "store_ids": ["aldi"], "stores_meta": {"aldi": {}},
    }

    reply = chat(s, "No i don't want the water")
    assert "Sparkling Water" in reply
    assert s.raw_items == []
    # Auto-rescue: list fully emptied → session transitioned back to
    # CLARIFY with a cleared plan, so the NEXT user message is parsed as
    # a new list instead of a refinement on nothing.
    assert s.shopping_plan is None
    assert s.state == "CLARIFY"
    assert "empty" in reply.lower()


# ---------- remove_item / justify side-flows ------------------------------

def test_remove_item_prunes_plan_and_items(stub_handlers, monkeypatch):
    """LLM router → remove_item handler should drop matching SKUs from the
    plan, matching ingredients from raw_items, and NOT reset the session."""
    _patch_switch(monkeypatch, True)
    _patch_classifier(monkeypatch, "remove_item", target="water", conf=0.9)

    s = ShoppingSession()
    s.state = "EXECUTE"
    s.raw_items = [
        {"name": "orange juice", "quantity": 240, "unit": "ml", "ambiguous": False},
        {"name": "sparkling water", "quantity": 1, "unit": None, "ambiguous": False},
    ]
    s.shopping_plan = {
        "plan": {
            "giant_eagle": [
                {"item": "Simply Pulp Free Orange Juice", "price": 2.59,
                 "store_display": "giant eagle", "url": None},
            ],
            "aldi": [
                {"item": "PurAqua Orange Mango Sparkling Water", "price": 0.65,
                 "store_display": "aldi", "url": None},
            ],
        },
        "total_cost": 3.24,
        "not_found": [],
        "store_ids": ["giant_eagle", "aldi"],
        "stores_meta": {"giant_eagle": {}, "aldi": {}},
    }

    reply = chat(s, "i don't need water")
    # SKU whose name contains "water" is dropped; orange juice stays.
    assert "Sparkling Water" in reply
    assert s.state == "EXECUTE"                 # state NOT reset
    assert len(s.raw_items) == 1
    assert s.raw_items[0]["name"] == "orange juice"
    plan = s.shopping_plan["plan"]
    assert "aldi" not in plan                   # empty store pruned
    assert "giant_eagle" in plan
    assert s.shopping_plan["total_cost"] == pytest.approx(2.59)
    # Re-optimize MUST NOT run — we're just pruning.
    assert stub_handlers["execute"] == 0


def test_remove_item_no_match_informs_user(stub_handlers, monkeypatch):
    _patch_switch(monkeypatch, True)
    _patch_classifier(monkeypatch, "remove_item", target="caviar", conf=0.9)

    s = ShoppingSession()
    s.state = "EXECUTE"
    s.raw_items = [{"name": "milk", "quantity": 1, "unit": None, "ambiguous": False}]
    s.shopping_plan = {
        "plan": {"aldi": [{"item": "Whole Milk 1 Gallon", "price": 3.19,
                           "store_display": "aldi", "url": None}]},
        "total_cost": 3.19, "store_ids": ["aldi"], "stores_meta": {"aldi": {}},
    }

    reply = chat(s, "drop the caviar")
    assert "couldn't find" in reply.lower() or "couldn" in reply.lower()
    assert len(s.raw_items) == 1                # untouched
    assert s.shopping_plan["total_cost"] == pytest.approx(3.19)


def test_justify_explains_plan_entry(stub_handlers, monkeypatch):
    _patch_switch(monkeypatch, True)
    _patch_classifier(monkeypatch, "justify", target="ginger ale", conf=0.9)

    s = ShoppingSession()
    s.state = "EXECUTE"
    s.raw_items = [{"name": "ginger", "quantity": 1, "unit": "piece", "ambiguous": False}]
    s.shopping_plan = {
        "plan": {
            "giant_eagle": [
                {"item": "Faygo Ginger Ale, Diet (20 oz)", "price": 0.89,
                 "store_display": "giant eagle", "url": "https://example.com/ga"},
            ],
        },
        "total_cost": 0.89, "store_ids": ["giant_eagle"], "stores_meta": {"giant_eagle": {}},
    }

    reply = chat(s, "why did you provide me ginger ale?")
    assert "Faygo Ginger Ale" in reply
    assert "$0.89" in reply
    assert s.state == "EXECUTE"                 # purely informational
    assert len(s.raw_items) == 1                # unchanged
    assert s.shopping_plan["total_cost"] == pytest.approx(0.89)


def test_justify_no_active_plan(stub_handlers, monkeypatch):
    _patch_switch(monkeypatch, True)
    _patch_classifier(monkeypatch, "justify", target="ginger", conf=0.9)

    s = ShoppingSession()
    s.state = "CLARIFY"
    # No shopping_plan yet.
    reply = chat(s, "why is ginger there")
    assert "no active plan" in reply.lower()
    assert s.state == "CLARIFY"


def test_post_plan_lowers_confidence_threshold(stub_handlers, monkeypatch):
    """A 0.4-confidence remove_item must fire in EXECUTE state (post-plan
    threshold = 0.35) but would bail at default 0.55 pre-plan."""
    _patch_switch(monkeypatch, True)
    _patch_classifier(monkeypatch, "remove_item", target="milk", conf=0.4)

    s = ShoppingSession()
    s.state = "EXECUTE"
    s.raw_items = [{"name": "milk", "quantity": 1, "unit": None, "ambiguous": False}]
    s.shopping_plan = {
        "plan": {"aldi": [{"item": "Whole Milk 1 Gallon", "price": 3.19,
                           "store_display": "aldi", "url": None}]},
        "total_cost": 3.19, "store_ids": ["aldi"], "stores_meta": {"aldi": {}},
    }

    reply = chat(s, "ugh skip the milk")
    assert "Whole Milk" in reply                # route DID fire
    assert s.raw_items == []
    # Auto-rescue: list emptied → shopping_plan cleared, state reset to CLARIFY.
    assert s.shopping_plan is None
    assert s.state == "CLARIFY"


def test_pre_plan_still_requires_higher_confidence(stub_handlers, monkeypatch):
    """Same 0.4 confidence but no plan → hits the tighter 0.55 pre-plan
    threshold → _llm_route returns None → default CLARIFY LLM path runs
    (so the remove handler must NOT fire)."""
    _patch_switch(monkeypatch, True)
    _patch_classifier(monkeypatch, "remove_item", target="milk", conf=0.4)

    s = ShoppingSession()
    s.state = "CLARIFY"                         # no plan → default threshold
    s.raw_items = [{"name": "milk", "quantity": 1, "unit": None, "ambiguous": False}]
    chat(s, "ugh skip the milk")
    # Remove handler would have emptied raw_items if it had fired.
    assert len(s.raw_items) == 1
    # And the state-machine default LLM path ran instead.
    assert stub_handlers["call_llm"] >= 1
