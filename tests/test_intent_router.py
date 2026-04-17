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


def _patch_classifier(monkeypatch, label: str, query: str = "", conf: float = 0.9):
    def fake_classify(message, session):
        return {"label": label, "query": query, "confidence": conf, "raw": ""}
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
