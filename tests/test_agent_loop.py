"""Unit tests for agent/loop.py: JSON parsing, retry, emergency fallback."""
from __future__ import annotations

import json

import pytest

from agent.state import AgentState
from agent import loop as loop_mod
from agent import agent as ag_mod


@pytest.fixture
def scripted(monkeypatch):
    """Replace call_llm with a queue-based stub. Returns a helper that
    loads a per-test queue of strings."""
    queue: list[str] = []

    def fake_call_llm(prompt, *, model=None, temperature=None, system=None):
        if queue:
            return queue.pop(0)
        # Force a reply so the loop doesn't hang.
        return json.dumps({"tool": "reply", "args": {"text": "(empty queue)"}})

    monkeypatch.setattr(ag_mod, "call_llm", fake_call_llm)
    return queue


# ─────────────────── happy path ──────────────────────────────────

class TestReplyTerminator:
    def test_single_reply_ends_turn(self, scripted):
        scripted.append(json.dumps({"tool": "reply", "args": {"text": "hello"}}))
        state = AgentState()
        out = loop_mod.chat(state, "hi")
        assert out == "hello"
        # Legacy {"reply": ...} shape.
        scripted.append(json.dumps({"reply": "hey"}))
        out2 = loop_mod.chat(state, "again")
        assert out2 == "hey"

    def test_tool_then_reply(self, scripted):
        scripted.append(json.dumps({
            "tool": "add_items",
            "args": {"items": [{"name": "milk", "quantity": 1, "unit": "gallon", "ambiguous": False}]},
        }))
        scripted.append(json.dumps({"tool": "reply", "args": {"text": "got milk"}}))
        state = AgentState()
        out = loop_mod.chat(state, "I need milk")
        assert out == "got milk"
        assert len(state.raw_items) == 1
        assert state.raw_items[0]["name"] == "milk"


# ─────────────────── parsing robustness ──────────────────────────

class TestJsonExtraction:
    def test_strips_code_fences(self):
        out = loop_mod._extract_json("```json\n{\"tool\": \"reply\", \"args\": {\"text\": \"hi\"}}\n```")
        assert out == {"tool": "reply", "args": {"text": "hi"}}

    def test_handles_leading_trailing_prose(self):
        raw = "Sure, here is my answer:\n{\"tool\": \"reply\", \"args\": {\"text\": \"hi\"}}\nDone!"
        assert loop_mod._extract_json(raw) == {"tool": "reply", "args": {"text": "hi"}}

    def test_returns_none_on_garbage(self):
        assert loop_mod._extract_json("no json here at all") is None
        assert loop_mod._extract_json("") is None
        assert loop_mod._extract_json(None) is None  # type: ignore[arg-type]

    def test_rejects_non_object(self):
        assert loop_mod._extract_json("[1, 2, 3]") is None
        assert loop_mod._extract_json("\"just a string\"") is None


class TestRetry:
    def test_retries_on_malformed_then_recovers(self, scripted):
        scripted.append("not json at all")
        scripted.append("still not json")
        scripted.append(json.dumps({"tool": "reply", "args": {"text": "finally"}}))
        state = AgentState()
        out = loop_mod.chat(state, "hi", max_steps=5)
        assert out == "finally"

    def test_exceeds_parse_retries_triggers_emergency(self, monkeypatch):
        # Every response is malformed; emergency fallback should run.
        def bad_call(prompt, *, model=None, temperature=None, system=None):
            return "never parses"
        monkeypatch.setattr(ag_mod, "call_llm", bad_call)
        # Patch the emergency fallback to return a known string (avoid
        # re-calling call_llm inside _emergency_reply).
        monkeypatch.setattr(loop_mod, "_emergency_reply",
                            lambda s, obs, *, error: f"EMERG:{error}")
        state = AgentState()
        out = loop_mod.chat(state, "hi", max_steps=10)
        assert out.startswith("EMERG:")


# ─────────────────── unknown / bad tool calls ─────────────────────

class TestToolDispatch:
    def test_unknown_tool_observation_then_reply(self, scripted):
        scripted.append(json.dumps({"tool": "nope_not_real", "args": {}}))
        scripted.append(json.dumps({"tool": "reply", "args": {"text": "recovered"}}))
        state = AgentState()
        out = loop_mod.chat(state, "hi", max_steps=4)
        assert out == "recovered"

    def test_tool_exception_caught(self, scripted, monkeypatch):
        # Force add_items to raise a non-ToolError exception.
        from agent import tools as tools_mod
        def boom(state, args):
            raise RuntimeError("kaboom")
        monkeypatch.setitem(tools_mod.TOOLS["add_items"], "fn", boom)

        scripted.append(json.dumps({"tool": "add_items", "args": {"items": []}}))
        scripted.append(json.dumps({"tool": "reply", "args": {"text": "recovered"}}))
        state = AgentState()
        out = loop_mod.chat(state, "hi", max_steps=4)
        assert out == "recovered"

    def test_args_must_be_object(self, scripted):
        scripted.append(json.dumps({"tool": "add_items", "args": "not an object"}))
        scripted.append(json.dumps({"tool": "reply", "args": {"text": "ok"}}))
        state = AgentState()
        out = loop_mod.chat(state, "hi", max_steps=4)
        assert out == "ok"


# ─────────────────── step cap ─────────────────────────────────────

class TestMaxSteps:
    def test_no_reply_within_limit_uses_emergency(self, scripted, monkeypatch):
        # Queue a tool that never emits a reply.
        for _ in range(10):
            scripted.append(json.dumps({"tool": "add_items", "args": {"items": [{"name": "x"}]}}))
        monkeypatch.setattr(loop_mod, "_emergency_reply",
                            lambda s, obs, *, error: f"EMERG:{error}")
        state = AgentState()
        out = loop_mod.chat(state, "hi", max_steps=3)
        assert out.startswith("EMERG:")


# ─────────────────── trace capture ────────────────────────────────

class TestTrace:
    def test_trace_logs_tool_calls(self, scripted):
        scripted.append(json.dumps({
            "tool": "add_items",
            "args": {"items": [{"name": "eggs", "quantity": 1, "unit": "dozen", "ambiguous": False}]},
        }))
        scripted.append(json.dumps({"tool": "reply", "args": {"text": "done"}}))
        state = AgentState()
        trace: list = []
        loop_mod.chat(state, "hi", max_steps=4, trace=trace)
        assert len(trace) == 2
        assert trace[0].tool == "add_items"
        assert trace[1].tool == "reply"
        assert trace[1].obs == {"text": "done"}


# ─────────────────── conversation history ─────────────────────────

class TestHistory:
    def test_user_and_agent_messages_appended(self, scripted):
        scripted.append(json.dumps({"tool": "reply", "args": {"text": "hi back"}}))
        state = AgentState()
        loop_mod.chat(state, "hi there")
        roles = [m["role"] for m in state.conversation_history]
        assert "user" in roles
        assert "agent" in roles
        assert state.conversation_history[-1]["text"] == "hi back"
