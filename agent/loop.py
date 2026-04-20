"""
LLM-driven agent loop.

Each call to `chat(state, user_message)` runs a tight ReAct-style loop:

    1. Render the orchestrator prompt (state + history + observations).
    2. Ask the LLM for exactly one JSON object.
    3. If it's {"tool": "reply", "args": {"text": ...}}, terminate.
    4. If it's another tool call, execute it, append the observation,
       and loop.
    5. If the JSON is malformed, retry up to MAX_PARSE_RETRIES times
       with a schema-reminder nudge.
    6. If we hit MAX_AGENT_STEPS without a reply, fall back to an
       emergency reply synthesized from the observations we have.

The loop is provider-agnostic: it just calls `call_llm()` from
`agent.agent`, which already handles Google GenAI vs OpenRouter.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from agent.prompts import render_loop_prompt
from agent.state import AgentState
from agent.tools import ReplySignal, run_tool

log = logging.getLogger(__name__)


# ────────────────────────── tunables ─────────────────────────────────


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


MAX_AGENT_STEPS = _int_env("MAX_AGENT_STEPS", 10)
MAX_PARSE_RETRIES = _int_env("MAX_AGENT_PARSE_RETRIES", 2)

# Loop temperature: a touch above 0 so the LLM doesn't lock into
# degenerate loops on retries, but low enough that tool choices are
# stable across identical inputs.
LOOP_TEMPERATURE = float(os.getenv("AGENT_LOOP_TEMPERATURE", "0.2") or "0.2")


@dataclass
class ToolTraceEntry:
    step: int
    tool: str
    args: dict
    obs: dict


# ────────────────────────── JSON extraction ──────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.I)


def _extract_json(raw: str) -> dict | None:
    """Best-effort JSON object extractor. Handles:
      - code fences (```json ... ```),
      - leading/trailing prose,
      - single-quote-ish typos the LLM sometimes emits.
    Returns None when nothing parseable is found."""
    if not raw:
        return None
    stripped = _FENCE_RE.sub("", raw).strip()

    # 1) direct parse
    try:
        val = json.loads(stripped)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        pass

    # 2) greedy balanced-brace slice from first "{" to last "}"
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last <= first:
        return None
    candidate = stripped[first : last + 1]
    try:
        val = json.loads(candidate)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        return None


# ────────────────────────── main entrypoint ──────────────────────────


def chat(
    state: AgentState,
    user_message: str,
    *,
    max_steps: int | None = None,
    trace: list[ToolTraceEntry] | None = None,
) -> str:
    """Process one user message. Returns the reply text.

    Parameters:
        state        Session state. Mutated in place.
        user_message The raw user input.
        max_steps    Override MAX_AGENT_STEPS per call (tests use this).
        trace        If provided, a list to append ToolTraceEntry objects
                     to — helpful for the REPL's --show-trace mode.

    Implementation notes:
      - Only the `reply` tool terminates normally. Everything else
        produces an observation that gets fed back to the LLM.
      - We cap the number of LLM calls to `max_steps * (MAX_PARSE_RETRIES
        + 1)` worst case (every step could be one parse failure + one
        successful parse). If the cap is exceeded, emergency reply.
    """
    from agent.agent import call_llm  # local import → avoid cycles

    state.add_message("user", user_message)

    limit = max_steps if max_steps is not None else MAX_AGENT_STEPS
    observations: list[dict] = []
    parse_fails = 0

    for step in range(limit):
        prompt = render_loop_prompt(state, user_message, observations)

        try:
            raw = call_llm(prompt, temperature=LOOP_TEMPERATURE)
        except Exception as e:
            log.warning("loop: LLM call failed on step %d: %s", step, e)
            reply = _emergency_reply(state, observations, error=str(e))
            state.add_message("agent", reply)
            return reply

        parsed = _extract_json(raw)

        if parsed is None:
            parse_fails += 1
            observations.append({
                "system": (
                    "Your last output was not valid JSON. Reply with EXACTLY "
                    "one JSON object: either "
                    "{\"tool\": \"<name>\", \"args\": {...}} or "
                    "{\"tool\": \"reply\", \"args\": {\"text\": \"...\"}}. "
                    "No prose, no markdown fences."
                ),
                "raw_excerpt": (raw or "")[:300],
            })
            if parse_fails > MAX_PARSE_RETRIES:
                log.warning("loop: exceeded %d parse retries", MAX_PARSE_RETRIES)
                reply = _emergency_reply(state, observations, error="malformed JSON from model")
                state.add_message("agent", reply)
                return reply
            continue

        tool_name = parsed.get("tool")
        args = parsed.get("args") or {}

        # A few models sometimes emit the top-level {"reply": "..."}
        # shape instead of {"tool": "reply", "args": {"text": ...}}.
        # Treat as a reply for robustness.
        if tool_name is None and "reply" in parsed and isinstance(parsed["reply"], str):
            text = parsed["reply"].strip()
            state.add_message("agent", text)
            return text

        if not isinstance(tool_name, str) or not tool_name:
            observations.append({"system": "Missing 'tool' field. Emit a valid JSON object."})
            continue

        if not isinstance(args, dict):
            observations.append({
                "tool": tool_name,
                "error": "'args' must be a JSON object",
            })
            continue

        try:
            obs = run_tool(state, tool_name, args)
        except ReplySignal as signal:
            text = signal.text.strip()
            if trace is not None:
                trace.append(ToolTraceEntry(step=step, tool="reply", args=args, obs={"text": text}))
            state.add_message("agent", text)
            return text

        if trace is not None:
            trace.append(ToolTraceEntry(step=step, tool=tool_name, args=args, obs=obs))
        observations.append({"tool": tool_name, "args": args, "obs": obs})

    # Hit MAX_AGENT_STEPS without emitting a reply.
    reply = _emergency_reply(state, observations, error="max steps exceeded")
    state.add_message("agent", reply)
    return reply


# ────────────────────────── safety net ───────────────────────────────


def _emergency_reply(
    state: AgentState,
    observations: list[dict],
    *,
    error: str,
) -> str:
    """When the loop can't produce a clean reply (bad JSON / exceeded
    max_steps / LLM error), make ONE last attempt with a very constrained
    prompt asking the model to just summarize what happened. If that
    also fails, return a static sorry message — strictly better than
    returning an empty string."""
    try:
        from agent.agent import call_llm
        fallback_prompt = (
            "You were in the middle of handling a user's grocery request but "
            "hit a problem: "
            f"{error}.\n\n"
            "Here is what you observed so far (may be empty):\n"
            f"{json.dumps(observations[-8:], ensure_ascii=False, indent=2)}\n\n"
            "Write a short, friendly reply to the user that either summarizes "
            "what was accomplished or apologizes and asks them to try again. "
            "Plain text only — no JSON, no tool calls."
        )
        text = call_llm(fallback_prompt, temperature=0.2)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass
    # Truly last-ditch.
    return (
        "Sorry — I ran into a hiccup assembling that response. "
        "Could you rephrase what you'd like to do, or say \"reset\" to start fresh?"
    )
