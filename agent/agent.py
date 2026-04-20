# ============================================================
# agent/agent.py
# Thin facade: LLM client + call_llm + back-compat exports.
#
# The old state-machine + regex side-flow implementation was deleted
# in favor of the LLM tool-calling loop in agent/loop.py. All routing
# decisions are now made by the orchestrator LLM looking at
# agent.state.AgentState.to_llm_view() each turn.
#
# This file intentionally keeps:
#   - LLM clients (Google GenAI + OpenRouter) and call_llm()
#   - PARSE_PROMPT + parse_items_from_message() (used by the
#     `parse_items` tool fallback in agent/tools.py)
#   - Re-exports of chat and ShoppingSession for existing callers
#     (scripts/chat.py, main.py, tests)
# ============================================================

from __future__ import annotations

import json
import re

from config.settings import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_TITLE,
    OPENROUTER_BASE_URL,
    OPENROUTER_HTTP_REFERER,
)

_google_genai_client = None
_openrouter_client = None


# ────────────────────────── system prompt ────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant for a Pittsburgh grocery shopping agent.
Be friendly, concise, and practical. Never make up prices or products."""


# ────────────────────────── parse items prompt ──────────────────────
# Kept here because `agent.tools.tool_parse_items` still uses it as a
# fallback when the orchestrator can't structure items directly.

PARSE_PROMPT = """Extract ALL grocery items from the user's message.
For each item, identify:
- name: the item name (normalized to English)
- quantity: number of units if mentioned (null if not mentioned)
- unit: unit of measure if mentioned, e.g. "lb", "gallon", "dozen" (null if not mentioned)
- ambiguous: true if quantity/unit is unclear for a priced-by-weight item

Items that are priced by weight (need quantity): meat, fish, seafood, deli items, produce sold by lb.
Items where quantity doesn't matter much: packaged goods with fixed sizes (milk gallon, egg dozen, bread loaf).

Return ONLY valid JSON in this exact format, no explanation:
{
  "items": [
    {"name": "whole milk", "quantity": 1, "unit": "gallon", "ambiguous": false},
    {"name": "pork shoulder", "quantity": null, "unit": null, "ambiguous": true}
  ]
}

User message: """


# ────────────────────────── LLM clients ──────────────────────────────


def _get_google_genai_client():
    global _google_genai_client
    if _google_genai_client is None:
        from google import genai
        _google_genai_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _google_genai_client


def _get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI

        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to .env when USE_OPENROUTER=true."
            )
        headers = {}
        if OPENROUTER_HTTP_REFERER:
            headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
        if OPENROUTER_APP_TITLE:
            headers["X-Title"] = OPENROUTER_APP_TITLE
        _openrouter_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            default_headers=headers or None,
        )
    return _openrouter_client


def call_llm(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    system: str | None = None,
) -> str:
    """Single-turn LLM call (Google GenAI or OpenRouter).

    `model` / `temperature` override defaults per call (the loop uses
    this to set a low temp for tool choice). `system` overrides the
    default SYSTEM_PROMPT — mostly for the loop, which carries its own
    domain instructions in the user prompt already.
    """
    effective_model = model or LLM_MODEL
    effective_temp = 0.3 if temperature is None else float(temperature)
    effective_system = system if system is not None else SYSTEM_PROMPT

    if LLM_PROVIDER == "openrouter":
        client = _get_openrouter_client()
        resp = client.chat.completions.create(
            model=effective_model,
            temperature=effective_temp,
            messages=[
                {"role": "system", "content": effective_system},
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("OpenRouter returned an empty response.")
        return text

    # Default: Google GenAI
    from google import genai
    client = _get_google_genai_client()
    response = client.models.generate_content(
        model=effective_model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=effective_system,
            temperature=effective_temp,
        ),
    )
    return response.text.strip()


# ────────────────────────── legacy parse helper ──────────────────────


def parse_items_from_message(user_message: str) -> list[dict]:
    """Extract structured items from free-text. Used by the
    `parse_items` tool as a fallback when the orchestrator can't
    construct the item list itself."""
    prompt = PARSE_PROMPT + f'"{user_message}"'
    raw = call_llm(prompt)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(raw)
        items = data.get("items", [])
        return items if isinstance(items, list) else []
    except json.JSONDecodeError:
        return [
            {"name": t.strip(), "quantity": None, "unit": None, "ambiguous": False}
            for t in user_message.split(",")
            if t.strip()
        ]


# ────────────────────────── public surface ───────────────────────────

from agent.loop import chat  # noqa: E402
from agent.state import AgentState, ShoppingSession  # noqa: E402

__all__ = [
    "call_llm",
    "parse_items_from_message",
    "chat",
    "AgentState",
    "ShoppingSession",
    "SYSTEM_PROMPT",
    "PARSE_PROMPT",
]
