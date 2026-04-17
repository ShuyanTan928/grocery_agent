# ============================================================
# tools/intent_classifier.py
# LLM fallback for the agent's intent router.
#
# The agent's chat() first tries cheap deterministic rules
# (regex + state-machine + keyword sets). If USE_LLM_INTENT_ROUTER is
# on and none of those fire, this module classifies the user's message
# into one of a fixed label set so the caller can still route correctly.
#
# Design notes:
#   - Returns a STRUCTURED dict — callers don't have to parse free text.
#   - On any failure (empty LLM reply, bad JSON, unknown label) we
#     return {"label": "passthrough", ...} so the caller can fall back
#     to the existing state-machine default. This module is strictly
#     additive: never worse than regex-only.
#   - Session context (state + whether items/plan exist) is included
#     in the prompt so the model can distinguish e.g. "yes" from "yes,
#     and add milk".
# ============================================================

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import ShoppingSession


# The fixed label set the classifier is allowed to return. Matches the
# routing branches in agent.chat().
ALLOWED_LABELS: tuple[str, ...] = (
    "list_options",   # "any alternatives?" / "show me more"
    "recommend",      # "what's the best X" / "cheapest X"
    "closer",         # "no thanks" / "bye" / "we're done"
    "refinement",     # modify current list (items/qty/brand/prefs)
    "new_list",       # start a fresh shopping request
    "confirm_yes",    # "yes go ahead" / "looks good"
    "confirm_no",     # pure "no" with no content
    "passthrough",    # don't know — let the default handler run
)


_PROMPT_TEMPLATE = """You are an INTENT CLASSIFIER for a grocery shopping agent.
Map the user's latest message to EXACTLY ONE label from this list:

  list_options  - asking to see product options/alternatives for the current item
                  (e.g. "show me more", "what else do you have", "any alternatives")
  recommend     - asking for a ranked recommendation for a product
                  (e.g. "best milk", "what's the cheapest pasta", "recommend chips")
  closer        - ending the conversation / declining further action
                  (e.g. "no thanks", "we're done", "bye", "that's all")
  refinement    - modifying the current in-progress shopping list
                  (e.g. "also add eggs", "make it organic", "2 lbs instead",
                   "no, find the heinz one")
  new_list      - starting a brand-new shopping request unrelated to any prior plan
                  (e.g. "I need milk, eggs, and bread")
  confirm_yes   - approving the proposed plan without changes
                  (e.g. "yes", "go ahead", "sounds good")
  confirm_no    - rejecting the plan with NO additional instructions
                  (e.g. "no", "cancel")
  passthrough   - none of the above, or too ambiguous to tell

Current session context:
  state: {state}
  has_items_in_progress: {has_items}
  has_executed_plan: {has_plan}
  items_in_progress: {items}

User's latest message:
  "{message}"

Return STRICT JSON (no prose, no code fences) in this shape:
{{"label": "<one of the labels above>",
  "query": "<if label is recommend or list_options, the item noun phrase the user asked about; else empty string>",
  "confidence": <0.0-1.0>}}
"""


def _items_preview(session: "ShoppingSession") -> str:
    names = [it.get("name") for it in (session.raw_items or []) if it.get("name")]
    if not names:
        return "[]"
    return json.dumps(names[:5])


def classify_intent(message: str, session: "ShoppingSession") -> dict:
    """Classify `message` into one of ALLOWED_LABELS, using session
    context. Always returns a dict with keys: label, query, confidence,
    raw. On failure falls back to {"label": "passthrough"}."""
    if not message or not message.strip():
        return {"label": "passthrough", "query": "", "confidence": 0.0, "raw": ""}

    # Imported lazily to avoid a circular import with agent.agent.
    from agent.agent import call_llm
    from config.settings import LLM_ROUTER_MODEL, LLM_ROUTER_TEMPERATURE

    prompt = _PROMPT_TEMPLATE.format(
        state=session.state,
        has_items=bool(session.raw_items),
        has_plan=bool(session.shopping_plan and session.shopping_plan.get("plan")),
        items=_items_preview(session),
        message=message.replace('"', "'"),
    )

    # Prefer a dedicated small/fast model for routing; fall back to the
    # main model when LLM_ROUTER_MODEL is unset (empty string).
    router_model = LLM_ROUTER_MODEL or None

    try:
        raw = call_llm(
            prompt,
            model=router_model,
            temperature=LLM_ROUTER_TEMPERATURE,
        )
    except Exception:
        return {"label": "passthrough", "query": "", "confidence": 0.0, "raw": ""}

    return parse_classifier_reply(raw)


def parse_classifier_reply(raw: str) -> dict:
    """Parse the LLM reply into a structured dict. Robust to code-fence
    wrapping, leading prose, or malformed output."""
    fallback = {"label": "passthrough", "query": "", "confidence": 0.0, "raw": raw or ""}
    if not raw:
        return fallback

    stripped = re.sub(r"```(?:json)?|```", "", raw).strip()
    # Grab the first {...} block in case the model wrote a preamble.
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return fallback

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return fallback

    label = str(data.get("label", "")).strip().lower()
    if label not in ALLOWED_LABELS:
        return fallback

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "label": label,
        "query": str(data.get("query") or "").strip(),
        "confidence": max(0.0, min(1.0, confidence)),
        "raw": raw,
    }
