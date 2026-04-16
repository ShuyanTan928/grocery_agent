# ============================================================
# agent/agent.py
# ReAct grocery agent: LLM via Google GenAI SDK or OpenRouter (OpenAI-compatible API).
#
# Conversation flow:
#   Turn 1 - User states their shopping needs (any language)
#          - Agent parses items, detects ambiguities
#          - Agent asks clarifying questions (quantities, preferences)
#   Turn 2 - User answers clarifications
#          - Agent confirms the final plan before executing
#   Turn 3 - User confirms → Agent runs optimizer + route planner
#          - Agent presents the full shopping plan
#
# ReAct states:
#   CLARIFY   → need more info before planning
#   CONFIRM   → plan ready, waiting for user approval
#   EXECUTE   → run tools and return results
#   DONE      → conversation complete
# ============================================================

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
from tools.price_optimizer import optimize_shopping_list, load_stores, find_at_store
from tools.synonyms import expand_query, matches_any
from tools.route_planner import plan_route
from tools.errand_runner import generate_errand_quote

_google_genai_client = None
_openrouter_client = None

# --------------- Prompts ---------------

SYSTEM_PROMPT = """You are a smart grocery shopping assistant for Pittsburgh, PA.
You help users find the cheapest prices across local stores (Giant Eagle, Aldi, Walmart, Trader Joe's, Whole Foods)
and plan the most efficient driving route.

Available stores:
- Giant Eagle (Squirrel Hill) — large selection, mid-range prices
- Aldi (Greenfield) — cheapest for most staples
- Walmart (Crafton) — cheap but farther away
- Trader Joe's (Shadyside) — good for specialty items, expensive for meat
- Whole Foods (East Liberty) — premium/organic, most expensive

Be friendly, concise, and practical. Never make up prices."""


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


CLARIFY_PROMPT = """You are a grocery shopping assistant. The user gave you a shopping list.

Parsed items:
{items_json}

Store preferences so far:
  avoid  (never buy item at these stores): {avoid_json}
  prefer (buy item only at these stores): {prefer_json}

Your task: Ask the user ONE friendly message that covers ALL clarifications needed.
Ask about:
1. Quantities for any ambiguous items (items with ambiguous=true) — mention bulk discounts if relevant
2. Whether they have store preferences (e.g. avoid meat at Trader Joe's, or require pork from Trader Joe's)
   — only ask this if BOTH avoid and prefer are still empty

Keep it short. Max 3-4 lines. Ask everything in ONE message, not multiple.
User's original message: {user_message}"""


CONFIRM_PROMPT = """You are a grocery shopping assistant. Present a brief summary of the shopping plan
for the user to confirm before executing.

Items with quantities:
{items_json}

Store preferences:
  avoid  (never buy item at these stores): {avoid_json}
  prefer (buy item only at these stores): {prefer_json}

Write a short 3-5 line confirmation message asking the user to confirm.
List the items. If any preferences are set, state them plainly (e.g. "I'll get pork from Trader Joe's").
End with: "Shall I find the best prices and plan your route? (yes/no)"""


SUMMARY_PROMPT = """You are a grocery shopping assistant. Present the final shopping plan clearly.

Shopping plan (grouped by store):
{shopping_json}

Route plan:
{route_json}

{errand_section}

Write a friendly summary. Structure:
1. Per-store shopping list with prices
2. Driving route with times
3. Total cost
4. If shopping_plan has a non-empty "unfulfilled_preferences" list, briefly note which
   preferences we could not honor and why (e.g. the requested store didn't carry the item).
5. (If errand quote exists) Errand runner option

End by asking: "Would you like to adjust anything or start a new list?"""


# --------------- Session state ---------------

class ShoppingSession:
    """
    Holds all state for one multi-turn shopping conversation.
    Create a new instance per user session.
    """

    def __init__(self):
        self.state = "CLARIFY"          # current ReAct state
        self.raw_items = []             # parsed item dicts from LLM
        self.preferences = {}           # AVOID: {"chicken": ["trader_joes_shadyside"]}
        self.preferred_stores = {}      # PREFER: {"pork": ["trader_joes_shadyside"]}
        self.conversation_history = []  # list of {"role": "user"/"agent", "text": str}
        self.shopping_plan = None
        self.route_plan = None
        self.errand_quote = None
        self.clarification_done = False
        self.want_errand = False

    def add_message(self, role: str, text: str):
        self.conversation_history.append({"role": role, "text": text})


# --------------- LLM helpers ---------------

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


def call_llm(prompt: str) -> str:
    """Single-turn LLM call (Google GenAI or OpenRouter)."""
    if LLM_PROVIDER == "openrouter":
        client = _get_openrouter_client()
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
        model=LLM_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )
    return response.text.strip()


def parse_items_from_message(user_message: str) -> list[dict]:
    """
    Use LLM to extract structured item list from natural language.
    Returns list of dicts: {name, quantity, unit, ambiguous}
    """
    prompt = PARSE_PROMPT + f'"{user_message}"'
    raw = call_llm(prompt)
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        data = json.loads(raw)
        return data.get("items", [])
    except json.JSONDecodeError:
        return [
            {"name": t.strip(), "quantity": None, "unit": None, "ambiguous": False}
            for t in user_message.split(",")
        ]


def _merge_pref_dict(existing: dict, new: dict) -> dict:
    """Union of store lists per item, in-place merge semantics."""
    for item, stores in (new or {}).items():
        if not isinstance(stores, list):
            continue
        if item in existing:
            existing[item] = list(dict.fromkeys(existing[item] + stores))
        else:
            existing[item] = list(stores)
    return existing


def extract_preferences_from_reply(
    user_reply: str, existing_avoid: dict, existing_prefer: dict
) -> tuple[dict, dict]:
    """
    Use LLM to extract TWO kinds of store preferences from the user's message:
      - avoid: user does NOT want to buy item X at store Y
               e.g. "don't buy meat at Trader Joe's"
      - prefer: user REQUIRES item X to come from store Y
               e.g. "get pork from Trader Joe's", "bananas only at Aldi"

    Returns (avoid_dict, prefer_dict), each shaped like:
      {"pork": ["trader_joes_shadyside"], ...}
    Lists are merged with existing state.
    """
    store_map = {s["name"].lower(): s["id"] for s in load_stores().values()}
    store_list = ", ".join(f"{v} ({k})" for k, v in store_map.items())

    prompt = f"""Extract two kinds of store preferences from the user's message.

AVOID: items the user does NOT want at a particular store.
  Signals: "don't buy X at Y", "not at Y", "avoid Y for X", "skip Y's meat".
PREFER: items the user wants to buy ONLY at a particular store.
  Signals: "X from Y", "get X at Y", "buy X only at Y", "prefer Y for X".

Available store IDs: {store_list}

User message: "{user_reply}"

Return ONLY valid JSON in this exact shape (omit empty maps is NOT allowed — use {{}}):
{{
  "avoid":  {{"<item>": ["<store_id>", ...]}},
  "prefer": {{"<item>": ["<store_id>", ...]}}
}}

Examples:
  User: "don't buy chicken at Trader Joe's"
    → {{"avoid": {{"chicken": ["trader_joes_shadyside"]}}, "prefer": {{}}}}
  User: "I want pork from Trader Joe's"
    → {{"avoid": {{}}, "prefer": {{"pork": ["trader_joes_shadyside"]}}}}
  User: "no preferences"
    → {{"avoid": {{}}, "prefer": {{}}}}
"""

    raw = call_llm(prompt)
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return existing_avoid, existing_prefer

    # Back-compat: if the model returns a flat {item: [stores]}, treat as avoid.
    if isinstance(parsed, dict) and ("avoid" in parsed or "prefer" in parsed):
        new_avoid = parsed.get("avoid") or {}
        new_prefer = parsed.get("prefer") or {}
    elif isinstance(parsed, dict):
        new_avoid = parsed
        new_prefer = {}
    else:
        return existing_avoid, existing_prefer

    _merge_pref_dict(existing_avoid, new_avoid if isinstance(new_avoid, dict) else {})
    _merge_pref_dict(existing_prefer, new_prefer if isinstance(new_prefer, dict) else {})
    return existing_avoid, existing_prefer


def update_quantities_from_reply(user_reply: str, items: list[dict]) -> list[dict]:
    """
    Use LLM to update item quantities based on the user's clarification reply.
    """
    prompt = f"""The user was asked to clarify quantities for grocery items.
Update the items list with any quantities or units mentioned in the reply.

Current items:
{json.dumps(items, indent=2)}

User reply: "{user_reply}"

Return the COMPLETE updated items list as JSON array. Same format as input.
Only change quantity/unit/ambiguous fields. Do not add or remove items.
Return ONLY the JSON array, no explanation."""

    raw = call_llm(prompt)
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        updated = json.loads(raw)
        return updated if isinstance(updated, list) else items
    except json.JSONDecodeError:
        return items


def items_to_query_strings(items: list[dict]) -> list[str]:
    """
    Convert structured item dicts to search strings for the price optimizer.
    E.g. {"name": "chicken breast", "quantity": 2, "unit": "lb"} → "chicken breast 2 lb"
    """
    queries = []
    for item in items:
        parts = [item["name"]]
        if item.get("quantity") and item.get("unit"):
            parts.append(f"{item['quantity']} {item['unit']}")
        elif item.get("unit"):
            parts.append(item["unit"])
        queries.append(" ".join(parts))
    return queries


def _recompute_plan_totals(shopping_plan: dict) -> None:
    """Rebuild total_cost, store_ids and prune empty stores in place."""
    plan = shopping_plan["plan"]
    all_stores = load_stores()

    for sid in list(plan.keys()):
        if not plan[sid]:
            del plan[sid]

    shopping_plan["store_ids"] = list(plan.keys())
    shopping_plan["stores_meta"] = {
        sid: all_stores[sid] for sid in shopping_plan["store_ids"] if sid in all_stores
    }
    shopping_plan["total_cost"] = round(
        sum(i["price"] for items in plan.values() for i in items), 2
    )


def _remove_item_from_plan(shopping_plan: dict, item_entry: dict) -> bool:
    """Remove a specific item entry from whatever store currently holds it. Returns True if removed."""
    for sid, entries in shopping_plan["plan"].items():
        for e in entries:
            if e is item_entry or (
                e.get("item") == item_entry.get("item")
                and abs(e.get("price", 0) - item_entry.get("price", 0)) < 1e-9
                and sid == item_entry.get("_store_id", sid)
            ):
                entries.remove(e)
                return True
    return False


def apply_preferred_stores(shopping_plan: dict, preferred_stores: dict) -> dict:
    """
    Force items whose name matches a preferred_stores key to be sourced from
    the user's preferred store (if that store carries the item).

    preferred_stores: {"pork": ["trader_joes_shadyside"], ...}
    Priority: first store in the list that actually has the item wins.
    If no preferred store has it, leave the item where it is and record a note.
    """
    if not preferred_stores:
        shopping_plan.setdefault("unfulfilled_preferences", [])
        return shopping_plan

    from tools.price_optimizer import load_prices

    price_data = load_prices()
    all_stores = load_stores()
    unfulfilled: list[dict] = []

    for pref_item, preferred_ids in preferred_stores.items():
        if not preferred_ids:
            continue

        # Build candidate substrings: "pork chops" -> ["pork chops", "pork", "pork loin chops", ...]
        candidates = expand_query(pref_item) or [pref_item.lower()]

        # Find the current placement of this item in the plan
        current_sid = None
        current_entry = None
        for sid, entries in shopping_plan["plan"].items():
            for e in entries:
                if matches_any(e["item"], candidates):
                    current_sid = sid
                    current_entry = e
                    break
            if current_entry:
                break

        if current_entry is None:
            continue  # item wasn't on the plan in the first place

        # If already at a preferred store, nothing to do
        if current_sid in preferred_ids:
            continue

        # Try each preferred store in order
        placed = False
        for target_sid in preferred_ids:
            alt = find_at_store(pref_item, target_sid, price_data, stores=all_stores)
            if alt is None:
                continue

            shopping_plan["plan"][current_sid].remove(current_entry)

            if target_sid not in shopping_plan["plan"]:
                shopping_plan["plan"][target_sid] = []
            shopping_plan["plan"][target_sid].append({
                "item": alt["item_name"],
                "price": alt["item_price"],
                "store_display": alt.get("store", target_sid),
            })
            placed = True
            break

        if not placed:
            unfulfilled.append({
                "item": pref_item,
                "preferred_stores": list(preferred_ids),
                "reason": "not available at any preferred store",
            })

    shopping_plan["unfulfilled_preferences"] = unfulfilled
    _recompute_plan_totals(shopping_plan)
    return shopping_plan


def apply_avoid_stores(shopping_plan: dict, avoid_stores: dict) -> dict:
    """
    Move items away from stores the user wants to avoid, picking the next
    cheapest non-avoided store that carries a matching item.
    """
    if not avoid_stores:
        return shopping_plan

    from tools.price_optimizer import load_prices

    price_data = load_prices()
    items_db = price_data.get("items", {})
    all_stores = load_stores()
    name_index = {s["display_name"].lower(): sid for sid, s in all_stores.items()}

    # Pre-expand candidates for each pref item once
    pref_candidates: dict[str, list[str]] = {
        k: expand_query(k) or [k.lower()] for k in avoid_stores
    }

    for store_id in list(shopping_plan["plan"].keys()):
        items_in_store = shopping_plan["plan"][store_id]
        to_move = [
            e for e in items_in_store
            if any(
                store_id in avoided and matches_any(e["item"], pref_candidates[pref_item])
                for pref_item, avoided in avoid_stores.items()
            )
        ]

        for item_entry in to_move:
            items_in_store.remove(item_entry)

            # Identify which pref_item matched, to know the avoid list
            matched_pref = next(
                (k for k in avoid_stores if matches_any(item_entry["item"], pref_candidates[k])),
                None,
            )
            avoided = avoid_stores.get(matched_pref, []) if matched_pref else []

            # Search the same category for next cheapest non-avoided store
            best_alt_sid = None
            best_alt_entry = None
            for category, entries in items_db.items():
                if matched_pref and (category in matched_pref.lower() or matched_pref.lower() in category):
                    candidates = sorted(entries, key=lambda e: e["item_price"])
                    for cand in candidates:
                        sid = name_index.get(cand["store"].lower())
                        if sid and sid not in avoided and sid != store_id:
                            best_alt_sid = sid
                            best_alt_entry = cand
                            break
                    break

            if best_alt_sid and best_alt_entry:
                shopping_plan["plan"].setdefault(best_alt_sid, []).append({
                    "item": best_alt_entry["item_name"],
                    "price": best_alt_entry["item_price"],
                    "store_display": best_alt_entry["store"],
                })

    _recompute_plan_totals(shopping_plan)
    return shopping_plan


# Back-compat alias for any legacy callers/tests
apply_preferences = apply_avoid_stores


# --------------- Main ReAct loop ---------------

def chat(session: ShoppingSession, user_message: str) -> str:
    """
    Process one user message and return the agent's response.
    Advances the session state machine as needed.
    """
    session.add_message("user", user_message)

    # ── State: CLARIFY ────────────────────────────────────────
    if session.state == "CLARIFY":

        if not session.raw_items:
            # First message: parse shopping list + try to catch any in-line preferences
            session.raw_items = parse_items_from_message(user_message)
            session.preferences, session.preferred_stores = extract_preferences_from_reply(
                user_message, session.preferences, session.preferred_stores,
            )
        else:
            # Follow-up: update quantities + extract preferences
            session.raw_items = update_quantities_from_reply(user_message, session.raw_items)
            session.preferences, session.preferred_stores = extract_preferences_from_reply(
                user_message, session.preferences, session.preferred_stores,
            )
            session.clarification_done = True

        # Detect errand request
        if any(w in user_message.lower() for w in ["errand", "someone else", "shop for me", "pick up for me"]):
            session.want_errand = True

        has_ambiguous = any(item.get("ambiguous") for item in session.raw_items)

        if has_ambiguous or not session.clarification_done:
            # Still need more info
            prompt = CLARIFY_PROMPT.format(
                items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
                avoid_json=json.dumps(session.preferences, ensure_ascii=False),
                prefer_json=json.dumps(session.preferred_stores, ensure_ascii=False),
                user_message=user_message,
            )
            reply = call_llm(prompt)
            session.state = "CLARIFY"
        else:
            # Ready to confirm
            session.state = "CONFIRM"
            prompt = CONFIRM_PROMPT.format(
                items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
                avoid_json=json.dumps(session.preferences, ensure_ascii=False),
                prefer_json=json.dumps(session.preferred_stores, ensure_ascii=False),
            )
            reply = call_llm(prompt)

    # ── State: CONFIRM ────────────────────────────────────────
    elif session.state == "CONFIRM":

        yes_words = ["yes", "yeah", "yep", "ok", "okay", "sure", "go", "confirm", "sounds good"]
        no_words  = ["no", "nope", "cancel", "change", "wrong", "incorrect"]

        msg_lower = user_message.lower()
        is_yes = any(w in msg_lower for w in yes_words)
        is_no  = any(w in msg_lower for w in no_words)

        if is_no or (not is_yes and len(user_message.split()) > 3):
            # User wants changes — update and re-confirm
            session.raw_items = update_quantities_from_reply(user_message, session.raw_items)
            session.preferences, session.preferred_stores = extract_preferences_from_reply(
                user_message, session.preferences, session.preferred_stores,
            )
            prompt = CONFIRM_PROMPT.format(
                items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
                avoid_json=json.dumps(session.preferences, ensure_ascii=False),
                prefer_json=json.dumps(session.preferred_stores, ensure_ascii=False),
            )
            reply = call_llm(prompt)
        else:
            # Confirmed — execute
            session.state = "EXECUTE"
            reply = session_execute(session)

    # ── State: EXECUTE (start new request) ───────────────────
    elif session.state == "EXECUTE":
        session.__init__()
        session.raw_items = parse_items_from_message(user_message)
        session.state = "CLARIFY"

        has_ambiguous = any(item.get("ambiguous") for item in session.raw_items)
        if has_ambiguous:
            prompt = CLARIFY_PROMPT.format(
                items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
                avoid_json="{}",
                prefer_json="{}",
                user_message=user_message,
            )
            reply = call_llm(prompt)
        else:
            session.state = "CONFIRM"
            prompt = CONFIRM_PROMPT.format(
                items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
                avoid_json="{}",
                prefer_json="{}",
            )
            reply = call_llm(prompt)

    else:
        session.__init__()
        reply = "Starting a new session! What groceries do you need today?"

    session.add_message("agent", reply)
    return reply


def session_execute(session: ShoppingSession) -> str:
    """Run all tools and return the formatted shopping plan summary."""
    query_strings = items_to_query_strings(session.raw_items)

    shopping_plan = optimize_shopping_list(query_strings)

    # Order matters: honor "must-buy-at" first, then move items away from "avoid" stores.
    if session.preferred_stores:
        shopping_plan = apply_preferred_stores(shopping_plan, session.preferred_stores)
    if session.preferences:
        shopping_plan = apply_avoid_stores(shopping_plan, session.preferences)

    route_plan = plan_route(
        store_ids=shopping_plan["store_ids"],
        stores_meta=shopping_plan["stores_meta"],
    )

    errand_quote = None
    if session.want_errand:
        errand_quote = generate_errand_quote(shopping_plan, route_plan)

    session.shopping_plan = shopping_plan
    session.route_plan = route_plan
    session.errand_quote = errand_quote

    errand_section = ""
    if errand_quote:
        errand_section = f"Errand runner quote:\n{json.dumps(errand_quote, indent=2)}"

    prompt = SUMMARY_PROMPT.format(
        shopping_json=json.dumps(shopping_plan, ensure_ascii=False, indent=2),
        route_json=json.dumps(route_plan, ensure_ascii=False, indent=2),
        errand_section=errand_section,
    )
    return call_llm(prompt)