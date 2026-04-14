# ============================================================
# agent/agent.py
# ReAct grocery agent powered by Gemini (via Google GenAI SDK).
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
from google import genai

from config.settings import GOOGLE_API_KEY, LLM_MODEL
from tools.price_optimizer import optimize_shopping_list, load_stores
from tools.route_planner import plan_route
from tools.errand_runner import generate_errand_quote

client = genai.Client(api_key=GOOGLE_API_KEY)

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

User's store preferences (items they don't want from certain stores):
{preferences_json}

Your task: Ask the user ONE friendly message that covers ALL clarifications needed.
Ask about:
1. Quantities for any ambiguous items (items with ambiguous=true) — mention bulk discounts if relevant
2. Whether they have store preferences (e.g. don't want to buy meat at Trader Joe's)
   — only ask this if preferences are not yet set

Keep it short. Max 3-4 lines. Ask everything in ONE message, not multiple.
User's original message: {user_message}"""


CONFIRM_PROMPT = """You are a grocery shopping assistant. Present a brief summary of the shopping plan
for the user to confirm before executing.

Items with quantities:
{items_json}

Store preferences (avoid these store-item combinations):
{preferences_json}

Write a short 3-5 line confirmation message asking the user to confirm.
List the items and any special preferences applied.
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
4. (If errand quote exists) Errand runner option

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
        self.preferences = {}           # {"chicken": ["trader_joes_shadyside"]}
        self.conversation_history = []  # list of {"role": "user"/"agent", "text": str}
        self.shopping_plan = None
        self.route_plan = None
        self.errand_quote = None
        self.clarification_done = False
        self.want_errand = False

    def add_message(self, role: str, text: str):
        self.conversation_history.append({"role": role, "text": text})


# --------------- LLM helpers ---------------

def call_llm(prompt: str) -> str:
    """Single-turn LLM call using new google-genai SDK."""
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


def extract_preferences_from_reply(user_reply: str, existing_prefs: dict) -> dict:
    """
    Use LLM to extract store preferences from the user's clarification reply.
    E.g. "don't buy meat at Trader Joe's" → {"chicken": ["trader_joes_shadyside"]}
    """
    store_map = {s["name"].lower(): s["id"] for s in load_stores().values()}
    store_list = ", ".join(f"{v} ({k})" for k, v in store_map.items())

    prompt = f"""Extract any store preferences from the user's message.
A preference means the user does NOT want to buy a certain item at a certain store.

Available store IDs: {store_list}

User message: "{user_reply}"

Return ONLY valid JSON mapping item names to a list of store IDs to avoid.
If no preferences mentioned, return {{}}.
Example: {{"chicken": ["trader_joes_shadyside"], "beef": ["whole_foods_east_liberty"]}}"""

    raw = call_llm(prompt)
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        new_prefs = json.loads(raw)
        for item, stores in new_prefs.items():
            if item in existing_prefs:
                existing_prefs[item] = list(set(existing_prefs[item] + stores))
            else:
                existing_prefs[item] = stores
        return existing_prefs
    except json.JSONDecodeError:
        return existing_prefs


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


def apply_preferences(shopping_plan: dict, preferences: dict) -> dict:
    """
    Re-assign items that violate user preferences to the next cheapest store.
    """
    if not preferences:
        return shopping_plan

    from tools.price_optimizer import load_prices

    price_data = load_prices()

    for store_id in list(shopping_plan["plan"].keys()):
        items_in_store = shopping_plan["plan"][store_id]
        to_move = []

        for item_entry in items_in_store:
            item_name = item_entry["item"]
            for pref_item, avoided_stores in preferences.items():
                if pref_item.lower() in item_name.lower() and store_id in avoided_stores:
                    to_move.append(item_entry)
                    break

        for item_entry in to_move:
            items_in_store.remove(item_entry)

            # Find next cheapest non-avoided store
            best_alt_store = None
            best_alt_price = float("inf")
            avoided = preferences.get(
                next((k for k in preferences if k.lower() in item_entry["item"].lower()), ""),
                []
            )

            for product in price_data["products"]:
                if item_entry["item"] in product["canonical_name"]:
                    for sid, price in sorted(product["prices"].items(), key=lambda x: x[1]):
                        if sid not in avoided and price < best_alt_price:
                            best_alt_price = price
                            best_alt_store = sid
                    break

            if best_alt_store:
                if best_alt_store not in shopping_plan["plan"]:
                    shopping_plan["plan"][best_alt_store] = []
                    if best_alt_store not in shopping_plan["store_ids"]:
                        shopping_plan["store_ids"].append(best_alt_store)
                        all_stores = load_stores()
                        if best_alt_store in all_stores:
                            shopping_plan["stores_meta"][best_alt_store] = all_stores[best_alt_store]

                shopping_plan["plan"][best_alt_store].append({
                    "item": item_entry["item"],
                    "price": best_alt_price,
                })
                shopping_plan["total_cost"] = round(sum(
                    i["price"]
                    for items in shopping_plan["plan"].values()
                    for i in items
                ), 2)

        # Remove now-empty stores
        if not items_in_store and store_id in shopping_plan["plan"]:
            del shopping_plan["plan"][store_id]
            shopping_plan["store_ids"] = [s for s in shopping_plan["store_ids"] if s != store_id]
            shopping_plan["stores_meta"].pop(store_id, None)

    return shopping_plan


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
            # First message: parse shopping list
            session.raw_items = parse_items_from_message(user_message)
        else:
            # Follow-up: update quantities + extract preferences
            session.raw_items = update_quantities_from_reply(user_message, session.raw_items)
            session.preferences = extract_preferences_from_reply(user_message, session.preferences)
            session.clarification_done = True

        # Detect errand request
        if any(w in user_message.lower() for w in ["errand", "someone else", "shop for me", "pick up for me"]):
            session.want_errand = True

        has_ambiguous = any(item.get("ambiguous") for item in session.raw_items)

        if has_ambiguous or not session.clarification_done:
            # Still need more info
            prompt = CLARIFY_PROMPT.format(
                items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
                preferences_json=json.dumps(session.preferences, ensure_ascii=False),
                user_message=user_message,
            )
            reply = call_llm(prompt)
            session.state = "CLARIFY"
        else:
            # Ready to confirm
            session.state = "CONFIRM"
            prompt = CONFIRM_PROMPT.format(
                items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
                preferences_json=json.dumps(session.preferences, ensure_ascii=False),
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
            session.preferences = extract_preferences_from_reply(user_message, session.preferences)
            prompt = CONFIRM_PROMPT.format(
                items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
                preferences_json=json.dumps(session.preferences, ensure_ascii=False),
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
                preferences_json="{}",
                user_message=user_message,
            )
            reply = call_llm(prompt)
        else:
            session.state = "CONFIRM"
            prompt = CONFIRM_PROMPT.format(
                items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
                preferences_json="{}",
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

    if session.preferences:
        shopping_plan = apply_preferences(shopping_plan, session.preferences)

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