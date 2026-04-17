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
from tools.price_optimizer import (
    optimize_shopping_list,
    load_stores,
    find_at_store_in_cache,
    find_cheapest_in_cache_excluding,
)
from tools.synonyms import expand_query, matches_any
from tools.route_planner import plan_route
from tools.errand_runner import generate_errand_quote
from tools.recommender import recommend_for_query, format_recommendation

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


def call_llm(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Single-turn LLM call (Google GenAI or OpenRouter).

    `model` / `temperature` can override the defaults per call — useful
    for cheap side tasks like the intent router, which should run on a
    small model at temperature 0 regardless of the main chat model.
    Passing None (the default) keeps the historical behavior.
    """
    effective_model = model or LLM_MODEL
    effective_temp = 0.3 if temperature is None else float(temperature)

    if LLM_PROVIDER == "openrouter":
        client = _get_openrouter_client()
        resp = client.chat.completions.create(
            model=effective_model,
            temperature=effective_temp,
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
        model=effective_model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=effective_temp,
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

    all_stores = load_stores()
    unfulfilled: list[dict] = []

    for pref_item, preferred_ids in preferred_stores.items():
        if not preferred_ids:
            continue

        candidates = expand_query(pref_item) or [pref_item.lower()]

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
            continue

        if current_sid in preferred_ids:
            continue

        placed = False
        for target_sid in preferred_ids:
            alt = find_at_store_in_cache(pref_item, target_sid, all_stores)
            if alt is None:
                continue

            shopping_plan["plan"][current_sid].remove(current_entry)
            shopping_plan["plan"].setdefault(target_sid, []).append({
                "item": alt["item_name"],
                "price": alt["item_price"],
                "store_display": alt.get("store", target_sid),
                "url": alt.get("url"),
                "source": alt.get("_source", "cache"),
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
    cheapest non-avoided store from the cached catalog.
    """
    if not avoid_stores:
        return shopping_plan

    all_stores = load_stores()
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

            matched_pref = next(
                (k for k in avoid_stores
                 if matches_any(item_entry["item"], pref_candidates[k])),
                None,
            )
            if matched_pref is None:
                continue
            avoided = list(avoid_stores.get(matched_pref) or [])
            # Always exclude the current store too — we never want to put
            # the item back where the user just told us to avoid.
            if store_id not in avoided:
                avoided.append(store_id)

            alt = find_cheapest_in_cache_excluding(matched_pref, avoided, all_stores)
            if alt is None:
                continue

            target_sid = alt.get("_store_id")
            if not target_sid:
                continue
            shopping_plan["plan"].setdefault(target_sid, []).append({
                "item": alt["item_name"],
                "price": alt["item_price"],
                "store_display": alt.get("store", target_sid),
                "url": alt.get("url"),
                "source": alt.get("_source", "cache"),
            })

    _recompute_plan_totals(shopping_plan)
    return shopping_plan


# Back-compat alias for any legacy callers/tests
apply_preferences = apply_avoid_stores


# --------------- Recommend ("what's the best X?") side-flow ---------------

# Detect "recommend X" / "best X" / "which X should I get" style requests.
# Captures the noun phrase the user is asking about.
_RECOMMEND_PATTERNS = [
    re.compile(r"(?:please\s+)?recommend(?:\s+me)?\s+(?:some\s+)?(.+)", re.I),
    re.compile(r"what(?:'s| is)\s+the\s+best\s+(.+?)(?:\s+to\s+buy)?[\?\.!]*$", re.I),
    re.compile(r"what(?:'s| is)\s+the\s+cheapest\s+(.+?)[\?\.!]*$", re.I),
    re.compile(r"^best\s+(.+?)[\?\.!]*$", re.I),
    re.compile(r"^cheapest\s+(.+?)[\?\.!]*$", re.I),
    re.compile(r"which\s+(.+?)\s+should\s+i\s+(?:buy|get|pick)[\?\.!]*$", re.I),
    re.compile(r"suggest(?:\s+me)?\s+(?:some\s+)?(.+?)[\?\.!]*$", re.I),
    re.compile(r"top\s+\d+\s+(.+?)[\?\.!]*$", re.I),
]

# Topk hints ("top 5 X", "best 3 X")
_TOPK_PATTERN = re.compile(r"\btop\s+(\d+)\b|\bbest\s+(\d+)\b", re.I)

# Lightweight inline preference signals the user might tack onto the request.
_PREF_KEYWORD_MAP = {
    "organic": "organic",
    "grass-fed": "organic",
    "grass fed": "organic",
    "all natural": "organic",
    "cheapest": "cheapest",
    "lowest price": "cheapest",
    "best deal": "cheapest",
    "bulk": "largest-pack",
    "largest": "largest-pack",
    "biggest pack": "largest-pack",
    "premium": "premium",
    "high quality": "premium",
    "store brand": "store-brand",
    "house brand": "store-brand",
}


def detect_recommend_intent(message: str) -> dict | None:
    """Return {"query","topk","preferences"} if the message looks like a
    "what's the best X" style request, else None."""
    if not message:
        return None
    msg = message.strip()
    query: str | None = None
    for pat in _RECOMMEND_PATTERNS:
        m = pat.search(msg)
        if m:
            query = (m.group(1) or "").strip(" .?!,'\"")
            break
    if not query:
        return None

    # Strip trailing fillers like "for me", "please", "today"
    query = re.sub(r"\b(for\s+me|please|today|tonight)\b", "", query, flags=re.I).strip()
    if not query:
        return None

    topk = 3
    m = _TOPK_PATTERN.search(msg)
    if m:
        try:
            topk = max(1, min(10, int(m.group(1) or m.group(2))))
        except (TypeError, ValueError):
            pass

    msg_lower = msg.lower()
    preferences: list[str] = []
    for kw, flag in _PREF_KEYWORD_MAP.items():
        if kw in msg_lower and flag not in preferences:
            preferences.append(flag)

    # The preference keyword is often inside the captured query (e.g.
    # "best organic milk") — keep it in the query too, the LLM uses both.
    return {"query": query, "topk": topk, "preferences": preferences}


# --------------- "List options" side-flow (question, not state transition) ---

# Phrases that signal "show me options / alternatives for what we're
# talking about" — a pure question. Handler answers with cached SKUs and
# does NOT advance the session state.
_LIST_OPTIONS_PATTERNS = [
    re.compile(
        r"\b(?:list|show(?:\s+me)?|give\s+me|see)\s+"
        r"(?:the\s+|some\s+|more\s+|other\s+|all\s+|a\s+few\s+)?"
        r"(?:options|alternatives|choices|suggestions|picks|products|items|varieties)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:do|does)\s+you\s+(?:have|got)\s+"
        r"(?:any\s+|some\s+|more\s+|other\s+)?"
        r"(?:options|alternatives|choices|varieties)\b",
        re.I,
    ),
    re.compile(
        r"\bwhat\s+(?:other|else|more)\s*(?:options|choices|alternatives)?\b",
        re.I,
    ),
    re.compile(r"\b(?:any|other|more)\s+(?:options|alternatives|choices)\b", re.I),
    re.compile(r"\bshow\s+(?:me\s+)?more\b", re.I),
    re.compile(r"\bany\s+alternatives?\b", re.I),
]


def detect_list_options_intent(message: str) -> bool:
    """True iff `message` is asking to see options/alternatives."""
    if not message:
        return False
    return any(p.search(message) for p in _LIST_OPTIONS_PATTERNS)


# Junk-word filter when we backsolve a query from the active shopping plan
# item names. Keep the semantic head (e.g. "Mini Avocados (Bag)" → "avocados").
_PLAN_ITEM_STRIP_TOKENS = {
    "organic", "natural", "fresh", "frozen", "whole", "mini", "large", "small",
    "bag", "bottle", "pack", "box", "carton", "can", "jar", "loaf", "each",
    "oz", "fl", "lb", "lbs", "gal", "gallon", "count", "ct", "dozen",
}


def _clean_plan_item_name(name: str) -> str:
    """Reduce a retail item name to a queryable noun phrase.
    "Mini Avocados (Bag)" → "avocados". Cheap heuristic, not perfect."""
    if not name:
        return ""
    # strip parenthesised notes, punctuation, digits
    s = re.sub(r"\([^)]*\)", " ", name)
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    tokens = [t for t in s.lower().split() if t and t not in _PLAN_ITEM_STRIP_TOKENS]
    # return last 1-2 meaningful tokens (usually the head noun)
    if not tokens:
        return name.lower()
    return " ".join(tokens[-2:]) if len(tokens) >= 2 else tokens[-1]


def _active_query_for_options(session: ShoppingSession) -> str | None:
    """Best guess at what item the user wants options for, based on
    current session context. Preference order:
      1. raw_items (mid-list-building) — use the user's own phrasing
      2. shopping_plan (post-execute) — strip brand/size fluff
      3. None — caller should ask the user to be specific
    """
    if session.raw_items:
        names = [it.get("name") for it in session.raw_items if it.get("name")]
        if names:
            # Multiple items? Pick the first — user can be more specific next turn.
            return names[0]
    if session.shopping_plan:
        for items in (session.shopping_plan.get("plan") or {}).values():
            if items:
                return _clean_plan_item_name(items[0].get("item", "")) or None
    return None


def handle_list_options_request(session: ShoppingSession, user_message: str) -> str:
    """Render a brief top-K options list for the active item WITHOUT
    advancing the state machine. This is a "question" handler — the
    user is asking about alternatives, not progressing the shopping
    flow."""
    from tools.product_search import search_products_ranked

    query = _active_query_for_options(session)
    if not query:
        return (
            "Happy to list options — which item do you want to see alternatives for? "
            "(e.g. \"options for avocados\")"
        )

    hits = search_products_ranked(query, include_mock=False, limit=8)
    hits = [h for h in hits if h.get("item_price") is not None][:5]
    if not hits:
        return (
            f"I couldn't find cached options for \"{query}\". "
            "Try a different name, or say \"recommend <item>\" for ranked picks."
        )

    stores = load_stores()
    lines = [f"Here are {len(hits)} \"{query}\" options I have in cache:\n"]
    for i, h in enumerate(hits, 1):
        sid = h.get("store_id") or ""
        store_name = (stores.get(sid) or {}).get("name") or h.get("store") or sid
        price = h.get("item_price")
        url = h.get("url")
        line = f"{i}. **{h.get('item_name','?')}** — {store_name} — ${price:.2f}"
        if url:
            line += f"  ({url})"
        lines.append(line)
    lines.append(
        "\nSay the number to lock one in (e.g. \"pick 2\"), "
        "or keep going with your list."
    )
    return "\n".join(lines)


def handle_recommend_request(intent: dict) -> str:
    """Run the recommender and produce a friendly chat reply."""
    result = recommend_for_query(
        intent["query"],
        topk=intent["topk"],
        preferences=intent.get("preferences") or None,
    )
    body = format_recommendation(result)
    pref_note = ""
    if intent.get("preferences"):
        pref_note = f" (preferences: {', '.join(intent['preferences'])})"
    header = (
        f"Here are my top {len(result['picks'])} picks for "
        f"\"{intent['query']}\"{pref_note} from cached store data:\n"
    )
    if not result["picks"]:
        return f"I couldn't find good matches for \"{intent['query']}\" in cached store data."
    footer = "\n\nWant me to add one of these to your shopping list? Just tell me which #."
    return header + body + footer


# --------------- LLM intent fallback (opt-in) ---------------

# If the classifier is less confident than this, we bail out and let
# the deterministic state machine handle the message instead of
# committing to an LLM-picked branch.
_LLM_ROUTE_MIN_CONFIDENCE = 0.55


def _llm_route(session: ShoppingSession, user_message: str) -> str | None:
    """Run the LLM intent classifier and, if the label is actionable,
    dispatch to the matching handler directly. Returns the reply text
    on success, or None to let the caller fall back to the default
    state-machine flow.

    The branches below mirror the deterministic handlers used in
    `chat()`: we never invent new behavior, we only route unmatched
    messages to the existing one.
    """
    from tools.intent_classifier import classify_intent

    result = classify_intent(user_message, session)
    label = result.get("label", "passthrough")
    conf = result.get("confidence", 0.0)
    if label == "passthrough" or conf < _LLM_ROUTE_MIN_CONFIDENCE:
        return None

    if label == "list_options":
        # If the classifier pulled out a specific query noun, hint it via
        # a synthetic raw_items entry only when we have no context.
        if result.get("query") and not session.raw_items and not session.shopping_plan:
            session.raw_items = [{
                "name": result["query"], "quantity": None, "unit": None,
                "ambiguous": False,
            }]
        return handle_list_options_request(session, user_message)

    if label == "recommend":
        query = result.get("query") or user_message.strip()
        return handle_recommend_request({
            "query": query, "topk": 3, "preferences": [],
        })

    if label == "closer":
        session.state = "DONE"
        return (
            "You're all set — happy shopping! "
            "Just say the word when you need another list."
        )

    if label == "confirm_yes" and session.state == "CONFIRM":
        session.state = "EXECUTE"
        return session_execute(session)

    if label == "confirm_no" and session.state == "CONFIRM":
        return (
            "No problem — what would you like to change? "
            "You can add/remove items, adjust quantities, set a brand, "
            "or tell me stores to prefer/avoid."
        )

    # For "refinement" / "new_list" we let the default state machine
    # handle it (it already does the right thing for those). Return
    # None so the caller continues.
    return None


# --------------- Main ReAct loop ---------------

def chat(session: ShoppingSession, user_message: str) -> str:
    """
    Process one user message and return the agent's response.
    Advances the session state machine as needed.
    """
    session.add_message("user", user_message)

    # Side-flow: "recommend X" / "best X" — answer directly without
    # disturbing the active shopping-list state machine.
    intent = detect_recommend_intent(user_message)
    if intent:
        reply = handle_recommend_request(intent)
        session.add_message("agent", reply)
        return reply

    # Side-flow: "list options / any alternatives / show me more" — also
    # a pure question. Answer with cached SKUs and DO NOT advance the
    # state machine. The user's quantity/preference answers (if any)
    # from the same turn are ignored on purpose; they can repeat them
    # after seeing the options.
    if detect_list_options_intent(user_message):
        reply = handle_list_options_request(session, user_message)
        session.add_message("agent", reply)
        return reply

    # Hybrid fallback: when USE_LLM_INTENT_ROUTER is on and the regex
    # rules above all missed, ask a small LLM classifier what the user
    # meant. Regex stays first so 80%+ of turns pay zero extra latency.
    from config.settings import USE_LLM_INTENT_ROUTER
    if USE_LLM_INTENT_ROUTER:
        llm_label = _llm_route(session, user_message)
        if llm_label is not None:
            session.add_message("agent", llm_label)
            return llm_label

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
        bare_cancel_phrases = {
            "no", "nope", "nah", "cancel", "stop", "no thanks", "no thank you",
            "not yet", "wait", "hold on",
        }

        msg_lower = user_message.lower().strip()
        msg_stripped = msg_lower.strip(" .,!?'\"")
        is_yes = any(w in msg_lower for w in yes_words)
        is_bare_cancel = msg_stripped in bare_cancel_phrases

        # Anything past the leading "no"/filler counts as substantive content.
        # Strip common cancel-ish prefixes and punctuation to see what's left.
        refinement_body = re.sub(
            r"^\s*(no|nope|nah|not yet|wait|hold on|actually|hmm|um)[\s,\.!:;-]+",
            "",
            msg_lower,
        ).strip()
        has_refinement = bool(refinement_body) and len(refinement_body.split()) >= 2

        if is_bare_cancel:
            # Pure cancel with no guidance — ask what they'd like to change
            # instead of silently re-rendering the same plan.
            reply = (
                "No problem — what would you like to change? "
                "You can add/remove items, adjust quantities, set a brand "
                "(e.g. \"heinz ketchup\"), or tell me stores to prefer/avoid."
            )
        elif has_refinement or (not is_yes and len(user_message.split()) >= 3):
            # Refinement with substantive content — apply AND execute in one
            # shot. We already know what the user wants (the updated list);
            # re-prompting yes/no is just friction.
            session.raw_items = update_quantities_from_reply(user_message, session.raw_items)
            session.preferences, session.preferred_stores = extract_preferences_from_reply(
                user_message, session.preferences, session.preferred_stores,
            )
            session.state = "EXECUTE"
            reply = session_execute(session)
        elif is_yes:
            # Confirmed — execute
            session.state = "EXECUTE"
            reply = session_execute(session)
        else:
            # Ambiguous short reply — treat as refinement (LLM will parse)
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

    # ── State: EXECUTE (post-plan: close out, or kick off new request) ─
    elif session.state == "EXECUTE":
        msg_stripped = user_message.lower().strip(" .,!?'\"")

        # Closers / thanks / "no more" — treat as end of session, not a new list.
        _closers = {
            "no", "nope", "nah", "no thanks", "no thank you", "nothing",
            "that's all", "thats all", "that is all", "i'm good", "im good",
            "all good", "done", "we're done", "were done", "we are done",
            "ok thanks", "thanks", "thank you", "thx", "ty", "bye", "goodbye",
            "see you", "cya",
        }
        if msg_stripped in _closers or msg_stripped.startswith(
            ("no thanks", "no thank", "thanks ", "thank you ", "nothing else")
        ):
            session.state = "DONE"
            reply = (
                "You're all set — happy shopping! "
                "Just say the word when you need another list."
            )
        else:
            # Treat as a brand-new request. Start a fresh session but keep
            # the same object (caller re-use).
            session.__init__()
            session.raw_items = parse_items_from_message(user_message)
            session.preferences, session.preferred_stores = extract_preferences_from_reply(
                user_message, session.preferences, session.preferred_stores,
            )

            # Guard: if the LLM couldn't pull out any items, don't pretend we
            # have a plan. Ask for a real list instead of rendering an empty
            # CONFIRM template (which is how the old flow hallucinated).
            if not session.raw_items:
                session.state = "CLARIFY"
                reply = (
                    "I didn't catch any grocery items in that. "
                    "What would you like me to plan next? "
                    "(e.g. \"I need milk, eggs, and chicken wings\")"
                )
            else:
                has_ambiguous = any(item.get("ambiguous") for item in session.raw_items)
                if has_ambiguous:
                    session.state = "CLARIFY"
                    prompt = CLARIFY_PROMPT.format(
                        items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
                        avoid_json=json.dumps(session.preferences, ensure_ascii=False),
                        prefer_json=json.dumps(session.preferred_stores, ensure_ascii=False),
                        user_message=user_message,
                    )
                    reply = call_llm(prompt)
                else:
                    session.state = "CONFIRM"
                    prompt = CONFIRM_PROMPT.format(
                        items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
                        avoid_json=json.dumps(session.preferences, ensure_ascii=False),
                        prefer_json=json.dumps(session.preferred_stores, ensure_ascii=False),
                    )
                    reply = call_llm(prompt)

    else:
        # DONE or anything unexpected → start a fresh session and handle
        # the current message through the CLARIFY entrypoint, so the user
        # doesn't have to repeat themselves.
        session.__init__()
        session.raw_items = parse_items_from_message(user_message)
        session.preferences, session.preferred_stores = extract_preferences_from_reply(
            user_message, session.preferences, session.preferred_stores,
        )
        if not session.raw_items:
            reply = "Starting a new session! What groceries do you need today?"
        else:
            has_ambiguous = any(item.get("ambiguous") for item in session.raw_items)
            if has_ambiguous:
                session.state = "CLARIFY"
                prompt = CLARIFY_PROMPT.format(
                    items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
                    avoid_json=json.dumps(session.preferences, ensure_ascii=False),
                    prefer_json=json.dumps(session.preferred_stores, ensure_ascii=False),
                    user_message=user_message,
                )
                reply = call_llm(prompt)
            else:
                session.state = "CONFIRM"
                prompt = CONFIRM_PROMPT.format(
                    items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
                    avoid_json=json.dumps(session.preferences, ensure_ascii=False),
                    prefer_json=json.dumps(session.preferred_stores, ensure_ascii=False),
                )
                reply = call_llm(prompt)

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

    # Hard short-circuit when nothing could be priced — don't let the LLM
    # hallucinate a template-shaped plan from empty data.
    if not shopping_plan.get("plan"):
        missing = ", ".join(shopping_plan.get("not_found") or query_strings) or "(unknown)"
        session.shopping_plan = shopping_plan
        return (
            "I couldn't find any of those items in our cached store data "
            f"(tried: {missing}). Want me to search with different names, "
            "or would you like recommendations instead? "
            "(e.g. \"recommend pringles\" or \"best chips\")"
        )

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