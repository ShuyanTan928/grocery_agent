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
        # Last list_options hits, kept so the user can say "pick 3" / "option 2"
        # and we can resolve that back to a concrete SKU. Cleared once consumed.
        self.last_options: list[dict] = []
        # A dish waiting on user approval to be merged into raw_items.
        # Populated by handle_dish_request; consumed / cleared by the
        # follow-up dish-confirm branch in chat().
        # Shape: {"name", "cuisine", "ingredients": [{name, quantity, unit, pantry}], "source"}
        self.pending_dish: dict | None = None

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


def _list_options_via_llm(query: str, topk: int = 5) -> list[dict]:
    """Run the recommender and project its LLM-ranked picks into the
    {item_name, item_price, store_id, url, reason} shape the list_options
    UI expects. Returns [] on no candidates or on LLM failure — caller
    is responsible for the fallback.
    """
    from tools.recommender import recommend_for_query
    try:
        result = recommend_for_query(query, topk=topk)
    except Exception:
        return []

    picks = result.get("picks") or []
    out: list[dict] = []
    for p in picks:
        c = p.get("candidate") or {}
        out.append({
            "item_name": c.get("name", "?"),
            "item_price": c.get("price"),
            "store_id": c.get("store_id") or "",
            "url": c.get("url"),
            "reason": p.get("reason") or "",
        })
    return out


def _list_options_via_search(query: str, topk: int = 5) -> list[dict]:
    """Raw cache search, tier-ordered — no LLM call. Used when
    USE_LLM_LIST_OPTIONS is off or the LLM path failed."""
    from tools.product_search import search_products_ranked
    hits = search_products_ranked(query, include_mock=False, limit=topk * 2)
    hits = [h for h in hits if h.get("item_price") is not None][:topk]
    return [
        {
            "item_name": h.get("item_name", "?"),
            "item_price": h.get("item_price"),
            "store_id": h.get("store_id") or "",
            "url": h.get("url"),
            "reason": "",
        }
        for h in hits
    ]


def handle_list_options_request(session: ShoppingSession, user_message: str) -> str:
    """Render a brief top-K options list for the active item WITHOUT
    advancing the state machine. This is a "question" handler — the
    user is asking about alternatives, not progressing the shopping
    flow.

    When USE_LLM_LIST_OPTIONS is on (default), we pass the cache
    candidates through the recommender's LLM so it can filter out
    brand-name drift (e.g. "lamb" → "Lamb Weston fries"). If the LLM
    path is off or returns nothing usable, we fall back to the raw
    relevance-tier search so the user at least sees something.
    """
    from config.settings import USE_LLM_LIST_OPTIONS

    query = _active_query_for_options(session)
    if not query:
        return (
            "Happy to list options — which item do you want to see alternatives for? "
            "(e.g. \"options for avocados\")"
        )

    options: list[dict] = []
    if USE_LLM_LIST_OPTIONS:
        options = _list_options_via_llm(query, topk=5)
    if not options:
        # Safety net: cheaper and deterministic.
        options = _list_options_via_search(query, topk=5)

    if not options:
        session.last_options = []
        return (
            f"I couldn't find cached options for \"{query}\". "
            "Try a different name, or say \"recommend <item>\" for ranked picks."
        )

    stores = load_stores()
    # Snapshot so "pick 3" on the next turn resolves back to a concrete
    # SKU instead of re-running search (which would re-introduce the
    # "Lamb Weston fries" brand-name drift).
    session.last_options = [
        {
            "item_name": o["item_name"],
            "item_price": o["item_price"],
            "store_id": o["store_id"],
            "store_display": (stores.get(o["store_id"]) or {}).get("name") or o["store_id"],
            "url": o.get("url"),
        }
        for o in options
    ]

    lines = [f"Here are {len(options)} \"{query}\" options I have in cache:\n"]
    for i, o in enumerate(options, 1):
        sid = o["store_id"]
        store_name = (stores.get(sid) or {}).get("name") or sid
        price = o["item_price"]
        line = f"{i}. **{o['item_name']}** — {store_name} — ${price:.2f}"
        if o.get("url"):
            line += f"  ({o['url']})"
        lines.append(line)
        if o.get("reason"):
            lines.append(f"   → {o['reason']}")
    lines.append(
        "\nSay the number to lock one in (e.g. \"pick 2\"), "
        "or keep going with your list."
    )
    return "\n".join(lines)


# --------------- "pick N" side-flow ---------------
#
# Only fires when session.last_options is populated (i.e. we just showed
# a list_options reply). Catches:
#   "pick 3", "choose 2", "I'll take 4", "go with 1",
#   "I prefer 3", "I want #2", "option 3", "number 4", "#3", bare "3".
# Guardrails: N must be in range; last_options is cleared after a pick.
_PICK_PATTERNS = [
    re.compile(r"^\s*#?\s*(\d+)\s*[.!?]?\s*$"),
    re.compile(
        r"\b(?:pick|choose|take|want|prefer|go\s+with|i['\u2019]?ll\s+take)"
        r"\s+(?:option\s*|number\s*|item\s*|#\s*)?(\d+)\b",
        re.I,
    ),
    re.compile(r"\b(?:option|number|item|no\.?)\s*#?\s*(\d+)\b", re.I),
    re.compile(r"#\s*(\d+)\b"),
]


def detect_pick_intent(message: str, session: ShoppingSession) -> int | None:
    """Return 1-indexed pick number iff the user is selecting one of the
    options we just showed. Returns None if there's nothing to pick from
    or the message doesn't look like a pick."""
    if not message or not session.last_options:
        return None
    for pat in _PICK_PATTERNS:
        m = pat.search(message)
        if not m:
            continue
        try:
            n = int(m.group(1))
        except (ValueError, IndexError):
            continue
        if 1 <= n <= len(session.last_options):
            return n
    return None


def handle_pick_request(session: ShoppingSession, pick_num: int) -> str:
    """Lock the user's choice from `session.last_options` into a one-item
    shopping plan. Skips `optimize_shopping_list` entirely so we don't
    re-derive the item from a fuzzy text query (which is exactly how
    'lamb' drifted to 'Lamb Weston fries' before)."""
    opt = session.last_options[pick_num - 1]
    stores = load_stores()

    sid = opt.get("store_id") or ""
    store_meta = stores.get(sid, {}) if sid else {}
    store_display = (
        store_meta.get("name")
        or opt.get("store_display")
        or sid
        or "the store"
    )
    try:
        price = float(opt.get("item_price") or 0.0)
    except (TypeError, ValueError):
        price = 0.0

    plan_entry = {
        "item": opt.get("item_name", "?"),
        "price": price,
        "store_display": store_display,
        "url": opt.get("url"),
        "source": "user_pick",
    }
    shopping_plan = {
        "plan": {sid: [plan_entry]} if sid else {},
        "total_cost": round(price, 2),
        "not_found": [],
        "store_ids": [sid] if sid else [],
        "stores_meta": {sid: store_meta} if sid and store_meta else {},
    }

    # Build a route if we have a resolvable store.
    route_plan = None
    if shopping_plan["store_ids"]:
        try:
            route_plan = plan_route(
                store_ids=shopping_plan["store_ids"],
                stores_meta=shopping_plan["stores_meta"],
            )
        except Exception:
            route_plan = None

    session.shopping_plan = shopping_plan
    session.route_plan = route_plan
    session.state = "EXECUTE"
    # Consume the options — otherwise a later bare "3" would re-pick.
    session.last_options = []

    addr = store_meta.get("address") if store_meta else None
    addr_line = f" ({addr})" if addr else ""
    url_line = f"\nProduct link: {opt['url']}" if opt.get("url") else ""

    return (
        f"Locked in #{pick_num}: **{opt.get('item_name', '?')}** at "
        f"{store_display}{addr_line} for **${price:.2f}**.{url_line}\n\n"
        "Want me to add anything else, or are you all set?"
    )


# --------------- dish → ingredients side-flow ---------------
#
# Users often name a dish ("I want to make carbonara") instead of
# individual SKUs. We:
#   1. detect_dish_intent: regex for common phrasings.
#   2. handle_dish_request: resolve via tools.dish_resolver → stash the
#      suggested ingredients in session.pending_dish and show the user
#      a numbered confirm prompt. Does NOT advance state yet.
#   3. handle_dish_confirm: when session.pending_dish is set and the
#      next message is yes / "add all" / "add 1 3 5" / no — either
#      splice the chosen ingredients into session.raw_items and move
#      to CLARIFY / CONFIRM, or drop the proposal.

_DISH_PATTERNS = [
    # cook intents
    re.compile(
        r"\b(?:i\s+(?:want\s+to|wanna|'?d\s+like\s+to|plan\s+to)\s+)?"
        r"(?:cook|make|prepare|do|fix)\s+"
        r"(?:some\s+|a\s+|an\s+|the\s+)?"
        r"(?P<dish>[a-z][a-z '&-]{2,})",
        re.I,
    ),
    # recipe / ingredient lookups
    re.compile(
        r"\b(?:ingredients?\s+(?:for|of)|recipe\s+(?:for|of|to\s+make))\s+"
        r"(?P<dish>[a-z][a-z '&-]{2,})",
        re.I,
    ),
    re.compile(
        r"\b(?:what\s+do\s+i\s+need\s+(?:to\s+make|for))\s+"
        r"(?P<dish>[a-z][a-z '&-]{2,})",
        re.I,
    ),
    re.compile(
        r"\b(?:let'?s|shall\s+we)\s+(?:cook|make)\s+"
        r"(?P<dish>[a-z][a-z '&-]{2,})",
        re.I,
    ),
    re.compile(
        r"\b(?:i\s+feel\s+like|craving)\s+"
        r"(?P<dish>[a-z][a-z '&-]{2,})",
        re.I,
    ),
]

# Words that never make sense as a dish name; prevents "I want to cook you
# dinner" kind of false positives from the regex above.
_DISH_STOPWORDS = {
    "dinner", "lunch", "breakfast", "something", "anything", "food",
    "you", "it", "this", "that", "tonight", "later", "tomorrow",
    "some", "a", "the",
}


def detect_dish_intent(message: str) -> str | None:
    """Return the extracted dish name (raw, user phrasing) iff the
    message looks like a "make / cook / recipe for X" request. Trims
    trailing filler like "please", "tonight", "for dinner"."""
    if not message:
        return None
    for pat in _DISH_PATTERNS:
        m = pat.search(message)
        if not m:
            continue
        dish = (m.group("dish") or "").strip().strip(".,!?'\"")
        # Strip trailing filler clauses
        dish = re.sub(
            r"\s+(for|tonight|tomorrow|later|please|today|this\s+\w+)\b.*$",
            "",
            dish,
            flags=re.I,
        ).strip()
        if not dish:
            continue
        first_word = dish.split()[0].lower()
        if first_word in _DISH_STOPWORDS or dish.lower() in _DISH_STOPWORDS:
            continue
        return dish
    return None


def _render_dish_proposal(dish: dict) -> str:
    """Pretty-print a resolved dish for the confirm prompt."""
    header_bits = [f"**{dish['name']}**"]
    if dish.get("cuisine") and dish["cuisine"] != "other":
        header_bits.append(f"({dish['cuisine'].title()})")
    header = " ".join(header_bits)

    buyable = [i for i in dish["ingredients"] if not i.get("pantry")]
    pantry = [i for i in dish["ingredients"] if i.get("pantry")]

    lines = [f"To make {header}, here's the shopping list I'd suggest:\n"]
    for idx, ing in enumerate(buyable, 1):
        qty = ing.get("quantity")
        unit = ing.get("unit") or ""
        qty_str = ""
        if qty is not None:
            qty_str = f" — {qty}{(' ' + unit) if unit else ''}"
        lines.append(f"{idx}. {ing['name']}{qty_str}")
    if pantry:
        pantry_names = ", ".join(p["name"] for p in pantry)
        lines.append(f"\n_Skipping pantry items you likely already have: {pantry_names}._")

    src = dish.get("source", "seed")
    if src == "llm":
        lines.append("\n(Ingredients suggested by the LLM — double-check for accuracy.)")

    lines.append(
        "\nAdd all to your shopping list? (`yes` / `no` / `only 1 3 5` "
        "to cherry-pick by number)"
    )
    return "\n".join(lines)


def handle_dish_request(session: ShoppingSession, dish_name: str) -> str:
    """Resolve `dish_name` and stash the proposal on `session.pending_dish`.
    Returns the confirm prompt to send to the user. Does NOT touch
    session.state or raw_items — that's the confirm step's job."""
    from tools.dish_resolver import resolve_dish
    dish = resolve_dish(dish_name)
    if dish is None or not dish.get("ingredients"):
        session.pending_dish = None
        return (
            f"I don't have a recipe for \"{dish_name}\" yet. "
            "You can enable the LLM fallback with `USE_LLM_DISH_FALLBACK=1`, "
            "or just list the ingredients you need directly."
        )
    session.pending_dish = dish
    return _render_dish_proposal(dish)


# Digits-only selection, e.g. "1 3 5", "1,3,5", "only 2 and 4".
_DISH_PICK_NUMS_RE = re.compile(r"\b(\d+)\b")


def handle_dish_confirm(session: ShoppingSession, user_message: str) -> str:
    """Called when session.pending_dish is set. Interprets the user's
    reply as:
      - yes / add all / all / sure      → splice all non-pantry ings
      - no / cancel / skip              → drop the proposal
      - "1 3 5" / "only 2 4" / "1,2"    → splice those 1-indexed picks
      - include pantry / with pantry    → also splice pantry staples
    Advances to CONFIRM (or CLARIFY if something's ambiguous) on accept.
    On rejection, stays in CLARIFY and apologizes briefly.
    """
    from tools.dish_resolver import ingredients_to_raw_items

    dish = session.pending_dish or {}
    all_ings = dish.get("ingredients") or []
    if not all_ings:
        session.pending_dish = None
        session.state = "CLARIFY"
        return (
            "That dish didn't come with any ingredients — tell me what "
            "you need and I'll take it from there."
        )

    msg = (user_message or "").strip().lower()
    # Tokenized form for robust yes/no detection — strips punctuation so
    # "yes, with pantry" parses the same as "yes with pantry".
    msg_tokens = set(re.findall(r"[a-z']+", msg))
    include_pantry = any(w in msg for w in ("with pantry", "include pantry", "plus pantry"))

    # Rejections first — explicit "no" / cancel / etc.
    if msg_tokens & {"no", "nope", "nah", "cancel", "skip"} \
            or "never mind" in msg or "forget it" in msg:
        session.pending_dish = None
        session.state = "CLARIFY"
        return (
            f"No problem — dropped \"{dish.get('name','')}\". "
            "What would you like to shop for?"
        )

    chosen: list[dict] | None = None
    yes_tokens = {"yes", "yeah", "yep", "sure", "ok", "okay", "add", "all"}
    phrase_accepts = ("sounds good", "go ahead", "let's do it", "lets do it",
                      "add all", "all of them")
    if msg_tokens & yes_tokens or any(p in msg for p in phrase_accepts):
        chosen = ingredients_to_raw_items(all_ings, include_pantry=include_pantry)

    if chosen is None:
        nums = [int(x) for x in _DISH_PICK_NUMS_RE.findall(msg)]
        buyable = [i for i in all_ings if not i.get("pantry")] if not include_pantry else all_ings
        if nums:
            picked: list[dict] = []
            for n in nums:
                if 1 <= n <= len(buyable):
                    picked.append(buyable[n - 1])
            if picked:
                chosen = ingredients_to_raw_items(picked, include_pantry=True)
                # include_pantry=True above is intentional: we already
                # hand-picked, so don't re-filter.

    if not chosen:
        # Ambiguous reply: keep the pending proposal and ask again.
        return (
            "Got it — reply `yes` to add everything, `no` to skip the dish, "
            "or list the numbers you want (e.g. `1 3 5`)."
        )

    # Splice into existing list, deduping by lowercase name
    existing_names = {
        (it.get("name") or "").strip().lower()
        for it in (session.raw_items or [])
    }
    added: list[str] = []
    for it in chosen:
        nm = (it.get("name") or "").strip().lower()
        if nm and nm not in existing_names:
            session.raw_items.append(it)
            existing_names.add(nm)
            added.append(it["name"])

    session.pending_dish = None

    if not added:
        session.state = "CLARIFY"
        return (
            "Looks like those were already on your list. "
            "Anything else to add?"
        )

    # If nothing in raw_items is ambiguous and user is done clarifying,
    # jump to CONFIRM so they can approve the full plan in one shot.
    has_ambiguous = any(it.get("ambiguous") for it in session.raw_items)
    if has_ambiguous:
        session.state = "CLARIFY"
        return (
            f"Added {len(added)} item(s) for {dish.get('name','the dish')}: "
            f"{', '.join(added)}. Anything else, or should I plan these?"
        )
    session.clarification_done = True
    session.state = "CONFIRM"
    prompt = CONFIRM_PROMPT.format(
        items_json=json.dumps(session.raw_items, ensure_ascii=False, indent=2),
        avoid_json=json.dumps(session.preferences, ensure_ascii=False),
        prefer_json=json.dumps(session.preferred_stores, ensure_ascii=False),
    )
    return call_llm(prompt)


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


# --------------- Remove / justify side-flows ---------------

# Regex side-flows for "drop this" / "why did you pick this" — same pattern
# as detect_pick_intent / detect_list_options_intent. These run BEFORE the
# LLM router so we don't rely on a 1B-param classifier to recognize a
# common and high-stakes intent (routing "remove X" to a session reset is
# catastrophic UX).

# Words we strip from the captured target, both at the head (articles,
# determiners) and the tail (trailing prepositional phrases like "from
# the list", "please"). Keeps just the product noun.
_REMOVE_TARGET_HEAD_STRIP = re.compile(
    r"^(?:the|that|those|my|our|a|an|any|some|all\s+(?:the|of))\s+",
    re.I,
)
_REMOVE_TARGET_TAIL_STRIP = re.compile(
    r"\s*(?:"
    r"from\s+(?:the\s+)?(?:list|plan|cart|order)|"
    r"in\s+(?:the\s+)?(?:list|plan|cart)|"
    r"off\s+(?:the\s+)?(?:list|plan|cart)|"
    r"out\s+of\s+(?:the\s+)?(?:list|plan|cart)|"
    r"please|pls|thanks|thank\s+you|thx|anymore|any\s+more|"
    r"right\s+now|for\s+now|this\s+time|today|tonight|tomorrow"
    r")\s*[.!?]*\s*$",
    re.I,
)

_REMOVE_PATTERNS = [
    # "remove X" / "drop X" / "delete X" / "skip X" / "take out X" / "get rid of X"
    re.compile(
        r"^\s*(?:please\s+|actually\s+|hmm\s+)?"
        r"(?:remove|drop|delete|skip|omit|forget|cancel|scrap|ditch|exclude|"
        r"take\s+(?:out|off)|get\s+rid\s+of|leave\s+(?:out|off)|hold\s+(?:off\s+on|the))\s+"
        r"(.+?)\s*[.!?]*\s*$",
        re.I,
    ),
    # "I don't want/need X" / "i don't need the X"
    re.compile(
        r"^\s*(?:no\s*,?\s+|actually\s+)?"
        r"(?:i\s+)?(?:don'?t|do\s+not)\s+"
        r"(?:want|need|like)\s+"
        r"(.+?)\s*[.!?]*\s*$",
        re.I,
    ),
    # "no X" / "no X please" — only when it's clearly a target, not a bare "no".
    # Requires at least 2 non-stopword tokens after "no".
    re.compile(
        r"^\s*no\s+(.+?)\s*[.!?]*\s*$",
        re.I,
    ),
    # "without the X"
    re.compile(r"^\s*(?:with|w)\/?out\s+(.+?)\s*[.!?]*\s*$", re.I),
]

# Minimum non-stopword tokens required in the extracted target. Guards
# against false positives like "no" → empty target, or "I don't want that".
_REMOVE_TARGET_STOPWORDS = {
    "", "the", "that", "this", "it", "them", "any", "anything",
    "more", "else", "other", "another", "one", "some", "all",
}


def _clean_remove_target(raw: str) -> str:
    """Trim articles/prepositional tails off a captured target phrase.
    Returns an empty string if nothing substantive is left."""
    if not raw:
        return ""
    out = raw.strip().strip(",.!?")
    # Strip determiners at the head, potentially repeated ("all of the X").
    prev = None
    while prev != out:
        prev = out
        out = _REMOVE_TARGET_HEAD_STRIP.sub("", out).strip()
    out = _REMOVE_TARGET_TAIL_STRIP.sub("", out).strip()
    return out.strip(" .,!?\"'")


def detect_remove_intent(message: str, session: ShoppingSession) -> str | None:
    """Return the target noun if `message` is a remove request we can act
    on, else None. Requires an active plan or existing raw_items — with
    nothing to remove from, the handler would be a no-op.
    """
    if not message:
        return None
    has_something_to_remove = bool(session.raw_items) or bool(
        (session.shopping_plan or {}).get("plan")
    )
    if not has_something_to_remove:
        return None

    for pat in _REMOVE_PATTERNS:
        m = pat.match(message)
        if not m:
            continue
        target = _clean_remove_target(m.group(1))
        if not target:
            continue
        target_tokens = [t for t in re.findall(r"[a-z]+", target.lower())
                         if t not in _REMOVE_TARGET_STOPWORDS]
        if not target_tokens:
            continue
        # Extra guard for the bare "no X" branch: too short can be a cancel
        # ("no thanks") rather than a remove target.
        if pat.pattern.startswith("^\\s*no\\s+") and len(target_tokens) < 1:
            continue
        return target
    return None


_JUSTIFY_PATTERNS = [
    # "why did you pick/provide/choose/select/include X"
    re.compile(
        r"^\s*why\s+(?:did\s+you|are\s+you|do\s+you|would\s+you|have\s+you)\s+"
        r"(?:pick(?:ed|ing)?|provid(?:e|ed|ing)|giv(?:e|en|ing)|chos(?:e|en)|"
        r"choos(?:e|ing)|select(?:ed|ing)?|put|includ(?:e|ed|ing)|add(?:ed|ing)?|"
        r"hav(?:e|ing)|recommend(?:ed|ing)?|suggest(?:ed|ing)?)\s+"
        r"(?:me\s+)?(.+?)\s*[\?.!]*\s*$",
        re.I,
    ),
    # "why is X in/on the list/plan"
    re.compile(
        r"^\s*why\s+(?:is|are)\s+(?:the\s+|that\s+|a\s+|my\s+)?"
        r"(.+?)\s+(?:in|on|part\s+of|here|there)\b.*?[\?.!]*\s*$",
        re.I,
    ),
    # "explain X" / "explain the X"
    re.compile(r"^\s*explain\s+(?:the\s+|that\s+|my\s+)?(.+?)\s*[\?.!]*\s*$", re.I),
]


def detect_justify_intent(message: str, session: ShoppingSession) -> str | None:
    """Return the target noun if `message` is a 'why is X in my plan'
    question. Fires whenever the phrasing matches — handle_justify_request
    itself decides how to respond if there's no plan yet (answering "I
    haven't picked anything for you yet" is strictly better than silently
    falling through to the state machine's refinement branch, which would
    treat the question as a change request)."""
    if not message:
        return None
    for pat in _JUSTIFY_PATTERNS:
        m = pat.match(message)
        if not m:
            continue
        target = _clean_remove_target(m.group(1))
        if target and any(t not in _REMOVE_TARGET_STOPWORDS
                          for t in re.findall(r"[a-z]+", target.lower())):
            return target
    return None


def _tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens, used for loose target matching."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _item_matches_target(name: str, target_tokens: list[str]) -> bool:
    """True if EVERY token in target_tokens appears (whole-word) in name.
    Matches both raw_items entries (ingredient phrases) and plan entries
    (SKU names). Empty target → False, so callers don't accidentally nuke
    the whole list."""
    if not target_tokens:
        return False
    name_tokens = set(_tokens(name))
    return all(t in name_tokens for t in target_tokens)


def _remove_match_level(name: str, target_norm: str, target_tokens: list[str]) -> int:
    """0 = no match, 1 = loose token-subset, 2 = exact normalized name.
    Exact matches take precedence to avoid "orange" nuking "orange juice"."""
    if not name:
        return 0
    name_norm = name.strip().lower()
    if name_norm == target_norm:
        return 2
    if _item_matches_target(name, target_tokens):
        return 1
    return 0


def handle_remove_request(session: "ShoppingSession", target: str) -> str:
    """Drop items matching `target` from both `raw_items` and any existing
    `shopping_plan`, without resetting the session.

    Two-pass matching avoids over-deletion:
      - Pass 1: exact normalized name match. If anything hits, use only those.
      - Pass 2: loose token-subset match; if AMBIGUOUS across raw_items
        (multiple different names hit), ask the user to clarify rather
        than silently nuking the wrong item.

    Re-optimize is deliberately skipped: pruning in place preserves prices
    the user hasn't asked us to touch.
    """
    target_norm = target.strip().lower()
    target_tokens = _tokens(target)
    if not target_tokens:
        return (
            "I caught a remove request but couldn't tell which item to drop. "
            "Try naming it, e.g. \"remove water\" or \"drop the ginger\"."
        )

    # ── Score matches in raw_items ─────────────────────────────
    raw_scored: list[tuple[int, dict, int]] = []  # (index, item, level)
    for idx, it in enumerate(session.raw_items or []):
        lvl = _remove_match_level(it.get("name", ""), target_norm, target_tokens)
        if lvl > 0:
            raw_scored.append((idx, it, lvl))

    # Prefer exact matches if any
    if any(lvl == 2 for _, _, lvl in raw_scored):
        raw_scored = [h for h in raw_scored if h[2] == 2]
    else:
        # Ambiguous loose match across distinct names → ask, don't guess.
        distinct_names = {it.get("name", "").strip().lower() for _, it, _ in raw_scored}
        if len(distinct_names) > 1:
            names_fmt = ", ".join(sorted({it.get("name", "") for _, it, _ in raw_scored}))
            return (
                f"\"{target}\" matches multiple items on your list: {names_fmt}. "
                "Which one should I drop? (say the full name)"
            )

    # ── Score matches in plan ─────────────────────────────────
    # Strategy:
    #   a) If we're removing raw_items, ALSO remove plan entries whose
    #      `source_item` (back-pointer set by optimize_shopping_list) ties
    #      back to a removed raw_item. This avoids SKU-name fuzziness
    #      (e.g. raw "orange" → SKU "Navel Oranges" wouldn't match via
    #      tokens otherwise, because "orange" ≠ "oranges").
    #   b) Additionally, include plan entries that exactly-match the
    #      target SKU name by itself, for cases where the user names the
    #      SKU directly ("remove simply orange juice").
    plan = (session.shopping_plan or {}).get("plan") or {}
    source_keys_to_drop: set[str] = {
        (it.get("name") or "").strip().lower()
        for _, it, _ in raw_scored
    }

    plan_scored: list[tuple[str, dict, int]] = []
    for sid, entries in plan.items():
        for entry in entries:
            src = (entry.get("source_item") or "").strip().lower()
            if src:
                # Back-pointer available → use it as the authoritative
                # signal (avoids SKU-name singular/plural fuzziness).
                if src in source_keys_to_drop:
                    plan_scored.append((sid, entry, 2))
                continue
            # No back-pointer (legacy cache, manually-built plan, or a
            # direct SKU-name query like "remove simply orange juice")
            # → fall back to token-based matching against the SKU name.
            lvl = _remove_match_level(entry.get("item", ""), target_norm, target_tokens)
            if lvl > 0:
                plan_scored.append((sid, entry, lvl))

    # Apply deletions
    removed_from_items: list[str] = []
    if raw_scored:
        drop_idx = {idx for idx, _, _ in raw_scored}
        kept: list[dict] = []
        for i, it in enumerate(session.raw_items or []):
            if i in drop_idx:
                removed_from_items.append(it.get("name", ""))
            else:
                kept.append(it)
        session.raw_items = kept

    removed_from_plan: list[str] = []
    if plan_scored:
        drop_keys = {(sid, entry.get("item", "")) for sid, entry, _ in plan_scored}
        new_plan: dict[str, list] = {}
        new_total = 0.0
        for sid, entries in plan.items():
            kept_entries = []
            for entry in entries:
                if (sid, entry.get("item", "")) in drop_keys:
                    removed_from_plan.append(entry.get("item", ""))
                else:
                    kept_entries.append(entry)
                    new_total += float(entry.get("price") or 0.0)
            if kept_entries:
                new_plan[sid] = kept_entries
        session.shopping_plan["plan"] = new_plan
        session.shopping_plan["total_cost"] = round(new_total, 2)
        session.shopping_plan["store_ids"] = list(new_plan.keys())
        session.shopping_plan["stores_meta"] = {
            sid: meta for sid, meta in (session.shopping_plan.get("stores_meta") or {}).items()
            if sid in new_plan
        }

    if not removed_from_items and not removed_from_plan:
        return (
            f"I couldn't find anything matching \"{target}\" in your list or plan. "
            "Want to see what's currently planned?"
        )

    parts = [f"Got it — dropped \"{target}\" from your shopping list."]
    if removed_from_plan:
        parts.append(f"Removed from plan: {', '.join(sorted(set(removed_from_plan)))}.")
        remaining_plan = (session.shopping_plan or {}).get("plan") or {}
        if remaining_plan:
            parts.append(
                f"New total: ${session.shopping_plan.get('total_cost', 0):.2f} "
                f"across {len(remaining_plan)} store(s)."
            )

    # ── Auto-rescue when the list is now empty ─────────────────
    # Previously we'd leave the session in CONFIRM/EXECUTE with an empty
    # `raw_items`, which caused the NEXT message to be treated as a
    # refinement and silently hit `session_execute` on an empty plan.
    # Transition back to CLARIFY so the next message is parsed as a new
    # list instead.
    plan_empty = not ((session.shopping_plan or {}).get("plan"))
    if not session.raw_items and plan_empty:
        session.state = "CLARIFY"
        session.clarification_done = False
        session.shopping_plan = None
        parts.append(
            "Your list is empty now — let me know what you'd like to add, "
            "or we can call it a day."
        )
    else:
        parts.append("Say the word if you want me to adjust anything else or start fresh.")

    return " ".join(parts)


def handle_justify_request(session: "ShoppingSession", target: str) -> str:
    """Explain why a SKU ended up in the plan. Best-effort: find plan
    entries whose SKU name matches `target`, then trace each back to the
    most-likely raw_items source by token overlap.

    This is a question, NOT a change — we don't touch session state."""
    target_tokens = _tokens(target)
    if not target_tokens:
        return (
            "Happy to explain — which item? "
            "Try \"why did you pick the ginger ale\" or similar."
        )

    plan = (session.shopping_plan or {}).get("plan") or {}
    if not plan:
        return (
            "There's no active plan right now, so nothing to justify. "
            "Want me to build one?"
        )

    hits: list[tuple[str, dict]] = []
    for sid, entries in plan.items():
        for entry in entries:
            if _item_matches_target(entry.get("item", ""), target_tokens):
                hits.append((sid, entry))

    if not hits:
        return (
            f"I don't see anything matching \"{target}\" in the current plan. "
            "It may have already been removed."
        )

    lines: list[str] = []
    for sid, entry in hits[:3]:
        sku = entry.get("item", "(unknown SKU)")
        price = entry.get("price")
        store = entry.get("store_display") or sid
        url = entry.get("url")

        # Best-effort backlink: which raw ingredient probably produced this SKU?
        best_source = None
        best_overlap = 0
        for raw in session.raw_items or []:
            raw_tokens = set(_tokens(raw.get("name", "")))
            sku_tokens = set(_tokens(sku))
            overlap = len(raw_tokens & sku_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_source = raw.get("name")

        source_note = (
            f" — picked for your \"{best_source}\" request"
            if best_source and best_overlap > 0 else ""
        )
        price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "(price n/a)"
        url_str = f" — {url}" if url else ""
        lines.append(
            f"• **{sku}** at {store} ({price_str}){source_note}{url_str}"
        )

    body = "\n".join(lines)
    footer = (
        "\n\nThese were the cheapest cache matches for the ingredient queries. "
        f"Say \"remove {target}\" to drop them, or \"list options for {target}\" "
        "to see alternatives."
    )
    return f"Here's why those are in your plan:\n{body}{footer}"


# --------------- LLM intent fallback (opt-in) ---------------

# If the classifier is less confident than this, we bail out and let
# the deterministic state machine handle the message instead of
# committing to an LLM-picked branch.
#
# Note: post-plan (EXECUTE/DONE) state the state machine is basically a
# reset-everything fallback, so we route on weaker signals there.
_LLM_ROUTE_MIN_CONFIDENCE = 0.55
_LLM_ROUTE_MIN_CONFIDENCE_POST_PLAN = 0.35


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

    # Post-plan states use a looser threshold: the state-machine fallback
    # there is "reset everything", which is almost never what the user
    # actually wants for a low-confidence message.
    post_plan = session.state in ("EXECUTE", "DONE") or bool(
        session.shopping_plan and session.shopping_plan.get("plan")
    )
    min_conf = _LLM_ROUTE_MIN_CONFIDENCE_POST_PLAN if post_plan else _LLM_ROUTE_MIN_CONFIDENCE

    if label == "passthrough" or conf < min_conf:
        return None

    if label == "remove_item":
        target = (result.get("target") or "").strip()
        # Fall back to the raw message if the classifier didn't isolate a
        # target noun — the handler will re-check and ask for clarification.
        return handle_remove_request(session, target or user_message)

    if label == "justify":
        target = (result.get("target") or "").strip()
        return handle_justify_request(session, target or user_message)

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

    # Side-flow: dish-confirm — only fires when a dish proposal is
    # awaiting the user's yes/no/numbers reply. This must run BEFORE
    # the pick-N handler so "yes / no / 1 3 5" aren't mis-routed there.
    if session.pending_dish is not None:
        reply = handle_dish_confirm(session, user_message)
        session.add_message("agent", reply)
        return reply

    # Side-flow: "pick 3" / "I prefer 2" / "option 4" — only fires when
    # we have list_options results staged. Resolves directly to that SKU
    # without going through optimize_shopping_list (which would fuzzy-
    # match the item name and could drift, e.g. "lamb" → "Lamb Weston fries").
    pick_n = detect_pick_intent(user_message, session)
    if pick_n is not None:
        reply = handle_pick_request(session, pick_n)
        session.add_message("agent", reply)
        return reply

    # Side-flow: "remove X" / "I don't want the X" / "drop the X" — prune
    # matching items from raw_items AND the active plan without resetting
    # the session. Runs BEFORE the LLM router so we don't depend on a 1B
    # classifier to catch this common case. Only fires when there IS
    # something to remove (see detect_remove_intent guards).
    remove_target = detect_remove_intent(user_message, session)
    if remove_target:
        reply = handle_remove_request(session, remove_target)
        session.add_message("agent", reply)
        return reply

    # Side-flow: "why did you pick X" / "why is X in there" — explain
    # SKU choices without touching state. Requires an active plan.
    justify_target = detect_justify_intent(user_message, session)
    if justify_target:
        reply = handle_justify_request(session, justify_target)
        session.add_message("agent", reply)
        return reply

    # Side-flow: "recommend X" / "best X" — answer directly without
    # disturbing the active shopping-list state machine.
    intent = detect_recommend_intent(user_message)
    if intent:
        reply = handle_recommend_request(intent)
        session.add_message("agent", reply)
        return reply

    # Side-flow: dish intent — "I want to make carbonara" / "recipe for X"
    # → resolve dish → stash proposal on session.pending_dish → ask the
    # user to confirm. The NEXT turn is caught by the dish-confirm branch
    # above.
    dish_name = detect_dish_intent(user_message)
    if dish_name:
        reply = handle_dish_request(session, dish_name)
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