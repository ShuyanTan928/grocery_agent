"""
Tool registry for the LLM-driven agent loop.

Each tool is a pure-ish function `(state, args) -> observation`. The
observation is a small JSON-serializable dict that gets fed back into
the orchestrator LLM so it can decide what to do next.

Tools that are meant to mutate state update `state` in place and return
a compact summary of what changed. Read-only tools don't touch state.

The special `reply` tool raises `ReplySignal` to terminate the loop —
that's the only way for a turn to end normally.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from agent.state import AgentState
from tools.dish_resolver import ingredients_to_raw_items, resolve_dish

# Strip a trailing "<qty?> <unit>" suffix off a query-string so we can
# compare against raw item names. Covers the common units that
# items_to_query_strings emits.
_SOURCE_QTY_TAIL_RE = re.compile(
    r"\s+(?:\d+(?:\.\d+)?\s+)?"
    r"(?:lb|lbs|pound|pounds|oz|ounce|ounces|fl\s*oz|gal|gallon|gallons|"
    r"l|ml|liter|litre|liters|litres|pt|quart|quarts|"
    r"dozen|loaf|loaves|pack|bag|bags|bottle|bottles|can|cans|"
    r"jar|jars|box|boxes|carton|cartons|count|ct|each|piece|pieces)\b.*$",
    re.I,
)


def _strip_qty_from_source(src: str) -> str:
    if not src:
        return ""
    return _SOURCE_QTY_TAIL_RE.sub("", src).strip()
from tools.list_ops import (
    apply_avoid_stores,
    apply_preferred_stores,
    clean_remove_target,
    item_matches_target,
    items_to_query_strings,
    remove_match_level,
    tokens,
)
from tools.price_optimizer import (
    find_at_store_in_cache,
    load_stores,
    optimize_shopping_list,
)
from tools.product_search import search_products_ranked
from tools.recommender import recommend_for_query
from tools.route_planner import plan_route
from tools.errand_runner import generate_errand_quote
from tools.geocode import geocode
from tools.promos import get_daily_promos


# ────────────────────────── signals ──────────────────────────────────


class ReplySignal(Exception):
    """Raised by the `reply` tool to end a turn. Caught by the loop."""

    def __init__(self, text: str):
        super().__init__(text)
        self.text = text


class ToolError(Exception):
    """Raised when a tool can't run with the given args. Caught by
    run_tool and surfaced as `{"error": "..."}` in the observation."""


# ────────────────────────── helpers ──────────────────────────────────


def _require_str(args: dict, key: str) -> str:
    val = args.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ToolError(f"missing or empty string arg '{key}'")
    return val.strip()


def _require_int(args: dict, key: str) -> int:
    val = args.get(key)
    if isinstance(val, bool):
        raise ToolError(f"arg '{key}' must be an int, not bool")
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.strip().lstrip("-").isdigit():
        return int(val.strip())
    raise ToolError(f"arg '{key}' must be an int")


def _dedup_by_name(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        nm = (it.get("name") or "").strip().lower()
        if not nm or nm in seen:
            continue
        seen.add(nm)
        out.append(it)
    return out


def _normalize_item_dict(d: Any) -> dict | None:
    """Accept either a string or a dict; coerce to the canonical
    `{name, quantity, unit, ambiguous}` shape the agent works with."""
    if isinstance(d, str):
        nm = d.strip()
        if not nm:
            return None
        return {"name": nm, "quantity": None, "unit": None, "ambiguous": False}
    if not isinstance(d, dict):
        return None
    nm = (d.get("name") or "").strip()
    if not nm:
        return None
    q = d.get("quantity")
    if isinstance(q, str):
        try:
            q = float(q) if "." in q else int(q)
        except ValueError:
            q = None
    unit = d.get("unit")
    if unit is not None and not isinstance(unit, str):
        unit = None
    return {
        "name": nm,
        "quantity": q,
        "unit": unit,
        "ambiguous": bool(d.get("ambiguous", False)),
    }


def _short_store_name(state_stores: dict, sid: str) -> str:
    meta = state_stores.get(sid) or {}
    return meta.get("name") or sid


# ────────────────────────── tool implementations ─────────────────────


# ---------- list mutation ----------

def tool_add_items(state: AgentState, args: dict) -> dict:
    raw = args.get("items")
    if not isinstance(raw, list) or not raw:
        raise ToolError("arg 'items' must be a non-empty list")
    new_items: list[dict] = []
    for entry in raw:
        norm = _normalize_item_dict(entry)
        if norm:
            new_items.append(norm)
    if not new_items:
        raise ToolError("no valid items extracted from 'items'")

    existing = {(it.get("name") or "").strip().lower() for it in state.raw_items}
    added: list[str] = []
    for it in new_items:
        key = (it.get("name") or "").strip().lower()
        if key and key not in existing:
            state.raw_items.append(it)
            existing.add(key)
            added.append(it["name"])
    return {
        "added": added,
        "skipped_duplicates": [it["name"] for it in new_items if it["name"] not in added],
        "raw_items_count": len(state.raw_items),
    }


def tool_parse_items(state: AgentState, args: dict) -> dict:
    """Fallback item parser: use the main LLM to extract structured items
    from a phrase, then splice them via add_items. Use this only when
    you can't build the list yourself from the user's wording."""
    phrase = _require_str(args, "phrase")
    # Lazy import to avoid cycles.
    from agent.agent import parse_items_from_message

    parsed = parse_items_from_message(phrase)
    if not parsed:
        return {"added": [], "note": "parser returned no items"}
    return tool_add_items(state, {"items": parsed})


def tool_remove_items(state: AgentState, args: dict) -> dict:
    """Drop items matching `target` from both raw_items and shopping_plan.
    Uses the same two-pass (exact-first, then loose) matching as the
    legacy handle_remove_request."""
    target = clean_remove_target(_require_str(args, "target"))
    if not target:
        raise ToolError("target reduced to empty after cleaning")
    target_norm = target.strip().lower()
    target_tokens = tokens(target)
    if not target_tokens:
        raise ToolError("target has no searchable tokens")

    # Pass 1: score raw_items.
    raw_scored: list[tuple[int, dict, int]] = []
    for idx, it in enumerate(state.raw_items):
        lvl = remove_match_level(it.get("name", ""), target_norm, target_tokens)
        if lvl > 0:
            raw_scored.append((idx, it, lvl))

    # Prefer exact matches; disambiguate on loose multi-match.
    if any(lvl == 2 for _, _, lvl in raw_scored):
        raw_scored = [h for h in raw_scored if h[2] == 2]
    else:
        distinct = {it.get("name", "").strip().lower() for _, it, _ in raw_scored}
        if len(distinct) > 1:
            return {
                "ambiguous": True,
                "matches": sorted({it.get("name", "") for _, it, _ in raw_scored}),
                "hint": "target matches multiple distinct items; ask the user which one to drop",
            }

    plan = (state.shopping_plan or {}).get("plan") or {}
    source_keys_to_drop = {
        (it.get("name") or "").strip().lower() for _, it, _ in raw_scored
    }
    plan_scored: list[tuple[str, dict, int]] = []
    for sid, entries in plan.items():
        for entry in entries:
            src = (entry.get("source_item") or "").strip().lower()
            if src:
                # source_item is usually the QUERY string
                # ("water 1 bottle"), not just the raw name ("water").
                # Compare after trimming any trailing "[<qty>] <unit>" tail
                # so both shapes hit. But NEVER match on a mere prefix —
                # "orange" must not drop "orange juice".
                #
                # source_item is AUTHORITATIVE when present: we do NOT
                # fall back to SKU-name token matching, because that
                # would re-introduce the "orange" -> "Simply Orange
                # Juice" false positive.
                src_clean = _strip_qty_from_source(src)
                if src in source_keys_to_drop or src_clean in source_keys_to_drop:
                    plan_scored.append((sid, entry, 2))
                continue
            lvl = remove_match_level(entry.get("item", ""), target_norm, target_tokens)
            if lvl > 0:
                plan_scored.append((sid, entry, lvl))

    removed_from_items: list[str] = []
    if raw_scored:
        drop_idx = {idx for idx, _, _ in raw_scored}
        kept: list[dict] = []
        for i, it in enumerate(state.raw_items):
            if i in drop_idx:
                removed_from_items.append(it.get("name", ""))
            else:
                kept.append(it)
        state.raw_items = kept

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
        if state.shopping_plan is not None:
            state.shopping_plan["plan"] = new_plan
            state.shopping_plan["total_cost"] = round(new_total, 2)
            state.shopping_plan["store_ids"] = list(new_plan.keys())
            state.shopping_plan["stores_meta"] = {
                sid: meta
                for sid, meta in (state.shopping_plan.get("stores_meta") or {}).items()
                if sid in new_plan
            }

    plan_empty = not ((state.shopping_plan or {}).get("plan"))
    if plan_empty and state.shopping_plan is not None:
        # Prune the now-empty plan so downstream has_plan checks stay clean.
        state.shopping_plan = None
        state.route_plan = None
        state.errand_quote = None

    if not removed_from_items and not removed_from_plan:
        return {"removed_items": [], "removed_plan_entries": [], "note": "nothing matched"}

    return {
        "removed_items": removed_from_items,
        "removed_plan_entries": removed_from_plan,
        "raw_items_count": len(state.raw_items),
        "plan_empty": plan_empty,
        "new_total": (state.shopping_plan or {}).get("total_cost") if state.shopping_plan else None,
    }


def tool_update_quantity(state: AgentState, args: dict) -> dict:
    name = _require_str(args, "name").lower()
    qty = args.get("quantity")
    unit = args.get("unit")
    if qty is None and unit is None:
        raise ToolError("provide at least one of 'quantity' or 'unit'")
    hits = 0
    for it in state.raw_items:
        if (it.get("name") or "").lower() == name:
            if qty is not None:
                it["quantity"] = qty
            if unit is not None:
                it["unit"] = unit
            it["ambiguous"] = False
            hits += 1
    if hits == 0:
        return {"updated": 0, "note": f"no item named '{name}' on list"}
    return {"updated": hits, "raw_items_count": len(state.raw_items)}


def tool_clear_list(state: AgentState, args: dict) -> dict:
    state.raw_items = []
    state.shopping_plan = None
    state.route_plan = None
    state.errand_quote = None
    state.last_options = []
    state.pending_dish = None
    state.preferences = {}
    state.preferred_stores = {}
    state.want_errand = False
    state.destinations = []
    return {"cleared": True}


# ---------- preferences ----------

def tool_set_preference(state: AgentState, args: dict) -> dict:
    item = _require_str(args, "item").lower()
    store_id = _require_str(args, "store_id")
    kind = _require_str(args, "kind").lower()
    if kind not in ("avoid", "prefer"):
        raise ToolError("kind must be 'avoid' or 'prefer'")
    stores = load_stores()
    if store_id not in stores:
        return {"ok": False, "error": f"unknown store_id '{store_id}'", "known": sorted(stores)}
    target = state.preferences if kind == "avoid" else state.preferred_stores
    target.setdefault(item, [])
    if store_id not in target[item]:
        target[item].append(store_id)
    return {"ok": True, "kind": kind, "item": item, "store_id": store_id}


def tool_unset_preference(state: AgentState, args: dict) -> dict:
    item = _require_str(args, "item").lower()
    store_id = args.get("store_id")
    removed: dict[str, list[str]] = {"avoid": [], "prefer": []}
    for bucket_name, bucket in (("avoid", state.preferences), ("prefer", state.preferred_stores)):
        if item in bucket:
            if store_id is None:
                removed[bucket_name] = list(bucket.pop(item, []) or [])
            else:
                before = list(bucket[item])
                bucket[item] = [s for s in before if s != store_id]
                removed[bucket_name] = [s for s in before if s == store_id]
                if not bucket[item]:
                    del bucket[item]
    return {"removed": removed}


def tool_set_errand(state: AgentState, args: dict) -> dict:
    state.want_errand = bool(args.get("want_errand", True))
    return {"want_errand": state.want_errand}


# ---------- destinations (non-shopping route waypoints) ----------

def _normalize_label(label: str) -> str:
    return " ".join((label or "").split()).lower()


def tool_add_destination(state: AgentState, args: dict) -> dict:
    """Register a non-shopping stop that must appear on the route
    (e.g. "also swing by CMU on the way home"). If lat/lng are not
    provided, tries to geocode the label/address."""
    label = _require_str(args, "label")
    address = args.get("address")
    lat = args.get("lat")
    lng = args.get("lng")

    if lat is not None and lng is not None:
        try:
            lat_f = float(lat)
            lng_f = float(lng)
        except (TypeError, ValueError) as e:
            raise ToolError(f"lat/lng must be numeric: {e}")
        dest = {
            "label": label,
            "address": (address or label).strip(),
            "lat": lat_f,
            "lng": lng_f,
            "source": "user_coords",
        }
    else:
        query = (address or label).strip()
        hit = geocode(query)
        if hit is None:
            return {
                "ok": False,
                "error": (
                    f"couldn't geocode '{query}'. Try a Pittsburgh landmark, "
                    f"set ORS_API_KEY for street addresses (unset "
                    f"USE_MOCK_GEOCODE), or pass lat/lng."
                ),
                "hint_landmarks": [
                    "cmu", "pitt", "downtown", "oakland", "shadyside",
                    "squirrel hill", "east liberty", "strip district",
                    "south side", "north shore", "airport",
                ],
            }
        dest = {
            "label": label,
            "address": hit.get("address") or query,
            "lat": hit["lat"],
            "lng": hit["lng"],
            "source": hit.get("source", "landmark"),
        }

    key = _normalize_label(label)
    state.destinations = [
        d for d in state.destinations if _normalize_label(d.get("label", "")) != key
    ]
    state.destinations.append(dest)

    # Invalidate the route half of the plan so the LLM re-runs
    # optimize_and_route (or pick_option) to pick up the new waypoint.
    state.route_plan = None
    state.errand_quote = None

    return {
        "ok": True,
        "label": dest["label"],
        "address": dest["address"],
        "lat": dest["lat"],
        "lng": dest["lng"],
        "source": dest["source"],
        "destinations_count": len(state.destinations),
        "note": "call optimize_and_route again (if a plan existed) to re-route through this stop.",
    }


def tool_remove_destination(state: AgentState, args: dict) -> dict:
    label = _require_str(args, "label")
    key = _normalize_label(label)
    before = len(state.destinations)
    dropped = [d for d in state.destinations if _normalize_label(d.get("label", "")) == key]
    state.destinations = [
        d for d in state.destinations if _normalize_label(d.get("label", "")) != key
    ]
    if before and len(state.destinations) != before:
        # Route will need to be recomputed.
        state.route_plan = None
        state.errand_quote = None
    return {
        "removed": len(dropped),
        "removed_labels": [d.get("label") for d in dropped],
        "destinations_count": len(state.destinations),
    }


def tool_set_home(state: AgentState, args: dict) -> dict:
    """Set the user's home address (route anchor). Accepts either an
    explicit lat/lng pair, or a free-form query that we route through
    the same geocoder used for destinations. Invalidates route_plan and
    errand_quote so the next optimize_and_route / pick_option rebuilds
    from the new anchor."""
    query = (
        args.get("query")
        or args.get("address")
        or args.get("label")
        or ""
    )
    query = query.strip() if isinstance(query, str) else ""
    lat = args.get("lat")
    lng = args.get("lng")

    if lat is not None and lng is not None:
        try:
            lat_f = float(lat)
            lng_f = float(lng)
        except (TypeError, ValueError) as e:
            raise ToolError(f"lat/lng must be numeric: {e}")
        address = (args.get("address") or query or "Home").strip()
        home = {
            "label": query or "Home",
            "address": address,
            "lat": lat_f,
            "lng": lng_f,
            "source": "user_coords",
        }
    else:
        if not query:
            raise ToolError(
                "set_home requires either `query`/`address` or a lat+lng pair"
            )
        hit = geocode(query)
        if hit is None:
            return {
                "ok": False,
                "error": (
                    f"couldn't geocode '{query}'. Try a Pittsburgh landmark, "
                    f"set ORS_API_KEY for street addresses (unset "
                    f"USE_MOCK_GEOCODE), or pass lat/lng."
                ),
                "hint_landmarks": [
                    "oakland", "shadyside", "squirrel hill", "east liberty",
                    "strip district", "downtown", "south side", "north shore",
                    "lawrenceville", "bloomfield", "highland park",
                    "cmu", "pitt",
                ],
            }
        home = {
            "label": query,
            "address": hit.get("address") or query,
            "lat": hit["lat"],
            "lng": hit["lng"],
            "source": hit.get("source", "landmark"),
        }

    state.home = home
    state.route_plan = None
    state.errand_quote = None
    return {
        "ok": True,
        "home": home,
        "note": (
            "call optimize_and_route (or pick_option) again to rebuild "
            "the route from this home."
        ),
    }


def tool_clear_home(state: AgentState, args: dict) -> dict:
    had = state.home is not None
    state.home = None
    if had:
        state.route_plan = None
        state.errand_quote = None
    return {"cleared": had}


def tool_clear_destinations(state: AgentState, args: dict) -> dict:
    n = len(state.destinations)
    state.destinations = []
    if n:
        state.route_plan = None
        state.errand_quote = None
    return {"cleared": n}


# ---------- read-only search / recommend ----------

def tool_search_products(state: AgentState, args: dict) -> dict:
    query = _require_str(args, "query")
    topk = int(args.get("topk") or 5)
    hits = search_products_ranked(query, include_mock=False, limit=topk * 2)
    hits = [h for h in hits if h.get("item_price") is not None][:topk]
    return {
        "query": query,
        "results": [
            {
                "item_name": h.get("item_name"),
                "item_price": h.get("item_price"),
                "store_id": h.get("store_id"),
                "url": h.get("url"),
            }
            for h in hits
        ],
    }


def tool_recommend_products(state: AgentState, args: dict) -> dict:
    query = _require_str(args, "query")
    topk = int(args.get("topk") or 3)
    preferences = args.get("preferences") or []
    result = recommend_for_query(query, topk=topk, preferences=preferences or None)
    picks = [
        {
            "rank": p.get("rank"),
            "name": (p.get("candidate") or {}).get("name"),
            "price": (p.get("candidate") or {}).get("price"),
            "store": (p.get("candidate") or {}).get("store"),
            "store_id": (p.get("candidate") or {}).get("store_id"),
            "url": (p.get("candidate") or {}).get("url"),
            "reason": p.get("reason"),
        }
        for p in (result.get("picks") or [])
    ]
    return {"query": query, "picks": picks, "summary": result.get("summary", "")}


def tool_find_at_store(state: AgentState, args: dict) -> dict:
    item = _require_str(args, "item")
    store_id = _require_str(args, "store_id")
    stores = load_stores()
    hit = find_at_store_in_cache(item, store_id, stores)
    if hit is None:
        return {"found": False, "item": item, "store_id": store_id}
    return {
        "found": True,
        "item_name": hit.get("item_name"),
        "item_price": hit.get("item_price"),
        "store": hit.get("store"),
        "url": hit.get("url"),
    }


# ---------- list options + pick ----------

def tool_list_options(state: AgentState, args: dict) -> dict:
    query = _require_str(args, "query")
    topk = int(args.get("topk") or 5)
    raw = search_products_ranked(query, include_mock=False, limit=topk * 2)
    raw = [h for h in raw if h.get("item_price") is not None][:topk]

    stores = load_stores()
    state.last_options = [
        {
            "item_name": h.get("item_name", "?"),
            "item_price": h.get("item_price"),
            "store_id": h.get("store_id") or "",
            "store_display": (stores.get(h.get("store_id") or "") or {}).get("name")
                              or h.get("store_id") or "",
            "url": h.get("url"),
        }
        for h in raw
    ]
    return {
        "query": query,
        "count": len(state.last_options),
        "options": [
            {
                "n": i + 1,
                "item_name": o["item_name"],
                "item_price": o["item_price"],
                "store_display": o["store_display"],
                "url": o.get("url"),
            }
            for i, o in enumerate(state.last_options)
        ],
    }


def tool_pick_option(state: AgentState, args: dict) -> dict:
    n = _require_int(args, "n")
    if not state.last_options:
        raise ToolError("no options staged; call list_options first")
    if not (1 <= n <= len(state.last_options)):
        raise ToolError(f"pick out of range: n={n}, available 1..{len(state.last_options)}")
    opt = state.last_options[n - 1]
    stores = load_stores()
    sid = opt.get("store_id") or ""
    store_meta = stores.get(sid, {}) if sid else {}
    try:
        price = float(opt.get("item_price") or 0.0)
    except (TypeError, ValueError):
        price = 0.0
    entry = {
        "item": opt.get("item_name", "?"),
        "price": price,
        "store_display": store_meta.get("name") or opt.get("store_display") or sid,
        "url": opt.get("url"),
        "source": "user_pick",
    }
    plan = {
        "plan": {sid: [entry]} if sid else {},
        "total_cost": round(price, 2),
        "not_found": [],
        "store_ids": [sid] if sid else [],
        "stores_meta": {sid: store_meta} if sid and store_meta else {},
    }
    route = None
    if plan["store_ids"] or state.destinations:
        try:
            route = plan_route(
                store_ids=plan["store_ids"],
                stores_meta=plan["stores_meta"],
                extra_waypoints=state.destinations or None,
                home_override=state.home,
            )
        except Exception:
            route = None
    state.shopping_plan = plan
    state.route_plan = route
    state.last_options = []
    return {
        "picked": n,
        "item_name": entry["item"],
        "price": price,
        "store_display": entry["store_display"],
        "url": entry.get("url"),
        "store_id": sid,
    }


# ---------- dish flow ----------

def tool_lookup_dish(state: AgentState, args: dict) -> dict:
    name = _require_str(args, "name")
    dish = resolve_dish(name)
    if dish is None or not dish.get("ingredients"):
        return {"found": False, "name": name}
    return {
        "found": True,
        "name": dish.get("name"),
        "cuisine": dish.get("cuisine"),
        "source": dish.get("source"),
        "ingredients": dish.get("ingredients"),
    }


def tool_propose_dish(state: AgentState, args: dict) -> dict:
    name = _require_str(args, "name")
    dish = resolve_dish(name)
    if dish is None or not dish.get("ingredients"):
        state.pending_dish = None
        return {"found": False, "name": name}
    state.pending_dish = dish
    buyable = [i for i in dish["ingredients"] if not i.get("pantry")]
    pantry = [i for i in dish["ingredients"] if i.get("pantry")]
    return {
        "found": True,
        "name": dish["name"],
        "cuisine": dish.get("cuisine"),
        "source": dish.get("source"),
        "buyable_ingredients": [
            {"n": i + 1, "name": ing["name"],
             "quantity": ing.get("quantity"), "unit": ing.get("unit")}
            for i, ing in enumerate(buyable)
        ],
        "pantry_ingredients": [p["name"] for p in pantry],
    }


def tool_apply_pending_dish(state: AgentState, args: dict) -> dict:
    if not state.pending_dish:
        raise ToolError("no pending_dish to apply")
    include_pantry = bool(args.get("include_pantry", False))
    only = args.get("only")
    ings = state.pending_dish.get("ingredients") or []
    buyable = ings if include_pantry else [i for i in ings if not i.get("pantry")]
    if only:
        if not isinstance(only, list) or not all(isinstance(n, int) for n in only):
            raise ToolError("'only' must be a list of 1-indexed ints")
        picked = [buyable[n - 1] for n in only if 1 <= n <= len(buyable)]
        chosen = ingredients_to_raw_items(picked, include_pantry=True)
    else:
        chosen = ingredients_to_raw_items(ings, include_pantry=include_pantry)
    if not chosen:
        state.pending_dish = None
        return {"added": [], "note": "nothing to add"}
    existing = {(it.get("name") or "").strip().lower() for it in state.raw_items}
    added: list[str] = []
    for it in chosen:
        nm = (it.get("name") or "").strip().lower()
        if nm and nm not in existing:
            state.raw_items.append(it)
            existing.add(nm)
            added.append(it["name"])
    dish_name = state.pending_dish.get("name", "")
    state.pending_dish = None
    return {"added": added, "dish_name": dish_name, "raw_items_count": len(state.raw_items)}


def tool_cancel_pending_dish(state: AgentState, args: dict) -> dict:
    name = (state.pending_dish or {}).get("name", "")
    state.pending_dish = None
    return {"cancelled": name}


# ---------- main pipeline ----------

def tool_optimize_and_route(state: AgentState, args: dict) -> dict:
    """Run optimize_shopping_list -> apply_preferred_stores ->
    apply_avoid_stores -> plan_route (-> errand_quote). Writes the
    shopping_plan, route_plan, errand_quote onto state."""
    if not state.raw_items:
        raise ToolError("raw_items is empty; nothing to optimize")

    query_strings = items_to_query_strings(state.raw_items)
    shopping_plan = optimize_shopping_list(query_strings)

    if state.preferred_stores:
        shopping_plan = apply_preferred_stores(shopping_plan, state.preferred_stores)
    if state.preferences:
        shopping_plan = apply_avoid_stores(shopping_plan, state.preferences)

    state.shopping_plan = shopping_plan

    if not shopping_plan.get("plan"):
        state.route_plan = None
        state.errand_quote = None
        return {
            "ok": False,
            "not_found": shopping_plan.get("not_found") or query_strings,
            "reason": "no items could be priced from cached store data",
        }

    try:
        route = plan_route(
            store_ids=shopping_plan["store_ids"],
            stores_meta=shopping_plan["stores_meta"],
            extra_waypoints=state.destinations or None,
            home_override=state.home,
        )
    except Exception as e:
        route = None
        route_error: str | None = str(e)
    else:
        route_error = None
    state.route_plan = route

    want_errand = bool(args.get("want_errand", state.want_errand))
    state.want_errand = want_errand
    if want_errand:
        try:
            state.errand_quote = generate_errand_quote(shopping_plan, route)
        except Exception:
            state.errand_quote = None
    else:
        state.errand_quote = None

    stores = load_stores()
    per_store = {
        sid: [
            {"item": e.get("item"), "price": e.get("price"),
             "source_item": e.get("source_item"), "url": e.get("url")}
            for e in entries
        ]
        for sid, entries in shopping_plan["plan"].items()
    }
    return {
        "ok": True,
        "total_cost": shopping_plan.get("total_cost"),
        "store_count": len(shopping_plan["plan"]),
        "per_store": per_store,
        "stores_meta": {
            sid: {"name": _short_store_name(stores, sid),
                  "address": (stores.get(sid) or {}).get("address")}
            for sid in shopping_plan["store_ids"]
        },
        "route": route,
        "route_error": route_error,
        "not_found": shopping_plan.get("not_found") or [],
        "unfulfilled_preferences": shopping_plan.get("unfulfilled_preferences") or [],
        "errand_quote": state.errand_quote,
    }


# ---------- justify ----------

def tool_justify_pick(state: AgentState, args: dict) -> dict:
    target = _require_str(args, "target")
    target_tokens = tokens(target)
    if not target_tokens:
        raise ToolError("target has no searchable tokens")
    plan = (state.shopping_plan or {}).get("plan") or {}
    if not plan:
        return {"hits": [], "note": "no active plan"}
    hits = []
    for sid, entries in plan.items():
        for entry in entries:
            if item_matches_target(entry.get("item", ""), target_tokens):
                best_source, best_overlap = None, 0
                sku_tokens = set(tokens(entry.get("item", "")))
                for raw in state.raw_items:
                    raw_tokens = set(tokens(raw.get("name", "")))
                    overlap = len(raw_tokens & sku_tokens)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_source = raw.get("name")
                hits.append({
                    "store_id": sid,
                    "item": entry.get("item"),
                    "price": entry.get("price"),
                    "source_item": entry.get("source_item") or best_source,
                    "url": entry.get("url"),
                })
    return {"hits": hits[:5], "target": target}


# ---------- promos (read-only) ----------

def tool_get_daily_promos(state: AgentState, args: dict) -> dict:
    """Return today's cached promo digest. Read-only."""
    topk = args.get("topk_per_store", 5)
    try:
        topk = int(topk)
    except (TypeError, ValueError):
        topk = 5
    topk = max(1, min(20, topk))

    stores_arg = args.get("stores")
    stores: list[str] | None = None
    if isinstance(stores_arg, list):
        stores = [str(s) for s in stores_arg if isinstance(s, (str, int))]
        if not stores:
            stores = None

    min_pct = args.get("min_discount_pct")
    try:
        min_pct = float(min_pct) if min_pct is not None else None
    except (TypeError, ValueError):
        min_pct = None

    return get_daily_promos(
        topk_per_store=topk,
        stores=stores,
        min_discount_pct=min_pct,
    )


# ---------- terminator ----------

def tool_reply(state: AgentState, args: dict) -> dict:
    text = _require_str(args, "text")
    raise ReplySignal(text)


# ────────────────────────── registry ─────────────────────────────────


TOOLS: dict[str, dict] = {
    # mutation
    "add_items": {
        "fn": tool_add_items,
        "description": (
            "Append items to the shopping list. Use this when the user names "
            "groceries. Dedupes by lowercase name. Args: items = list of "
            "{name:str, quantity:int|float|null, unit:str|null, ambiguous:bool}. "
            "Strings are accepted and coerced into {name, ambiguous=false}."
        ),
        "args": {"items": "list[dict|str]"},
    },
    "parse_items": {
        "fn": tool_parse_items,
        "description": (
            "Fallback: call the legacy LLM parser on a natural-language phrase "
            "and append the result. Prefer add_items when you can structure the "
            "items yourself."
        ),
        "args": {"phrase": "str"},
    },
    "remove_items": {
        "fn": tool_remove_items,
        "description": (
            "Drop items matching `target` from raw_items AND any existing plan. "
            "Exact name matches win over loose token matches. Returns "
            "{ambiguous:true, matches:[...]} when the target is ambiguous; in "
            "that case reply and ask the user to pick one before retrying."
        ),
        "args": {"target": "str"},
    },
    "update_quantity": {
        "fn": tool_update_quantity,
        "description": (
            "Set quantity and/or unit on a raw_item by exact name match "
            "(case-insensitive). Also clears ambiguous=false on that row."
        ),
        "args": {"name": "str", "quantity": "int|float|null (optional)", "unit": "str|null (optional)"},
    },
    "clear_list": {
        "fn": tool_clear_list,
        "description": "Wipe everything: raw_items, plan, dish proposal, options, preferences.",
        "args": {},
    },
    # preferences
    "set_preference": {
        "fn": tool_set_preference,
        "description": (
            "Constrain where an item can/can't be bought. kind='avoid' to "
            "blacklist a store for that item, kind='prefer' to whitelist it. "
            "Applied during optimize_and_route."
        ),
        "args": {"item": "str", "store_id": "str (e.g. trader_joes_shadyside)", "kind": "'avoid'|'prefer'"},
    },
    "unset_preference": {
        "fn": tool_unset_preference,
        "description": (
            "Remove a previously-set preference. Omit store_id to wipe all "
            "preferences for that item."
        ),
        "args": {"item": "str", "store_id": "str (optional)"},
    },
    "set_errand": {
        "fn": tool_set_errand,
        "description": "Flip the want_errand flag (so optimize_and_route generates a quote).",
        "args": {"want_errand": "bool"},
    },
    # destinations (non-shopping route waypoints)
    "add_destination": {
        "fn": tool_add_destination,
        "description": (
            "Register a non-shopping stop the user wants to include on "
            "the driving route (e.g. 'I also need to swing by CMU'). If "
            "lat/lng are omitted the label/address is geocoded via a "
            "curated Pittsburgh landmark dict (CMU, Pitt, Downtown, "
            "Squirrel Hill, Shadyside, East Liberty, Strip District, "
            "Airport, etc.), then ORS when ORS_API_KEY is set (even if "
            "USE_MOCK_DATA is true for products). On unknown "
            "places returns {ok:false, error:...} — ask the user for an "
            "address or explicit coords. Destinations are mandatory stops "
            "on the next optimize_and_route (or pick_option) call; they "
            "are NOT shopping stores."
        ),
        "args": {
            "label": "str (short display name, e.g. 'CMU')",
            "address": "str (optional; falls back to label)",
            "lat": "float (optional; required with lng to skip geocoding)",
            "lng": "float (optional; required with lat to skip geocoding)",
        },
    },
    "remove_destination": {
        "fn": tool_remove_destination,
        "description": (
            "Drop a previously-added destination by label (case-insensitive "
            "exact match on the normalized label)."
        ),
        "args": {"label": "str"},
    },
    "clear_destinations": {
        "fn": tool_clear_destinations,
        "description": "Drop all non-shopping destinations from state.",
        "args": {},
    },
    # home anchor (route start / end)
    "set_home": {
        "fn": tool_set_home,
        "description": (
            "Set the user's home address — the route anchor plan_route "
            "starts and ends at, and the H marker on the web map. Accepts "
            "a free-form `query` (same landmark dict + ORS + disk cache "
            "as add_destination) OR an explicit `lat`+`lng` pair. Street "
            "addresses need ORS_API_KEY (and USE_MOCK_GEOCODE unset); "
            "otherwise prefer a neighborhood/landmark (oakland, shadyside, "
            "squirrel hill, east liberty, strip district, downtown, cmu, "
            "pitt, ...) or pass lat/lng explicitly. On failure returns "
            "{ok:false, error:...} — ask the user to rephrase. On success "
            "invalidates route_plan + errand_quote, so call "
            "optimize_and_route (or pick_option) again to rebuild the "
            "route from the new home."
        ),
        "args": {
            "query": "str (optional; e.g. 'oakland' or '419 melwood ave')",
            "address": "str (optional; friendlier display text)",
            "lat": "float (optional; required together with lng)",
            "lng": "float (optional; required together with lat)",
        },
    },
    "clear_home": {
        "fn": tool_clear_home,
        "description": (
            "Reset home to the default configured in config/settings.py. "
            "Invalidates the route so it rebuilds from the default anchor."
        ),
        "args": {},
    },
    # read-only search
    "search_products": {
        "fn": tool_search_products,
        "description": (
            "Relevance-ranked search over the per-store cache, topk results. "
            "Read-only. Good for verifying what's available before building a plan."
        ),
        "args": {"query": "str", "topk": "int (default 5)"},
    },
    "recommend_products": {
        "fn": tool_recommend_products,
        "description": (
            "LLM-ranked top picks for `query`, with reasons. Read-only. "
            "Use when the user says 'recommend X' or 'best X'."
        ),
        "args": {"query": "str", "topk": "int (default 3)", "preferences": "list[str] (optional, e.g. ['organic','cheapest'])"},
    },
    "find_at_store": {
        "fn": tool_find_at_store,
        "description": "Cheapest `item` at a specific store_id. Read-only.",
        "args": {"item": "str", "store_id": "str"},
    },
    # list options / pick N
    "list_options": {
        "fn": tool_list_options,
        "description": (
            "Like search_products, but ALSO stages the results on state.last_options "
            "so the user can then 'pick N'. Use this when the user asks 'list the "
            "options' or 'show me alternatives'."
        ),
        "args": {"query": "str", "topk": "int (default 5)"},
    },
    "pick_option": {
        "fn": tool_pick_option,
        "description": (
            "Lock in the Nth option (1-indexed) from the most recent list_options "
            "call. Builds a 1-item shopping_plan directly — skips price-optimize."
        ),
        "args": {"n": "int (1-indexed)"},
    },
    # dish
    "lookup_dish": {
        "fn": tool_lookup_dish,
        "description": "Resolve a dish name to ingredients. Read-only; doesn't touch state.",
        "args": {"name": "str"},
    },
    "propose_dish": {
        "fn": tool_propose_dish,
        "description": (
            "Resolve a dish and stage its ingredients on state.pending_dish. "
            "Also returns the ingredient breakdown so you can render a "
            "confirmation prompt."
        ),
        "args": {"name": "str"},
    },
    "apply_pending_dish": {
        "fn": tool_apply_pending_dish,
        "description": (
            "Splice the pending_dish's non-pantry ingredients into raw_items "
            "(or only a subset via `only=[1,3,5]`). Clears pending_dish."
        ),
        "args": {"include_pantry": "bool (default false)", "only": "list[int] (optional 1-indexed)"},
    },
    "cancel_pending_dish": {
        "fn": tool_cancel_pending_dish,
        "description": "Drop the pending dish proposal without adding any ingredients.",
        "args": {},
    },
    # main pipeline
    "optimize_and_route": {
        "fn": tool_optimize_and_route,
        "description": (
            "THE main pipeline. Runs price optimization + preference application "
            "+ route planning (+ errand quote if want_errand). Writes the plan "
            "onto state. Call this when the user has confirmed their list and "
            "wants a concrete plan. Returns {ok, total_cost, per_store, route, ...}."
        ),
        "args": {"want_errand": "bool (optional; inherits state.want_errand)"},
    },
    # justify
    "justify_pick": {
        "fn": tool_justify_pick,
        "description": (
            "Explain why a SKU is in the active plan by tracing it back to the "
            "raw_items source that produced it. Read-only."
        ),
        "args": {"target": "str"},
    },
    # promos
    "get_daily_promos": {
        "fn": tool_get_daily_promos,
        "description": (
            "Read today's cached promo digest built from data/promos.json "
            "(refreshed by scripts/refresh_promos.py). Returns "
            "{generated_at, total, per_store:{<store_id>:[{item_name, "
            "sale_price, reg_price, discount_pct, reason, url}]}, empty}. "
            "Use this on the FIRST turn of a fresh session (empty "
            "conversation_history AND empty raw_items) to surface top deals "
            "in your greeting, or when the user asks 'any deals?' / "
            "'what's on sale?'. If empty:true, just greet normally without "
            "mentioning promos."
        ),
        "args": {
            "topk_per_store": "int (default 5, max 20)",
            "stores": "list[str] (optional; restrict to specific store_ids)",
            "min_discount_pct": "float (optional; drop rows below this %)",
        },
    },
    # terminator
    "reply": {
        "fn": tool_reply,
        "description": (
            "Produce the final user-facing reply text and end this turn. "
            "Every turn MUST end in exactly one reply call. Craft the text "
            "using the observations you've seen so far — for example, quote "
            "the totals from optimize_and_route's observation when rendering "
            "a shopping plan summary."
        ),
        "args": {"text": "str"},
    },
}


# ────────────────────────── dispatcher ───────────────────────────────


def run_tool(state: AgentState, name: str, args: dict) -> dict:
    """Execute `name` with `args` against `state`. Exceptions that aren't
    ReplySignal are turned into an `{"error": "..."}` observation so the
    loop can show them to the LLM and let it recover."""
    spec = TOOLS.get(name)
    if spec is None:
        return {"error": f"unknown tool '{name}'", "available": sorted(TOOLS)}
    fn: Callable[[AgentState, dict], dict] = spec["fn"]
    try:
        return fn(state, args or {})
    except ReplySignal:
        raise
    except ToolError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def list_tool_specs() -> list[dict]:
    """Auto-generated summary of the tool registry for the LLM prompt."""
    out = []
    for name, spec in TOOLS.items():
        out.append({
            "name": name,
            "description": spec["description"],
            "args": spec.get("args", {}),
        })
    return out
