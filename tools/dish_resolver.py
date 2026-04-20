# ============================================================
# tools/dish_resolver.py
# Resolve a dish name (e.g. "spaghetti carbonara", "mapo tofu")
# into a structured ingredient list usable by the shopping-list
# side of the agent.
#
# Resolution order:
#   1. data/dishes.json                — hand-curated seed set.
#   2. data/dishes_cache.json          — previously LLM-resolved answers.
#   3. LLM (when USE_LLM_DISH_FALLBACK=true) — freeform dish → ingredients;
#      cached back to dishes_cache.json on success.
#
# Public surface:
#   resolve_dish(name)          -> dict | None
#   ingredients_to_raw_items(ings, include_pantry=False) -> list[dict]
# ============================================================

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED_PATH = DATA_DIR / "dishes.json"
CACHE_PATH = DATA_DIR / "dishes_cache.json"


# --------------- normalization --------------------------------------------

_NORM_RE = re.compile(r"[^a-z0-9\s]+")


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation/diacritics, collapse whitespace.
    `Spaghetti Carbonara!` -> `spaghetti carbonara`."""
    if not name:
        return ""
    lowered = name.strip().lower()
    cleaned = _NORM_RE.sub(" ", lowered)
    return " ".join(cleaned.split())


# --------------- seed + cache loading --------------------------------------

_SEED_CACHE: dict | None = None


def _load_seed() -> dict:
    global _SEED_CACHE
    if _SEED_CACHE is not None:
        return _SEED_CACHE
    try:
        with open(SEED_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"dishes": {}}
    _SEED_CACHE = data
    return data


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {"dishes": {}}
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"dishes": {}}


def _write_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# --------------- lookups ---------------------------------------------------

def _lookup_in(store: dict, norm: str) -> dict | None:
    """Exact-key / alias match in a dict whose `dishes` keys are already
    normalized. Returns the canonical dish entry with `_canonical_name`."""
    dishes = store.get("dishes", {}) or {}
    if norm in dishes:
        entry = dict(dishes[norm])
        entry["_canonical_name"] = norm
        return entry
    for canonical, entry in dishes.items():
        aliases = entry.get("aliases") or []
        norm_aliases = [_normalize(a) for a in aliases]
        if norm in norm_aliases:
            out = dict(entry)
            out["_canonical_name"] = canonical
            return out
    return None


def _fuzzy_seed_match(norm: str) -> dict | None:
    """Loose substring match — for when the user types 'carbonara' and the
    seed key is 'spaghetti carbonara'. We pick the seed entry whose
    canonical name or any alias is a superstring / substring of `norm`.
    Returns the longest-matching entry to prefer 'spaghetti carbonara'
    over a hypothetical 'carbonara salad'."""
    seed = _load_seed()
    dishes = seed.get("dishes", {}) or {}
    candidates: list[tuple[int, str, dict]] = []
    for canonical, entry in dishes.items():
        keys = [canonical] + [_normalize(a) for a in (entry.get("aliases") or [])]
        for k in keys:
            if not k:
                continue
            if k in norm or norm in k:
                candidates.append((len(k), canonical, entry))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[0])
    _, canonical, entry = candidates[0]
    out = dict(entry)
    out["_canonical_name"] = canonical
    return out


# --------------- LLM fallback ----------------------------------------------

_DISH_SYSTEM = (
    "You are a culinary assistant. The user names a dish; you return a "
    "minimal shopping ingredient list for cooking it at home. Favor simple, "
    "widely-available grocery store items. Mark true pantry staples (salt, "
    "pepper, common oils, flour, sugar, basic spices) with \"pantry\": true. "
    "Return STRICT JSON only — no prose, no markdown."
)

_DISH_TEMPLATE = """Dish: "{name}"

Return strict JSON of shape:
{{
  "dish": "<canonical english name>",
  "cuisine": "<cuisine family, lowercase>",
  "servings": <int, default 2>,
  "ingredients": [
    {{"name": "<ingredient>", "quantity": <number or null>, "unit": "<unit or null>", "pantry": <true|false>}},
    ...
  ]
}}

Rules:
- 4 to 10 ingredients. Omit garnishes.
- Use store-shelf names (e.g. "canned tomato sauce", not "tomato puree").
- Do NOT invent branded SKUs. These go through a grocery price search.
- Units should be one of: g, ml, lb, cup, tbsp, tsp, clove, slice, head,
  bunch, bottle, jar, can, bag, packet, or null.
"""


def _call_llm_for_dish(name: str) -> dict | None:
    """Ask the main LLM for an ingredient list. Returns parsed dict or
    None on any failure. Never raises to the caller."""
    try:
        from tools.recommender import _call_llm  # reuse provider plumbing
    except Exception:
        return None
    user = _DISH_TEMPLATE.format(name=name.replace('"', "'"))
    try:
        raw = _call_llm(_DISH_SYSTEM, user)
    except Exception:
        return None
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]+\}", cleaned)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _sanitize_llm_entry(entry: dict) -> dict | None:
    """Coerce the LLM's dict into the same shape as seed entries.
    Drops bad rows instead of raising."""
    if not isinstance(entry, dict):
        return None
    raw_ings = entry.get("ingredients") or []
    if not isinstance(raw_ings, list) or not raw_ings:
        return None

    clean_ings: list[dict] = []
    for row in raw_ings:
        if not isinstance(row, dict):
            continue
        nm = str(row.get("name") or "").strip()
        if not nm:
            continue
        qty = row.get("quantity")
        if isinstance(qty, str):
            try:
                qty = float(qty)
            except ValueError:
                qty = None
        if qty is not None and not isinstance(qty, (int, float)):
            qty = None
        unit = row.get("unit")
        if unit is not None:
            unit = str(unit).strip() or None
        pantry = bool(row.get("pantry", False))
        clean_ings.append({
            "name": nm.lower(),
            "quantity": qty,
            "unit": unit,
            "pantry": pantry,
        })
    if not clean_ings:
        return None

    servings = entry.get("servings") or 2
    try:
        servings = int(servings)
    except (TypeError, ValueError):
        servings = 2

    return {
        "cuisine": str(entry.get("cuisine") or "").lower() or "other",
        "servings": servings,
        "aliases": [],
        "ingredients": clean_ings,
    }


# --------------- public API ------------------------------------------------

def resolve_dish(name: str) -> dict | None:
    """Resolve `name` into a dish entry:
      {
        "name": <normalized user-facing name>,
        "cuisine": str,
        "servings": int,
        "ingredients": [{name, quantity, unit, pantry}, ...],
        "source": "seed" | "cache" | "llm",
      }
    Returns None when nothing matches and the LLM path is disabled or
    failed.
    """
    norm = _normalize(name)
    if not norm:
        return None

    # 1. seed — exact / alias
    seed = _load_seed()
    hit = _lookup_in(seed, norm)
    if hit is None:
        hit = _fuzzy_seed_match(norm)
    if hit is not None:
        return {
            "name": hit.get("_canonical_name") or norm,
            "cuisine": hit.get("cuisine", "other"),
            "servings": hit.get("servings", 2),
            "ingredients": hit.get("ingredients", []),
            "source": "seed",
        }

    # 2. persistent LLM cache
    cache = _load_cache()
    cache_hit = _lookup_in(cache, norm)
    if cache_hit is not None:
        return {
            "name": cache_hit.get("_canonical_name") or norm,
            "cuisine": cache_hit.get("cuisine", "other"),
            "servings": cache_hit.get("servings", 2),
            "ingredients": cache_hit.get("ingredients", []),
            "source": "cache",
        }

    # 3. LLM fallback (opt-in)
    try:
        from config.settings import USE_LLM_DISH_FALLBACK
    except Exception:
        USE_LLM_DISH_FALLBACK = False
    if not USE_LLM_DISH_FALLBACK:
        return None

    raw = _call_llm_for_dish(norm)
    if raw is None:
        return None
    clean = _sanitize_llm_entry(raw)
    if clean is None:
        return None

    # persist by normalized key; `dish` echoed by LLM is kept as alias.
    cache.setdefault("dishes", {})[norm] = clean
    try:
        _write_cache(cache)
    except Exception:
        pass

    return {
        "name": norm,
        "cuisine": clean["cuisine"],
        "servings": clean["servings"],
        "ingredients": clean["ingredients"],
        "source": "llm",
    }


def ingredients_to_raw_items(
    ingredients: Iterable[dict],
    *,
    include_pantry: bool = False,
) -> list[dict]:
    """Project a dish's ingredient list into the agent's `raw_items`
    shape so they can be spliced directly into a ShoppingSession.

    Skips `pantry: true` rows by default (salt, pepper, flour, etc.)
    — users typically don't want those on their shopping list.
    """
    out: list[dict] = []
    for ing in ingredients or []:
        if not isinstance(ing, dict):
            continue
        if not include_pantry and ing.get("pantry"):
            continue
        nm = (ing.get("name") or "").strip()
        if not nm:
            continue
        out.append({
            "name": nm,
            "quantity": ing.get("quantity"),
            "unit": ing.get("unit"),
            "ambiguous": False,
        })
    return out


def list_seed_dishes() -> list[str]:
    """Return the canonical names of all built-in dishes."""
    return sorted((_load_seed().get("dishes") or {}).keys())
