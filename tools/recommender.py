# ============================================================
# tools/recommender.py
# LLM-backed top-K product recommender over real cached store data.
#
# Pipeline:
#   1. search_products_ranked  (relevance-first tiered + price-asc)
#   2. normalize candidates    (size / brand / unit_price / short store id)
#   3. ask LLM to pick top-K   (with optional user preferences)
#   4. parse strict-JSON       (tolerant of code fences / surrounding prose)
#
# Public surface:
#   recommend_for_query(query, *, topk=3, preferences=None, store_ids=None,
#                       max_candidates=40) -> dict
#       returns: {"query", "topk", "candidates", "picks", "summary",
#                 "timings": {"search_ms", "llm_ms"}, "raw_llm": str}
#
# Used by:
#   - scripts/test_recommend.py  (interactive CLI)
#   - scripts/eval_recommendations.py  (batch eval)
#   - agent/agent.py  (recommend / "what's the best X" intent)
# ============================================================

from __future__ import annotations

import json
import re
import time
from typing import Iterable

from tools.product_search import search_products_ranked


# ---------- candidate normalization ----------------------------------------

_STORE_SHORT = {
    "trader_joes_shadyside": "trader_joes",
    "giant_eagle_squirrel_hill": "giant_eagle",
    "target_east_liberty": "target",
    "aldi_greenfield": "aldi",
}


def _short_store(store_id: str | None) -> str:
    if not store_id:
        return ""
    return _STORE_SHORT.get(store_id, store_id)


def _normalize_size(item: dict) -> str:
    raw = item.get("_raw") or {}
    if raw.get("size"):
        return str(raw["size"])
    if raw.get("sales_size") and raw.get("sales_uom"):
        return f"{raw['sales_size']} {raw['sales_uom']}"
    return ""


def _normalize_unit_price(item: dict) -> str:
    raw = item.get("_raw") or {}
    if raw.get("display_price_per_unit"):
        return str(raw["display_price_per_unit"])
    if raw.get("price_per_unit_string"):
        return str(raw["price_per_unit_string"])
    up = raw.get("unit_price")
    if up is not None:
        try:
            up_str = f"${float(up):.2f}"
        except (TypeError, ValueError):
            up_str = str(up)
        suffix = raw.get("unit_price_suffix") or ""
        return f"{up_str}{suffix}"
    return ""


def _normalize_brand(item: dict) -> str:
    return ((item.get("_raw") or {}).get("brand") or "").strip()


def build_candidates(
    query: str,
    *,
    store_ids: Iterable[str] | None = None,
    max_candidates: int = 40,
) -> list[dict]:
    """Search caches, drop priceless rows, normalize into a compact dict
    list ready for the LLM. Relevance-tiered via search_products_ranked.
    """
    raw = search_products_ranked(
        query,
        store_ids=store_ids,
        include_mock=False,
        expand_synonyms=True,
    )
    candidates: list[dict] = []
    for it in raw:
        price = it.get("item_price")
        if price is None:
            continue
        candidates.append({
            "id": len(candidates) + 1,
            "store": _short_store(it.get("store_id")),
            "name": it.get("item_name") or "",
            "size": _normalize_size(it),
            "brand": _normalize_brand(it),
            "price": float(price),
            "unit_price": _normalize_unit_price(it),
            "_tier": it.get("_relevance_tier", 2),
        })
        if len(candidates) >= max_candidates:
            break
    return candidates


def render_candidate_block(cands: list[dict]) -> str:
    """Compact, model-friendly candidate table."""
    lines = []
    for c in cands:
        size = f" [{c['size']}]" if c["size"] else ""
        brand = f" by {c['brand']}" if c["brand"] else ""
        unit = f"  ({c['unit_price']})" if c["unit_price"] else ""
        lines.append(
            f"  id={c['id']:<3} {c['store']:<12} ${c['price']:>6.2f}  "
            f"{c['name']}{size}{brand}{unit}"
        )
    return "\n".join(lines)


# ---------- preferences ----------------------------------------------------

# Map short flag -> guidance line embedded in the LLM prompt.
_PREF_GUIDANCE = {
    "cheapest":     "User strongly prefers the lowest absolute price (still must be relevant).",
    "organic":      "User wants ORGANIC, grass-fed, or all-natural items if any are present.",
    "largest-pack": "User prefers larger pack sizes / bulk for better $/unit value.",
    "premium":      "User prefers premium / branded / high-quality items, not the cheapest.",
    "store-brand":  "User prefers the store's in-house brand for value.",
}


def _expand_preferences(preferences: list[str] | None) -> str:
    """Render preference flags as a guidance block for the prompt.

    Supports:
      - the keys above (e.g. "organic", "largest-pack")
      - "brand:NAME"  -> "User prefers brand 'NAME' if available."
      - free-form strings are passed through verbatim
    """
    if not preferences:
        return ""
    lines: list[str] = []
    for p in preferences:
        p = (p or "").strip()
        if not p:
            continue
        if p in _PREF_GUIDANCE:
            lines.append(f"- {_PREF_GUIDANCE[p]}")
            continue
        if p.lower().startswith("brand:"):
            brand = p.split(":", 1)[1].strip()
            if brand:
                lines.append(f"- User prefers brand '{brand}' if available.")
                continue
        lines.append(f"- {p}")
    if not lines:
        return ""
    return "User preferences:\n" + "\n".join(lines) + "\n"


# ---------- prompt + LLM call ---------------------------------------------

RECOMMEND_SYSTEM = (
    "You are a grocery shopping assistant. The user gives you a query "
    "and a list of real candidate products from 4 Pittsburgh stores "
    "(Trader Joe's, Giant Eagle, Target, Aldi). Pick the top K best "
    "buys for what the user most likely wants. Reward genuine relevance "
    "to the query (e.g. 'pork' -> real pork meat, NOT pork & beans, "
    "pork rinds, or pork-flavored sauce). Within relevant items, prefer "
    "lower price-per-unit, larger pack value, and reputable in-house "
    "brands — unless the user states a preference that overrides this. "
    "Return STRICT JSON only — no prose, no markdown."
)

RECOMMEND_USER_TEMPLATE = """User query: "{query}"
{preference_block}
Candidates (already filtered to ones that mention the query word and have a price; relevance-sorted then cheapest first):
{candidate_block}

Return strict JSON of shape:
{{
  "picks": [
    {{
      "rank": 1,
      "candidate_id": <int matching id above>,
      "reason": "<one short sentence>"
    }},
    ...up to {topk} items, ordered best -> worst
  ],
  "summary": "<one sentence overall recommendation>"
}}

Rules:
- Use ONLY ids from the candidate list. Do not invent new items.
- If fewer than {topk} candidates are genuinely relevant, return fewer.
- Prefer relevance over absolute cheapness (a $1 can of pork & beans is NOT a good answer to "pork").
- Honor the user preferences above if present."""


def _build_user_prompt(query: str, cands: list[dict], topk: int,
                       preferences: list[str] | None) -> str:
    pref_block = _expand_preferences(preferences)
    return RECOMMEND_USER_TEMPLATE.format(
        query=query,
        preference_block=("\n" + pref_block if pref_block else ""),
        candidate_block=render_candidate_block(cands),
        topk=topk,
    )


def _call_llm(system: str, user: str, *, temperature: float = 0.2) -> str:
    """Provider-agnostic LLM call returning raw text. Reuses settings from
    config.settings (Google GenAI or OpenRouter)."""
    from config.settings import (
        GOOGLE_API_KEY, LLM_MODEL, LLM_PROVIDER,
        OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
        OPENROUTER_HTTP_REFERER, OPENROUTER_APP_TITLE,
    )

    if LLM_PROVIDER == "openrouter":
        from openai import OpenAI
        headers = {}
        if OPENROUTER_HTTP_REFERER:
            headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
        if OPENROUTER_APP_TITLE:
            headers["X-Title"] = OPENROUTER_APP_TITLE
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            default_headers=headers or None,
        )
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    from google import genai
    client = genai.Client(api_key=GOOGLE_API_KEY)
    resp = client.models.generate_content(
        model=LLM_MODEL,
        contents=user,
        config=genai.types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
        ),
    )
    return (resp.text or "").strip()


def parse_recommendation(raw: str) -> dict:
    """Tolerant JSON extraction (strips ```json fences / surrounding prose)."""
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]+\}", cleaned)
    if not m:
        raise ValueError(f"No JSON object in LLM output:\n{raw}")
    return json.loads(m.group(0))


# ---------- public API -----------------------------------------------------

def recommend_for_query(
    query: str,
    *,
    topk: int = 3,
    preferences: list[str] | None = None,
    store_ids: Iterable[str] | None = None,
    max_candidates: int = 40,
) -> dict:
    """Recommend top-K real products for `query` from cached store data.

    Returns a dict suitable for both pretty-print and structured use:

        {
          "query":      str,
          "topk":       int,
          "candidates": list[dict],   # full normalized candidate list
          "picks":      list[dict],   # each: {rank, candidate, reason}
          "summary":    str,
          "timings":    {"search_ms": float, "llm_ms": float},
          "raw_llm":    str,          # for debugging / logging
        }

    On no candidates, returns an empty `picks` list and a fallback summary
    instead of raising — so callers can render "no matches" cleanly.
    """
    t0 = time.perf_counter()
    cands = build_candidates(
        query, store_ids=store_ids, max_candidates=max_candidates,
    )
    search_ms = (time.perf_counter() - t0) * 1000

    if not cands:
        return {
            "query": query,
            "topk": topk,
            "candidates": [],
            "picks": [],
            "summary": f"No candidates found for '{query}' in cached stores.",
            "timings": {"search_ms": search_ms, "llm_ms": 0.0},
            "raw_llm": "",
        }

    user_prompt = _build_user_prompt(query, cands, topk, preferences)
    t1 = time.perf_counter()
    raw = _call_llm(RECOMMEND_SYSTEM, user_prompt)
    llm_ms = (time.perf_counter() - t1) * 1000

    try:
        parsed = parse_recommendation(raw)
    except (ValueError, json.JSONDecodeError):
        return {
            "query": query,
            "topk": topk,
            "candidates": cands,
            "picks": [],
            "summary": "LLM returned malformed JSON.",
            "timings": {"search_ms": search_ms, "llm_ms": llm_ms},
            "raw_llm": raw,
        }

    by_id = {c["id"]: c for c in cands}
    picks: list[dict] = []
    for pick in parsed.get("picks") or []:
        cid = pick.get("candidate_id")
        cand = by_id.get(cid)
        if not cand:
            continue
        picks.append({
            "rank": pick.get("rank", len(picks) + 1),
            "candidate": cand,
            "reason": pick.get("reason", ""),
        })

    return {
        "query": query,
        "topk": topk,
        "candidates": cands,
        "picks": picks[:topk],
        "summary": parsed.get("summary", ""),
        "timings": {"search_ms": search_ms, "llm_ms": llm_ms},
        "raw_llm": raw,
    }


def format_recommendation(result: dict) -> str:
    """Human-friendly multi-line rendering of a recommend_for_query() result.
    Used by the agent for chat replies and by the CLI for pretty-print."""
    lines = []
    picks = result.get("picks") or []
    if not picks:
        return result.get("summary") or "No recommendations available."

    for p in picks:
        c = p["candidate"]
        size = f" [{c['size']}]" if c.get("size") else ""
        brand = f" by {c['brand']}" if c.get("brand") else ""
        unit = f"  ({c['unit_price']})" if c.get("unit_price") else ""
        lines.append(
            f"  #{p['rank']} {c['store']:<12} ${c['price']:>6.2f}  "
            f"{c['name']}{size}{brand}{unit}"
        )
        if p.get("reason"):
            lines.append(f"       → {p['reason']}")

    if result.get("summary"):
        lines.append("")
        lines.append(f"Summary: {result['summary']}")
    return "\n".join(lines)
