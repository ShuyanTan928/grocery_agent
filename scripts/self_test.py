"""Non-interactive self-test harness: drive chat() through many scripted
scenarios with LLM + price-optimizer stubbed, to catch logic holes that
unit tests don't cover (stuck states, silent drops, mis-routed intents).

Run:
    uv run python scripts/self_test.py
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import agent as a
from agent.agent import ShoppingSession, chat


# ───────────────────────────── LLM stub ─────────────────────────────

def fake_call_llm(prompt, *, model=None, temperature=None):
    """Minimal but realistic LLM stub. Dispatch by sniffing known prompt
    templates so each state hits a plausible branch instead of the generic
    "(stubbed)" fallback (which would hide real bugs)."""
    p = prompt or ""

    # --- parse_items_from_message (PARSE_PROMPT) -------------------
    if "Extract" in p and "JSON" in p and p.rstrip().endswith('"'):
        # Pull the user message out of the trailing quoted string.
        m = re.search(r'"([^"]+)"\s*$', p)
        msg = (m.group(1) if m else "").lower()
        items: list[dict] = []
        # Crude noun extractor that's good enough for the harness.
        for word in ("milk", "eggs", "bread", "bananas", "beef", "chicken",
                     "water", "ginger", "orange juice", "lamb", "pasta",
                     "pork", "cheese", "rice"):
            if word in msg:
                items.append({"name": word, "quantity": None, "unit": None, "ambiguous": False})
        return json.dumps({"items": items})

    # --- intent classifier ----------------------------------------
    if "INTENT CLASSIFIER" in p:
        # Pretend the small model is unsure → always passthrough at conf 0.
        # Forces regex side-flows + state machine to do the work.
        return json.dumps({"label": "passthrough", "query": "", "target": "", "confidence": 0.0})

    # --- CLARIFY / CONFIRM / SUMMARY ------------------------------
    if "clarification" in p.lower() or "ambiguous" in p.lower():
        return "[LLM: clarify prompt]"
    if "confirm" in p.lower() or "Shall" in p or "approve" in p.lower():
        return "Sounds good — shall I find the best prices and plan your route? (yes/no)"
    if "shopping" in p.lower() and "route" in p.lower():
        return "### Your plan\n* Giant Eagle — milk: $3.19\nTotal: $3.19"
    return "[LLM: fallback]"


# ───────────────────────────── price stub ─────────────────────────────

# Intentionally spread across distinct stores so plan_size reflects the
# number of items in tests (1 item → 1 store, 2 items → 2 stores).
_FAKE_CATALOG = {
    "milk": ("aldi", "Whole Milk 1 Gallon", 3.19),
    "eggs": ("target", "Large Eggs Dozen", 2.49),
    "bread": ("trader_joes", "Sourdough Loaf", 2.99),
    "orange juice": ("giant_eagle", "Simply Pulp Free Orange Juice", 2.59),
    "ginger": ("whole_foods", "Faygo Ginger Ale, Diet (20 oz)", 0.89),  # the bad pick
    "water": ("aldi", "PurAqua Sparkling Water", 0.65),
    "rice": ("target", "Earthly Grains Instant Jasmine Rice", 2.15),
    "pasta": ("trader_joes", "Penne Pasta 1 lb", 1.19),
    "chicken": ("giant_eagle", "Bell & Evans Chicken Thighs", 1.99),
    "beef": ("trader_joes", "Ground Beef 1lb", 5.99),
}

def fake_optimize_shopping_list(items):
    plan: dict[str, list] = {}
    total = 0.0
    not_found: list[str] = []
    for q in items:
        key = None
        q_lower = q.lower()
        for k in _FAKE_CATALOG:
            if k in q_lower:
                key = k
                break
        if key is None:
            not_found.append(q)
            continue
        sid, sku, price = _FAKE_CATALOG[key]
        plan.setdefault(sid, []).append({
            "item": sku, "source_item": q, "price": price, "store_display": sid,
            "url": f"https://example.com/{sid}/{key}", "source": "cache",
        })
        total += price
    return {
        "plan": plan, "total_cost": round(total, 2), "not_found": not_found,
        "store_ids": list(plan.keys()),
        "stores_meta": {sid: {"id": sid, "name": sid, "address": "123 Fake St"} for sid in plan},
    }


def fake_plan_route(store_ids, stores_meta):
    return {"stops": [{"store_id": sid, "name": sid} for sid in store_ids], "total_minutes": 5}


# ───────────────────────────── harness ─────────────────────────────

ISSUES: list[str] = []

def record(scenario: str, turn_idx: int, issue: str):
    ISSUES.append(f"[{scenario}] turn {turn_idx}: {issue}")

def run_scenario(name: str, messages: list[tuple[str, dict]], seed_session: ShoppingSession | None = None):
    """Each entry is (user_msg, expected_checks) where expected_checks is
    a dict of soft asserts to validate after the turn:
      - state: str          session.state should equal this
      - state_not: str      session.state should NOT equal this
      - reply_contains: str substring expected in reply (case-insensitive)
      - reply_not: str      substring that MUST NOT appear
      - items_len: int      len(session.raw_items)
      - plan_size: int      number of stores in shopping_plan.plan
    """
    print(f"\n━━━ Scenario: {name} ━━━")
    s = seed_session if seed_session is not None else ShoppingSession()
    for i, (msg, checks) in enumerate(messages, 1):
        print(f"  [{i}] > {msg}")
        try:
            reply = chat(s, msg)
        except Exception as e:
            record(name, i, f"CRASH: {type(e).__name__}: {e}")
            traceback.print_exc()
            return
        print(f"      state={s.state!r} | items={len(s.raw_items)} | "
              f"plan_stores={len((s.shopping_plan or {}).get('plan') or {})}")
        snippet = reply.replace("\n", " ")[:110]
        print(f"      Agent: {snippet}...")

        if not reply or not reply.strip():
            record(name, i, "empty reply")

        expected_state = checks.get("state")
        if expected_state and s.state != expected_state:
            record(name, i, f"state={s.state!r} expected {expected_state!r}")

        forbidden_state = checks.get("state_not")
        if forbidden_state and s.state == forbidden_state:
            record(name, i, f"state is forbidden value {forbidden_state!r}")

        needle = checks.get("reply_contains")
        if needle and needle.lower() not in reply.lower():
            record(name, i, f"reply missing {needle!r}")

        blocked = checks.get("reply_not")
        if blocked and blocked.lower() in reply.lower():
            record(name, i, f"reply contains blocked {blocked!r}")

        n_items = checks.get("items_len")
        if n_items is not None and len(s.raw_items) != n_items:
            record(name, i, f"raw_items={len(s.raw_items)} expected {n_items}")

        n_plan = checks.get("plan_size")
        if n_plan is not None:
            got = len((s.shopping_plan or {}).get("plan") or {})
            if got != n_plan:
                record(name, i, f"plan_size={got} expected {n_plan}")


def main():
    import agent.agent as ag_mod
    import tools.price_optimizer as opt_mod
    ag_mod.call_llm = fake_call_llm
    ag_mod.optimize_shopping_list = fake_optimize_shopping_list
    ag_mod.plan_route = fake_plan_route
    opt_mod.optimize_shopping_list = fake_optimize_shopping_list

    # ─── Scenario battery ───────────────────────────────────────
    # NOTE: flow is CLARIFY → (yes #1 → CONFIRM) → (yes #2 → EXECUTE).
    # Two "yes" turns is intentional: the first confirms "that's my list"
    # (exit clarify), the second approves the priced plan.

    run_scenario("happy-path-basic-list", [
        ("I need milk, eggs, and bread", {"items_len": 3}),
        ("that's all", {"state": "CONFIRM"}),
        ("yes", {"state": "EXECUTE", "plan_size": 3}),   # milk/eggs/bread → 3 stores
        ("no thanks", {"state": "DONE", "reply_contains": "happy"}),
    ])

    run_scenario("dish-flow-carbonara", [
        ("I want to make carbonara tonight", {"reply_contains": "carbonara"}),
        ("yes", {}),                                # add ingredients
        ("yes", {"state": "EXECUTE"}),              # approve the plan
    ])

    run_scenario("dish-unknown-returns-helpful", [
        ("I want to make ropa vieja", {"reply_not": "happy"}),
    ])

    run_scenario("remove-after-plan", [
        # milk → aldi, water → aldi (both in one store post-optimize).
        # Use bread + water so they spread across 2 distinct stores.
        ("I need bread and water", {"items_len": 2}),
        ("that's it", {"state": "CONFIRM"}),
        ("yes", {"state": "EXECUTE", "plan_size": 2}),
        ("No i don't want the water", {
            "reply_contains": "water",
            "state_not": "CLARIFY",
            "items_len": 1, "plan_size": 1,
        }),
    ])

    run_scenario("remove-twice-second-noop-now-empties-to-clarify", [
        ("I need water", {"items_len": 1}),
        ("that's all", {"state": "CONFIRM"}),
        ("yes", {"state": "EXECUTE", "plan_size": 1}),
        ("remove the water", {
            "items_len": 0, "plan_size": 0,
            "state": "CLARIFY",                       # post-empty auto-rescue
            "reply_contains": "empty",
        }),
        ("remove the water", {
            # No list, no plan → remove intent bails; treated as new list.
            "state_not": "EXECUTE",
        }),
    ])

    run_scenario("justify-after-plan", [
        ("I need ginger", {"items_len": 1}),
        ("that's all", {"state": "CONFIRM"}),
        ("yes", {"state": "EXECUTE", "plan_size": 1}),
        ("why did you pick the ginger ale?", {
            "reply_contains": "ginger ale",
            "state_not": "CLARIFY",
            "items_len": 1, "plan_size": 1,
        }),
    ])

    run_scenario("justify-before-plan-now-answers", [
        # No plan yet → handler should politely say "nothing to justify".
        ("why did you pick milk", {
            "reply_contains": "no active plan",
            "state_not": "EXECUTE",
        }),
    ])

    run_scenario("close-then-new-list", [
        ("I need milk", {}),
        ("that's all", {"state": "CONFIRM"}),
        ("yes", {"state": "EXECUTE"}),
        ("thanks", {"state": "DONE"}),
        ("actually I need bread too", {"state_not": "DONE"}),
    ])

    run_scenario("ambiguous-cancel-in-confirm-asks-for-guidance", [
        ("I need milk", {}),
        ("that's all", {"state": "CONFIRM"}),
        ("no", {"state": "CONFIRM", "reply_contains": "change"}),
    ])

    run_scenario("refinement-in-confirm", [
        ("I need milk", {}),
        ("that's all", {"state": "CONFIRM"}),
        ("actually make it 2 gallons of whole milk", {"state": "EXECUTE"}),
    ])

    run_scenario("empty-and-punctuation-noise", [
        ("", {}),
        ("?", {}),
        ("hmm", {}),                # ambiguous short reply
    ])

    run_scenario("remove-in-fresh-session-noop", [
        # No list, no plan — should NOT trigger remove, should NOT crash.
        ("drop the water", {"state_not": "EXECUTE"}),
    ])

    run_scenario("why-not-does-not-trigger-justify", [
        ("I need milk", {}),
        ("that's all", {"state": "CONFIRM"}),
        ("yes", {"state": "EXECUTE"}),
        ("why not add more?", {"reply_not": "here's why"}),
    ])

    # Construct an ambiguous session directly: raw_items has BOTH "orange"
    # and "orange mango juice" (no single exact match). User says "remove
    # the orange" — handler must ask, not silently nuke the juice.
    ambiguous_session = ShoppingSession()
    ambiguous_session.state = "EXECUTE"
    ambiguous_session.clarification_done = True
    ambiguous_session.raw_items = [
        {"name": "orange mango",  "quantity": 1, "unit": None, "ambiguous": False},
        {"name": "orange juice",  "quantity": 1, "unit": None, "ambiguous": False},
    ]
    ambiguous_session.shopping_plan = {
        "plan": {
            "giant_eagle": [
                {"item": "Simply Orange Juice", "source_item": "orange juice",
                 "price": 2.59, "store_display": "giant_eagle", "url": None},
            ],
            "aldi": [
                {"item": "Orange Mango Smoothie", "source_item": "orange mango",
                 "price": 1.99, "store_display": "aldi", "url": None},
            ],
        },
        "total_cost": 4.58, "not_found": [],
        "store_ids": ["giant_eagle", "aldi"],
        "stores_meta": {"giant_eagle": {}, "aldi": {}},
    }
    run_scenario("remove-orange-ambiguous-asks-for-clarification", [
        ("remove the orange", {
            "items_len": 2, "plan_size": 2,            # UNTOUCHED
            "reply_contains": "matches multiple items",
            "state": "EXECUTE",                        # no state change
        }),
    ], seed_session=ambiguous_session)

    # Exact-match precedence: target "orange" exactly matches the raw
    # item named "orange", even though "orange juice" also token-matches.
    exact_session = ShoppingSession()
    exact_session.state = "EXECUTE"
    exact_session.clarification_done = True
    exact_session.raw_items = [
        {"name": "orange",        "quantity": 1, "unit": None, "ambiguous": False},
        {"name": "orange juice",  "quantity": 1, "unit": None, "ambiguous": False},
    ]
    exact_session.shopping_plan = {
        "plan": {
            "giant_eagle": [
                {"item": "Simply Orange Juice", "source_item": "orange juice",
                 "price": 2.59, "store_display": "giant_eagle", "url": None},
            ],
            "aldi": [
                {"item": "Navel Oranges 3 lb", "source_item": "orange",
                 "price": 3.99, "store_display": "aldi", "url": None},
            ],
        },
        "total_cost": 6.58, "not_found": [],
        "store_ids": ["giant_eagle", "aldi"],
        "stores_meta": {"giant_eagle": {}, "aldi": {}},
    }
    run_scenario("remove-orange-exact-match-wins-leaves-juice", [
        ("remove the orange", {
            "items_len": 1,                            # only "orange" removed
            "plan_size": 1,                            # juice SKU kept
            "reply_not": "Simply Orange Juice",        # juice NOT in reply
        }),
    ], seed_session=exact_session)

    run_scenario("pick-intent-without-last-options-noop", [
        ("I need beef", {}),
        ("pick 3", {"state_not": "EXECUTE"}),
    ])

    # ─── Report ─────────────────────────────────────────────────
    print("\n" + "═" * 50)
    if ISSUES:
        print(f"Found {len(ISSUES)} potential issue(s):")
        for iss in ISSUES:
            print(f"  • {iss}")
        return 1
    print("All scenarios passed — no anomalies detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
