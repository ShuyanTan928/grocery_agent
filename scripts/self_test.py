"""Non-interactive self-test harness for the LLM tool-calling agent.

The real chat() loop normally makes at least one LLM call per turn to
decide which tool to invoke. For deterministic regression testing we
stub `call_llm` with a `ScriptedLLM` that just pops pre-written JSON
tool calls off a per-turn queue. `optimize_shopping_list` and
`plan_route` are also stubbed so tests don't hit the cache/ORS.

Each scenario is a list of turns; each turn names the user message,
the exact sequence of tool-call/reply JSON strings the "LLM" should
emit for that turn, and a dict of post-turn assertions (state /
raw_items / plan / reply).

Run:
    uv run python scripts/self_test.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.agent import ShoppingSession, chat


# ───────────────────────────── scripted LLM ─────────────────────────


class ScriptedLLM:
    """Minimal stub that hands out pre-written JSON tool calls. The
    scenario runner calls `set()` at the start of each turn to load a
    fresh queue, then `chat()` will call `__call__` one or more times
    to drive the loop until a `reply` is emitted."""

    def __init__(self):
        self._queue: list[str] = []
        self._fallback_reply: str = "(scripted LLM: no response queued)"
        self.log: list[str] = []

    def set(self, responses: list[str], *, fallback_reply: str | None = None) -> None:
        self._queue = list(responses)
        if fallback_reply is not None:
            self._fallback_reply = fallback_reply
        else:
            self._fallback_reply = "(scripted LLM: no response queued)"

    def __call__(self, prompt, *, model=None, temperature=None, system=None):
        self.log.append(prompt[-200:])
        if self._queue:
            return self._queue.pop(0)
        # Safety net: if the loop asks for more responses than the
        # scenario queued, force an emergency-style reply so the loop
        # terminates instead of hanging in an infinite retry.
        return json.dumps({"tool": "reply", "args": {"text": self._fallback_reply}})


SCRIPT = ScriptedLLM()


# ───────────────────────────── pricing stub ─────────────────────────

# Distribute items across distinct stores so plan_size reflects item
# count (1 item -> 1 store, 2 items -> 2 stores). Store ids match the
# real mock_stores.json so apply_* helpers don't balk.
_FAKE_CATALOG = {
    "milk":         ("aldi_greenfield",          "Whole Milk 1 Gallon",            3.19),
    "eggs":         ("target_east_liberty",      "Large Eggs Dozen",               2.49),
    "bread":        ("trader_joes_shadyside",    "Sourdough Loaf",                 2.99),
    "orange":       ("aldi_greenfield",          "Navel Oranges 3 lb",             3.99),
    "orange juice": ("giant_eagle_squirrel_hill","Simply Pulp Free Orange Juice",  2.59),
    "water":        ("aldi_greenfield",          "PurAqua Sparkling Water",        0.65),
    "ginger":       ("whole_foods_east_liberty", "Faygo Ginger Ale, Diet (20 oz)", 0.89),
    "rice":         ("target_east_liberty",      "Earthly Grains Jasmine Rice",    2.15),
    "pasta":        ("trader_joes_shadyside",    "Penne Pasta 1 lb",               1.19),
    "chicken":      ("giant_eagle_squirrel_hill","Bell & Evans Chicken Thighs",    1.99),
    "beef":         ("trader_joes_shadyside",    "Ground Beef 1 lb",               5.99),
    "lamb":         ("giant_eagle_squirrel_hill","Catelli Lamb Leg",              10.87),
}


def fake_optimize_shopping_list(items):
    plan: dict[str, list] = {}
    total = 0.0
    not_found: list[str] = []
    for q in items:
        q_lower = q.lower()
        key = next((k for k in _FAKE_CATALOG if k in q_lower), None)
        if key is None:
            not_found.append(q)
            continue
        sid, sku, price = _FAKE_CATALOG[key]
        plan.setdefault(sid, []).append({
            "item": sku,
            "source_item": q,
            "price": price,
            "store_display": sid,
            "url": f"https://example.com/{sid}/{key.replace(' ', '_')}",
            "source": "cache",
        })
        total += price
    return {
        "plan": plan,
        "total_cost": round(total, 2),
        "not_found": not_found,
        "store_ids": list(plan.keys()),
        "stores_meta": {sid: {"id": sid, "name": sid, "address": "123 Fake St"}
                        for sid in plan},
    }


def fake_plan_route(store_ids, stores_meta):
    return {
        "stops": [{"store_id": sid, "name": sid} for sid in store_ids],
        "total_minutes": 5 * max(1, len(store_ids)),
    }


# ───────────────────────────── harness ─────────────────────────────

ISSUES: list[str] = []


def record(scenario: str, turn_idx: int, issue: str) -> None:
    ISSUES.append(f"[{scenario}] turn {turn_idx}: {issue}")


def _plan_size(session) -> int:
    return len((session.shopping_plan or {}).get("plan") or {})


def run_scenario(
    name: str,
    turns: list[tuple[str, list[str], dict]],
    *,
    seed_session: ShoppingSession | None = None,
) -> None:
    """Each turn = (user_msg, scripted_llm_outputs, checks).

    checks keys:
      reply_contains        substring (case-insensitive)
      reply_not             substring that MUST NOT appear
      items_len             expected len(raw_items)
      plan_size             expected store count in shopping_plan.plan
      plan_total            expected total_cost (float tolerant)
      plan_missing          must not have a plan (plan None or empty)
      has_plan              True/False quick check
      pending_dish_name     substring of pending_dish name, or None
      last_options_count    expected len(last_options)
    """
    print(f"\n━━━ Scenario: {name} ━━━")
    s = seed_session if seed_session is not None else ShoppingSession()
    for i, (msg, script, checks) in enumerate(turns, 1):
        SCRIPT.set(script, fallback_reply=f"(no scripted reply in turn {i})")
        print(f"  [{i}] > {msg}")
        try:
            reply = chat(s, msg, max_steps=len(script) + 2)
        except Exception as e:
            record(name, i, f"CRASH: {type(e).__name__}: {e}")
            traceback.print_exc()
            return
        print(
            f"      items={len(s.raw_items)} plan_stores={_plan_size(s)} "
            f"pending_dish={'yes' if s.pending_dish else 'no'}"
        )
        print(f"      Agent: {reply[:100]}...")

        if not reply or not reply.strip():
            record(name, i, "empty reply")

        needle = checks.get("reply_contains")
        if needle and needle.lower() not in (reply or "").lower():
            record(name, i, f"reply missing {needle!r}")

        blocked = checks.get("reply_not")
        if blocked and blocked.lower() in (reply or "").lower():
            record(name, i, f"reply contains blocked {blocked!r}")

        n_items = checks.get("items_len")
        if n_items is not None and len(s.raw_items) != n_items:
            record(name, i, f"raw_items={len(s.raw_items)} expected {n_items}")

        n_plan = checks.get("plan_size")
        if n_plan is not None and _plan_size(s) != n_plan:
            record(name, i, f"plan_size={_plan_size(s)} expected {n_plan}")

        total = checks.get("plan_total")
        if total is not None:
            got = (s.shopping_plan or {}).get("total_cost")
            if got is None or abs(float(got) - float(total)) > 1e-6:
                record(name, i, f"plan_total={got} expected {total}")

        if checks.get("plan_missing"):
            if s.shopping_plan and s.shopping_plan.get("plan"):
                record(name, i, f"expected no plan, got {_plan_size(s)} stores")

        hp = checks.get("has_plan")
        if hp is True and not (s.shopping_plan and s.shopping_plan.get("plan")):
            record(name, i, "expected a plan")
        if hp is False and (s.shopping_plan and s.shopping_plan.get("plan")):
            record(name, i, "did not expect a plan")

        pdn = checks.get("pending_dish_name")
        if pdn is not None:
            actual = (s.pending_dish or {}).get("name", "")
            if pdn and pdn.lower() not in actual.lower():
                record(name, i, f"pending_dish_name={actual!r} missing {pdn!r}")

        loc = checks.get("last_options_count")
        if loc is not None and len(s.last_options) != loc:
            record(name, i, f"last_options_count={len(s.last_options)} expected {loc}")


# ───────────────────────── scripted helpers ────────────────────────


def tool_call(_name: str, **args) -> str:
    return json.dumps({"tool": _name, "args": args})


def reply(text: str) -> str:
    return json.dumps({"tool": "reply", "args": {"text": text}})


# ───────────────────────────── main ────────────────────────────────


def main() -> int:
    # Monkey-patch BEFORE any scenario runs.
    import agent.agent as ag_mod
    import agent.tools as tools_mod
    ag_mod.call_llm = SCRIPT
    # The loop imports call_llm at runtime from agent.agent — replacing
    # the module attribute is enough.
    tools_mod.optimize_shopping_list = fake_optimize_shopping_list
    tools_mod.plan_route = fake_plan_route

    # ── Scenario battery ──────────────────────────────────────────

    run_scenario("happy-path-basic-list", [
        ("I need milk, eggs, and bread", [
            tool_call("add_items", items=[
                {"name": "milk", "quantity": 1, "unit": "gallon", "ambiguous": False},
                {"name": "eggs", "quantity": 1, "unit": "dozen", "ambiguous": False},
                {"name": "bread", "quantity": 1, "unit": "loaf", "ambiguous": False},
            ]),
            reply("Got it: milk, eggs, bread. Shall I plan the best prices? (yes/no)"),
        ], {"items_len": 3, "has_plan": False, "reply_contains": "milk"}),
        ("yes", [
            tool_call("optimize_and_route"),
            reply("Here's the cheapest plan: ...\nTotal: $8.67"),
        ], {"plan_size": 3, "plan_total": 3.19 + 2.49 + 2.99, "reply_contains": "total"}),
        ("no thanks", [
            reply("You're all set — happy shopping!"),
        ], {"reply_contains": "happy"}),
    ])

    run_scenario("ambiguous-quantity-asks-clarify", [
        ("I need pork chops", [
            tool_call("add_items", items=[
                {"name": "pork chops", "quantity": None, "unit": None, "ambiguous": True},
            ]),
            reply("How many lbs of pork chops would you like?"),
        ], {"items_len": 1, "reply_contains": "lbs"}),
        ("2 lbs", [
            tool_call("update_quantity", name="pork chops", quantity=2, unit="lb"),
            reply("Got it — 2 lb pork chops. Ready to plan? (yes/no)"),
        ], {"reply_contains": "2 lb"}),
    ])

    run_scenario("dish-flow-carbonara", [
        ("I want to make carbonara tonight", [
            tool_call("propose_dish", name="carbonara"),
            reply("To make spaghetti carbonara, here's the list: 1) spaghetti 2) bacon 3) eggs 4) parmesan. Add all? (yes/no)"),
        ], {"pending_dish_name": "carbonara", "reply_contains": "spaghetti"}),
        ("yes", [
            tool_call("apply_pending_dish"),
            reply("Added 4 items — ready to plan? (yes/no)"),
        ], {"pending_dish_name": "", "reply_contains": "4"}),
    ])

    run_scenario("dish-unknown-returns-helpful", [
        ("I want to make ropa vieja", [
            tool_call("lookup_dish", name="ropa vieja"),
            reply("I don't have a recipe for ropa vieja yet — you can list the ingredients directly, or enable USE_LLM_DISH_FALLBACK."),
        ], {"reply_contains": "don't have", "has_plan": False}),
    ])

    run_scenario("remove-after-plan", [
        ("I need bread and water", [
            tool_call("add_items", items=[
                {"name": "bread", "quantity": 1, "unit": "loaf", "ambiguous": False},
                {"name": "water", "quantity": 1, "unit": "bottle", "ambiguous": False},
            ]),
            reply("Got it: bread, water. Plan? (yes/no)"),
        ], {"items_len": 2}),
        ("yes", [
            tool_call("optimize_and_route"),
            reply("Plan ready: 2 stores, $3.64 total."),
        ], {"plan_size": 2}),
        ("No I don't want the water", [
            tool_call("remove_items", target="water"),
            reply("Dropped the water — just bread left ($2.99 at Trader Joe's)."),
        ], {"items_len": 1, "plan_size": 1, "reply_contains": "water"}),
    ])

    run_scenario("remove-empties-list-resets", [
        ("I need water", [
            tool_call("add_items", items=[
                {"name": "water", "quantity": 1, "unit": "bottle", "ambiguous": False},
            ]),
            reply("Got it: water. Plan? (yes/no)"),
        ], {"items_len": 1}),
        ("yes", [
            tool_call("optimize_and_route"),
            reply("Plan ready: water at Aldi for $0.65."),
        ], {"plan_size": 1}),
        ("remove the water", [
            tool_call("remove_items", target="water"),
            reply("Dropped the water — your list is empty. Want to start fresh?"),
        ], {"items_len": 0, "plan_missing": True, "reply_contains": "empty"}),
    ])

    run_scenario("justify-after-plan", [
        ("I need ginger", [
            tool_call("add_items", items=[
                {"name": "ginger", "quantity": 1, "unit": None, "ambiguous": False},
            ]),
            reply("Got it. Plan? (yes/no)"),
        ], {"items_len": 1}),
        ("yes", [
            tool_call("optimize_and_route"),
            reply("Plan ready: Faygo Ginger Ale at Whole Foods for $0.89."),
        ], {"plan_size": 1, "reply_contains": "ginger ale"}),
        ("why did you pick the ginger ale?", [
            tool_call("justify_pick", target="ginger ale"),
            reply("Faygo Ginger Ale was picked for your 'ginger' request — it was the cheapest cache match."),
        ], {"reply_contains": "ginger ale", "items_len": 1, "plan_size": 1}),
    ])

    run_scenario("justify-before-plan-says-none", [
        ("why did you pick milk", [
            tool_call("justify_pick", target="milk"),
            reply("There's no active plan right now — nothing to justify yet."),
        ], {"reply_contains": "no active plan", "has_plan": False}),
    ])

    run_scenario("list-options-then-pick", [
        ("I need lamb", [
            tool_call("add_items", items=[
                {"name": "lamb", "quantity": 1, "unit": None, "ambiguous": False},
            ]),
            reply("Got it. Plan? (yes/no)"),
        ], {"items_len": 1}),
        ("list the options", [
            tool_call("list_options", query="lamb", topk=3),
            reply("Here are a few lamb options:\n1. Catelli Lamb Leg — Giant Eagle — $10.87"),
        ], {"reply_contains": "lamb"}),  # count depends on real cache — don't pin
        ("pick 1", [
            tool_call("pick_option", n=1),
            reply("Locked in Catelli Lamb Leg at Giant Eagle for $10.87."),
        ], {"plan_size": 1, "reply_contains": "lamb"}),
    ])

    run_scenario("pick-without-staged-options-errors-and-recovers", [
        ("pick 3", [
            tool_call("pick_option", n=3),        # obs is {"error": "..."}
            reply("I don't have any options staged — say 'list options for X' first."),
        ], {"reply_contains": "list options", "plan_missing": True}),
    ])

    run_scenario("preferences-set-then-optimize", [
        ("I want pork from Trader Joe's", [
            tool_call("add_items", items=[
                {"name": "pork", "quantity": 2, "unit": "lb", "ambiguous": False},
            ]),
            tool_call("set_preference", item="pork", store_id="trader_joes_shadyside", kind="prefer"),
            reply("Got it — pork (2 lb) from Trader Joe's. Plan? (yes/no)"),
        ], {"items_len": 1, "reply_contains": "Trader"}),
    ])

    run_scenario("close-then-new-list", [
        ("I need milk", [
            tool_call("add_items", items=[
                {"name": "milk", "quantity": 1, "unit": "gallon", "ambiguous": False},
            ]),
            reply("Got it. Plan? (yes/no)"),
        ], {}),
        ("yes", [
            tool_call("optimize_and_route"),
            reply("Plan ready: milk at Aldi for $3.19."),
        ], {"plan_size": 1}),
        ("thanks", [
            reply("You're all set — happy shopping!"),
        ], {"reply_contains": "happy"}),
        ("actually I need bread too", [
            tool_call("add_items", items=[
                {"name": "bread", "quantity": 1, "unit": "loaf", "ambiguous": False},
            ]),
            reply("Got it — added bread. Plan? (yes/no)"),
        ], {"items_len": 2, "reply_contains": "bread"}),
    ])

    # Seed an ambiguous session: two raw items that both loose-match
    # "orange". remove_items should return {ambiguous:true, matches:[...]},
    # and the LLM should reply asking for clarification — NOT nuke the list.
    ambiguous = ShoppingSession()
    ambiguous.raw_items = [
        {"name": "orange mango", "quantity": 1, "unit": None, "ambiguous": False},
        {"name": "orange juice", "quantity": 1, "unit": None, "ambiguous": False},
    ]
    ambiguous.shopping_plan = {
        "plan": {
            "giant_eagle_squirrel_hill": [
                {"item": "Simply Orange Juice", "source_item": "orange juice",
                 "price": 2.59, "store_display": "giant_eagle_squirrel_hill", "url": None},
            ],
            "aldi_greenfield": [
                {"item": "Orange Mango Smoothie", "source_item": "orange mango",
                 "price": 1.99, "store_display": "aldi_greenfield", "url": None},
            ],
        },
        "total_cost": 4.58, "not_found": [],
        "store_ids": ["giant_eagle_squirrel_hill", "aldi_greenfield"],
        "stores_meta": {"giant_eagle_squirrel_hill": {}, "aldi_greenfield": {}},
    }
    run_scenario("remove-orange-ambiguous-asks-clarification", [
        ("remove the orange", [
            tool_call("remove_items", target="orange"),
            reply("'orange' matches multiple items on your list: orange mango, orange juice. Which one should I drop?"),
        ], {"items_len": 2, "plan_size": 2, "reply_contains": "multiple"}),
    ], seed_session=ambiguous)

    # Exact match precedence: raw item "orange" with raw item "orange juice"
    # alongside. remove target="orange" must drop ONLY the exact-match row,
    # keep the juice.
    exact = ShoppingSession()
    exact.raw_items = [
        {"name": "orange",       "quantity": 1, "unit": None, "ambiguous": False},
        {"name": "orange juice", "quantity": 1, "unit": None, "ambiguous": False},
    ]
    exact.shopping_plan = {
        "plan": {
            "giant_eagle_squirrel_hill": [
                {"item": "Simply Orange Juice", "source_item": "orange juice",
                 "price": 2.59, "store_display": "giant_eagle_squirrel_hill", "url": None},
            ],
            "aldi_greenfield": [
                {"item": "Navel Oranges 3 lb", "source_item": "orange",
                 "price": 3.99, "store_display": "aldi_greenfield", "url": None},
            ],
        },
        "total_cost": 6.58, "not_found": [],
        "store_ids": ["giant_eagle_squirrel_hill", "aldi_greenfield"],
        "stores_meta": {"giant_eagle_squirrel_hill": {}, "aldi_greenfield": {}},
    }
    run_scenario("remove-orange-exact-match-keeps-juice", [
        ("remove the orange", [
            tool_call("remove_items", target="orange"),
            reply("Dropped the orange — the orange juice is still on your list."),
        ], {"items_len": 1, "plan_size": 1, "reply_not": "Simply Orange Juice"}),
    ], seed_session=exact)

    run_scenario("malformed-json-retries-then-reply", [
        ("I need beef", [
            "Sure, let's go!",                         # not JSON
            "```json not quite either```",             # still not
            tool_call("add_items", items=[
                {"name": "beef", "quantity": 1, "unit": "lb", "ambiguous": False},
            ]),
            reply("Got it — beef. Plan? (yes/no)"),
        ], {"items_len": 1, "reply_contains": "beef"}),
    ])

    run_scenario("unknown-tool-recovers", [
        ("do something", [
            tool_call("does_not_exist", foo="bar"),    # obs: {"error": "unknown tool..."}
            reply("I can't do that directly. What groceries do you need?"),
        ], {"reply_contains": "what groceries"}),
    ])

    run_scenario("chained-preferences-then-plan", [
        ("I need milk and chicken, avoid trader joes for meat", [
            tool_call("add_items", items=[
                {"name": "milk", "quantity": 1, "unit": "gallon", "ambiguous": False},
                {"name": "chicken", "quantity": 2, "unit": "lb", "ambiguous": False},
            ]),
            tool_call("set_preference", item="chicken",
                      store_id="trader_joes_shadyside", kind="avoid"),
            reply("Got it — milk and chicken, avoiding Trader Joe's for meat. Plan? (yes/no)"),
        ], {"items_len": 2, "reply_contains": "avoiding"}),
        ("yes", [
            tool_call("optimize_and_route"),
            reply("Plan ready: 2 stores."),
        ], {"plan_size": 2}),
    ])

    # ── Report ──────────────────────────────────────────────────
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
