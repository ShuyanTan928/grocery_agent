#!/usr/bin/env python3
"""End-to-end agent conversation that exercises the full ReAct flow:
  - mixed shopping list (dairy + meat + produce + pantry)
  - ambiguous-by-weight item that should trigger CLARIFY for quantity
  - 'prefer this store' preference (pork → Trader Joe's)
  - 'avoid this store' preference (chicken → not Whole Foods)
  - explicit yes-confirmation to reach EXECUTE

Prints each turn, the post-turn state, and then dumps the final plan.

    uv run python scripts/realistic_chat.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from agent.agent import ShoppingSession, chat  # noqa: E402


SCRIPT = [
    # Turn 1: kitchen-sink list — pork is by-weight (ambiguous), milk/eggs/bread fixed-size.
    "Hi! I need pork chops, a gallon of whole milk, a dozen eggs, a loaf of bread, "
    "and some bananas. Get the pork from Trader Joe's please.",

    # Turn 2: clarify the ambiguous quantity + add an avoid preference.
    "Make it 2 lb of pork chops, and 2 lb of bananas. Also please don't get the bread "
    "from Whole Foods — too expensive.",

    # Turn 3: confirm to execute.
    "yes, looks good",
]


def banner(label: str) -> None:
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)


def render_state(session: ShoppingSession) -> str:
    pieces = [
        f"state={session.state}",
        f"clarification_done={session.clarification_done}",
    ]
    ambig = [i["name"] for i in session.raw_items if i.get("ambiguous")]
    if ambig:
        pieces.append(f"ambiguous={ambig}")
    if session.preferences:
        pieces.append(f"avoid={session.preferences}")
    if session.preferred_stores:
        pieces.append(f"prefer={session.preferred_stores}")
    return "[" + " | ".join(pieces) + "]"


def main() -> int:
    banner("Realistic agent chat — fresh session, real LLM, mock prices")
    session = ShoppingSession()

    for i, msg in enumerate(SCRIPT, start=1):
        banner(f"Turn {i}")
        print(f"You   : {msg}")
        reply = chat(session, msg)
        print(f"Agent : {reply}")
        print(render_state(session))

    banner("Final plan — internal state")
    print(json.dumps({
        "raw_items": session.raw_items,
        "avoid": session.preferences,
        "prefer": session.preferred_stores,
    }, indent=2, ensure_ascii=False))

    if session.shopping_plan:
        plan = session.shopping_plan
        print()
        print(f"Total: ${plan['total_cost']}  across {len(plan['store_ids'])} stores")
        for sid, items in plan["plan"].items():
            print(f"\n  ── {sid} ──")
            for it in items:
                print(f"     ${it['price']:>5.2f}  {it['item']}")
        if plan.get("unfulfilled_preferences"):
            print("\n  Unfulfilled preferences:")
            for u in plan["unfulfilled_preferences"]:
                print(f"     - {u}")
        if plan.get("not_found"):
            print(f"\n  Not found: {plan['not_found']}")

    if session.route_plan:
        print()
        print("Route:")
        for s in session.route_plan.get("ordered_stops", []):
            leg = s.get("leg_duration_min")
            leg_s = f"+{leg} min" if leg else "(start)"
            print(f"     {leg_s}  {s['name']}")
        rp = session.route_plan
        if rp.get("total_duration_min"):
            print(f"     ─ total drive: {rp['total_duration_min']} min, "
                  f"{rp.get('total_distance_km', '?')} km ─")

    return 0 if session.state == "EXECUTE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
