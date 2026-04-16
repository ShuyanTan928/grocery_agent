#!/usr/bin/env python3
"""Trace a scenario: user wants pork, preferring Trader Joe's.

Shows:
  1. What the TJ real cache has for "pork" (top cheapest)
  2. What the agent's current (mock-backed) flow produces
  3. Internal state after each turn (items, preferences, plan)

    uv run python scripts/test_pork_tj.py
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
from tools.product_search import search_products  # noqa: E402


def show_real_tj_pork() -> None:
    print("=" * 60)
    print("[1] Real Trader Joe's cache — 'pork' top 5 cheapest")
    print("=" * 60)
    hits = search_products("pork", store_ids=["trader_joes_shadyside"], limit=5)
    for h in hits:
        print(f"  ${h['item_price']:>5.2f}  {h['item_name']}")
    print(f"  (total pork matches in cache: "
          f"{len(search_products('pork', store_ids=['trader_joes_shadyside']))})")


def run_conversation(script: list[str]) -> ShoppingSession:
    print()
    print("=" * 60)
    print("[2] Agent conversation (uses MOCK prices in price_optimizer)")
    print("=" * 60)
    session = ShoppingSession()
    for i, msg in enumerate(script, start=1):
        print(f"\n--- Turn {i} ---")
        print(f"You   : {msg}")
        reply = chat(session, msg)
        print(f"Agent : {reply}")
        print(f"[state={session.state}  "
              f"clarification_done={session.clarification_done}  "
              f"ambiguous={[i for i in session.raw_items if i.get('ambiguous')]} ]")
    return session


def dump_final_plan(session: ShoppingSession) -> None:
    print()
    print("=" * 60)
    print("[3] Final internal state")
    print("=" * 60)
    print("raw_items:")
    print(json.dumps(session.raw_items, indent=2, ensure_ascii=False))
    print("\npreferences (avoid):")
    print(json.dumps(session.preferences, indent=2, ensure_ascii=False))
    print("\npreferred_stores (require):")
    print(json.dumps(session.preferred_stores, indent=2, ensure_ascii=False))
    if session.shopping_plan:
        print("\nshopping_plan:")
        print(json.dumps(session.shopping_plan, indent=2, ensure_ascii=False))
    if session.route_plan:
        print("\nroute_plan (ordered_stops):")
        for s in session.route_plan.get("ordered_stops", []):
            print(f"  - {s['name']}  leg={s.get('leg_duration_min')} min")


def main() -> int:
    show_real_tj_pork()
    script = [
        "I want 2 lb pork chops from Trader Joe's, and a gallon of milk",
        "no other preferences",
        "yes",
    ]
    session = run_conversation(script)
    dump_final_plan(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
