#!/usr/bin/env python3
"""Step-by-step pipeline doctor for the grocery agent.

Runs independent checks in order, prints OK/FAIL per step. Later steps
are skipped automatically if an earlier dependency fails. Designed to
make it obvious which layer broke when something regresses.

    uv run python scripts/test_pipeline.py
    uv run python scripts/test_pipeline.py --skip-llm     # offline checks only
    uv run python scripts/test_pipeline.py --skip-network # also skip TJ scrape
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


# --------------- tiny step runner ---------------

class StepResult:
    def __init__(self, name: str):
        self.name = name
        self.ok: bool | None = None
        self.detail: str = ""
        self.elapsed_ms: float = 0.0


def run_step(name: str, fn: Callable[[], str], *, skip_if_failed: list[StepResult] | None = None) -> StepResult:
    r = StepResult(name)
    if skip_if_failed and any(s.ok is False for s in skip_if_failed):
        r.ok = None
        r.detail = f"skipped (depends on {[s.name for s in skip_if_failed if s.ok is False]})"
        print(f"  [SKIP] {name}: {r.detail}")
        return r

    print(f"  [ .. ] {name} ...", end="", flush=True)
    t0 = time.perf_counter()
    try:
        r.detail = fn() or ""
        r.ok = True
    except Exception as e:
        r.ok = False
        r.detail = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    r.elapsed_ms = (time.perf_counter() - t0) * 1000
    status = "OK" if r.ok else "FAIL"
    print(f"\r  [ {status:<4} ] {name}  ({r.elapsed_ms:.0f} ms)  {r.detail}")
    return r


# --------------- individual checks ---------------

def check_settings() -> str:
    import importlib
    import config.settings as s
    importlib.reload(s)
    missing = []
    if s.LLM_PROVIDER == "openrouter" and not s.OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    if s.LLM_PROVIDER == "google" and (not s.GOOGLE_API_KEY or s.GOOGLE_API_KEY.startswith("YOUR_")):
        missing.append("GOOGLE_API_KEY")
    if missing:
        raise RuntimeError(f"missing env: {missing}")
    return f"provider={s.LLM_PROVIDER} model={s.LLM_MODEL} tj_store={s.TRADER_JOES_STORE_CODE}"


def check_mock_data() -> str:
    from config.settings import MOCK_DATA_DIR
    base = Path(MOCK_DATA_DIR)
    for name in ("mock_prices.json", "mock_stores.json", "mock_distance_matrix.json"):
        p = base / name
        if not p.exists():
            raise RuntimeError(f"missing {p}")
        json.loads(p.read_text())
    return "mock_prices / mock_stores / mock_distance_matrix OK"


def check_cache_roundtrip() -> str:
    from tools import price_cache
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as td:
        orig = price_cache.PRICE_CACHE_DIR
        price_cache.PRICE_CACHE_DIR = td
        try:
            price_cache.save_cache("doctor_test", {"items": [], "item_count": 0})
            assert price_cache.load_cached("doctor_test") is not None
        finally:
            price_cache.PRICE_CACHE_DIR = orig
    return "save/load same-day cache OK"


def check_tj_cache_exists() -> str:
    from tools.price_cache import cache_info
    info = cache_info("trader_joes_shadyside")
    if not info:
        raise RuntimeError("no TJ cache — run `uv run python -m tools.refresh_prices --store trader_joes` first")
    return f"{info['item_count']} items, date={info['scraped_date']}"


def check_product_search() -> str:
    from tools.product_search import search_products
    res = search_products("milk", store_ids=["trader_joes_shadyside"], limit=3)
    if not res:
        raise RuntimeError("no TJ results for 'milk' — is the cache populated?")
    example = res[0]
    return f"{len(res)} hits; top='{example['item_name']}' ${example['item_price']}"


def check_price_optimizer() -> str:
    from tools.price_optimizer import optimize_shopping_list
    plan = optimize_shopping_list(["milk", "eggs", "bananas"])
    if plan["total_cost"] <= 0 or not plan["store_ids"]:
        raise RuntimeError(f"unexpected plan: {plan}")
    return f"total=${plan['total_cost']} across {len(plan['store_ids'])} stores"


def check_route_planner() -> str:
    from tools.price_optimizer import optimize_shopping_list
    from tools.route_planner import plan_route
    plan = optimize_shopping_list(["milk", "eggs", "bananas"])
    route = plan_route(plan["store_ids"], plan["stores_meta"])
    if route.get("total_duration_min") is None:
        raise RuntimeError(f"unexpected route: {route}")
    return f"{len(route['ordered_stops'])} stops, {route['total_duration_min']} min"


def check_llm_ping(timeout_s: float = 30.0) -> str:
    from agent.agent import call_llm
    out = call_llm("Reply with exactly one word: OK")
    if not out:
        raise RuntimeError("empty LLM response")
    return f"reply={out[:60]!r}"


def check_llm_parse() -> str:
    from agent.agent import parse_items_from_message
    items = parse_items_from_message("2 lb chicken breast, a gallon of milk")
    if not any("chicken" in (i.get("name") or "").lower() for i in items):
        raise RuntimeError(f"parse missed chicken: {items}")
    return f"parsed {len(items)} items, e.g. {items[0]}"


def check_agent_full_chain() -> str:
    """Drive the state machine through CLARIFY → CONFIRM → EXECUTE."""
    from agent.agent import ShoppingSession, chat
    session = ShoppingSession()
    script = [
        "I need milk, eggs, and 1 lb bananas",
        "no preferences, looks good",   # ends CLARIFY, moves to CONFIRM
        "yes",                          # confirms → EXECUTE
    ]
    states_seen: list[str] = []
    final_reply = ""
    for msg in script:
        final_reply = chat(session, msg)
        states_seen.append(session.state)

    if session.state != "EXECUTE":
        raise RuntimeError(f"expected EXECUTE, got trace={states_seen}")
    if session.shopping_plan is None or session.route_plan is None:
        raise RuntimeError("tools did not run (shopping_plan / route_plan missing)")

    total = session.shopping_plan.get("total_cost")
    stops = len(session.route_plan.get("ordered_stops") or [])
    return f"states={states_seen} | plan total=${total} across {stops} stops"


# --------------- main ---------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-network", action="store_true", help="skip any step needing LLM or internet")
    args = parser.parse_args()

    print("=== Grocery Agent Pipeline Doctor ===\n")

    print("[1] Environment & config")
    r1 = run_step("load .env + settings", check_settings)
    r2 = run_step("mock data files present", check_mock_data)

    print("\n[2] Cache layer")
    r3 = run_step("cache save/load roundtrip (tmp)", check_cache_roundtrip)
    r4 = run_step("trader_joes cache populated", check_tj_cache_exists)

    print("\n[3] Pure tools (offline)")
    r5 = run_step("product_search against TJ cache", check_product_search, skip_if_failed=[r4])
    r6 = run_step("price_optimizer (mock)", check_price_optimizer, skip_if_failed=[r2])
    r7 = run_step("route_planner (mock matrix)", check_route_planner, skip_if_failed=[r2])

    llm_steps: list[StepResult] = []
    if not args.skip_llm and not args.skip_network:
        print("\n[4] LLM smoke")
        r8 = run_step("LLM ping", check_llm_ping, skip_if_failed=[r1])
        r9 = run_step("LLM parse_items", check_llm_parse, skip_if_failed=[r8])
        llm_steps = [r8, r9]

        print("\n[5] Agent end-to-end")
        run_step("scripted conversation → EXECUTE", check_agent_full_chain,
                 skip_if_failed=[r8, r6, r7])
    else:
        print("\n[4] LLM + agent skipped (--skip-llm / --skip-network)")

    # summary
    all_steps = [r1, r2, r3, r4, r5, r6, r7, *llm_steps]
    failed = [s for s in all_steps if s.ok is False]
    print("\n--- summary ---")
    print(f"  passed: {sum(1 for s in all_steps if s.ok is True)}")
    print(f"  failed: {len(failed)}")
    print(f"  skipped: {sum(1 for s in all_steps if s.ok is None)}")
    if failed:
        print("  first failure:", failed[0].name, "→", failed[0].detail)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
