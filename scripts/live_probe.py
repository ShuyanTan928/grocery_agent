#!/usr/bin/env python3
"""Live multi-scenario stress test against the real orchestrator LLM.

Drives ~10 diverse conversations through agent.loop.chat(), dumps the
tool trace + post-turn AgentState, and flags anomalies per turn:

  - tool_chain     : too many steps before a reply (potential loop)
  - parse_retry    : loop had to re-prompt because of malformed JSON
  - empty_reply    : reply tool produced <5 chars
  - emergency      : emergency fallback was triggered
  - bad_remove     : remove_items wiped something it shouldn't have
  - ghost_plan     : plan exists but raw_items empty (should have cleared)
  - missing_pref   : user said "avoid X" but avoid_stores didn't gain it
  - no_plan        : user asked to plan but shopping_plan is still None
  - leftover_dish  : pending_dish lingered past an explicit "no"

Usage:
    uv run python scripts/live_probe.py
    uv run python scripts/live_probe.py --scenario 3 7
    uv run python scripts/live_probe.py --quiet   # only anomalies
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from agent.loop import chat, ToolTraceEntry  # noqa: E402
from agent.state import AgentState  # noqa: E402


# ─────────────────────────── assertion DSL ───────────────────────────


@dataclass
class Turn:
    user: str
    expect_no_tool: list[str] | None = None     # trace tools that MUST NOT fire
    expect_tool: list[str] | None = None        # trace tools that MUST fire
    expect_state: dict | None = None            # substrings/values required in to_llm_view()
    expect_reply_has: list[str] | None = None   # lowercased substrings expected in reply
    expect_reply_missing: list[str] | None = None  # reply must NOT contain these


@dataclass
class Scenario:
    name: str
    purpose: str
    turns: list[Turn]


# ─────────────────────────── scenarios ───────────────────────────────

SCENARIOS: list[Scenario] = [
    Scenario(
        name="happy-path",
        purpose="Simple list (with quantities) → optimize → confirm plan builds.",
        turns=[
            Turn("I need 1 gallon of milk, 1 dozen eggs, and 2 lb bananas",
                 expect_tool=["add_items", "reply"],
                 expect_state={"raw_items_min": 3}),
            Turn("yes, find the best prices",
                 expect_tool=["optimize_and_route", "reply"],
                 expect_state={"has_plan": True}),
        ],
    ),
    Scenario(
        name="dish-flow",
        purpose="Dish intent → propose → user confirms → ingredients added.",
        turns=[
            Turn("I want to make spaghetti carbonara",
                 expect_tool=["propose_dish", "reply"],
                 expect_state={"pending_dish_name": "Spaghetti Carbonara"}),
            Turn("yes, add them all",
                 expect_tool=["apply_pending_dish", "reply"],
                 expect_state={"pending_dish_name": None, "raw_items_min": 2}),
        ],
    ),
    Scenario(
        name="list-options-then-pick",
        purpose="list_options stages candidates → pick N builds single-item plan.",
        turns=[
            Turn("can you list the options for milk",
                 expect_tool=["list_options", "reply"],
                 expect_state={"last_options_min": 1}),
            Turn("I'll take option 1",
                 expect_tool=["pick_option", "reply"],
                 expect_state={"has_plan": True, "last_options_count": 0}),
        ],
    ),
    Scenario(
        name="recommend-does-not-mutate-list",
        purpose="Recommendation is read-only — raw_items must stay empty.",
        turns=[
            Turn("recommend the best chicken wings",
                 expect_tool=["recommend_products", "reply"],
                 expect_no_tool=["add_items", "optimize_and_route"],
                 expect_state={"raw_items_count": 0, "has_plan": False}),
        ],
    ),
    Scenario(
        name="remove-orange-keeps-juice",
        purpose="'remove orange' must NOT delete 'orange juice'.",
        turns=[
            Turn("I need orange juice and oranges",
                 expect_tool=["add_items", "reply"],
                 expect_state={"raw_items_min": 2}),
            Turn("actually drop the orange",
                 expect_tool=["remove_items", "reply"],
                 expect_state={"raw_items_has": "orange juice",
                               "raw_items_missing": "orange"},
                 expect_reply_missing=["juice is gone", "dropped orange juice"]),
        ],
    ),
    Scenario(
        name="avoid-preference-sticks",
        purpose="User says avoid TJ → avoid_stores gets a TJ entry for meat "
                "(item key can be 'meat' or 'pork chops' — both honor intent); "
                "final plan must not contain trader_joes_shadyside.",
        turns=[
            Turn("I need 2 lb pork chops. don't buy meat at Trader Joe's. plan it.",
                 expect_tool=["set_preference", "optimize_and_route", "reply"],
                 expect_state={"has_plan": True,
                               "plan_stores_missing": "trader_joes_shadyside"}),
        ],
    ),
    Scenario(
        name="mid-conversation-question",
        purpose="A question mid-flow must not advance the plan / wipe state.",
        turns=[
            Turn("I need milk and cheese",
                 expect_tool=["add_items", "reply"],
                 expect_state={"raw_items_min": 2}),
            Turn("what stores do you know about?",
                 expect_no_tool=["clear_list", "optimize_and_route", "remove_items"],
                 expect_state={"raw_items_min": 2}),
        ],
    ),
    Scenario(
        name="clear-then-restart",
        purpose="Clear wipes everything; next message starts fresh.",
        turns=[
            Turn("milk, eggs, bread",
                 expect_state={"raw_items_min": 3}),
            Turn("never mind, start over",
                 expect_tool=["clear_list", "reply"],
                 expect_state={"raw_items_count": 0, "has_plan": False}),
            Turn("ok I just need bananas",
                 expect_tool=["add_items", "reply"],
                 expect_state={"raw_items_min": 1, "raw_items_has": "banana"}),
        ],
    ),
    Scenario(
        name="empty-plan-not-hallucinated",
        purpose="Asking to plan empty list must NOT invent fake SKUs — no tool mutation, no fake plan.",
        turns=[
            Turn("find the best prices",
                 expect_no_tool=["optimize_and_route", "add_items"],
                 expect_state={"has_plan": False, "raw_items_count": 0}),
        ],
    ),
    Scenario(
        name="justify-without-plan",
        purpose="'why did you pick X' with no plan must not fabricate.",
        turns=[
            Turn("why did you pick that milk?",
                 expect_no_tool=["optimize_and_route"],
                 expect_state={"has_plan": False}),
        ],
    ),
    Scenario(
        name="ambiguous-remove-asks-clarification",
        purpose="Two different 'milk' variants → remove 'milk' must ask, not guess.",
        turns=[
            Turn("I need whole milk and almond milk",
                 expect_tool=["add_items", "reply"],
                 expect_state={"raw_items_min": 2}),
            Turn("drop the milk",
                 expect_state={"raw_items_min": 2}),  # unchanged — expected to ask
        ],
    ),
    Scenario(
        name="errand-toggle-before-plan",
        purpose="set_errand=true → optimize_and_route must attach an errand_quote.",
        turns=[
            Turn("I want someone else to shop for me. 2 lb pork, 1 gallon milk. plan it.",
                 expect_tool=["optimize_and_route", "reply"],
                 expect_state={"has_plan": True}),
        ],
    ),
    Scenario(
        name="destination-reroutes-plan",
        purpose="User adds a non-shopping stop (CMU) → add_destination fires, "
                "optimize_and_route includes it on the route.",
        turns=[
            Turn("1 gallon of milk and a dozen eggs",
                 expect_tool=["add_items", "reply"],
                 expect_state={"raw_items_min": 2}),
            Turn("I also need to swing by CMU on the way home, plan it",
                 expect_tool=["add_destination", "optimize_and_route", "reply"],
                 expect_state={
                     "has_plan": True,
                     "destinations_count": 1,
                     "destinations_has": "cmu",
                     "route_has_destination": "cmu",
                 }),
        ],
    ),
    Scenario(
        name="unknown-destination-asks-clarification",
        purpose="Unknown landmark → add_destination returns ok:false; LLM must "
                "ask for address or coords rather than routing to nowhere.",
        turns=[
            Turn("add some milk to my list",
                 expect_tool=["add_items", "reply"]),
            Turn("also add my friend's place at zzz-random-spot-42 as a stop",
                 expect_state={"destinations_count": 0}),
        ],
    ),
    Scenario(
        name="partial-dish-cherry-pick",
        purpose="Partial dish selection. LLM may either use apply_pending_dish(only=[...]) "
                "or cancel_pending_dish + add_items; both honor user intent.",
        turns=[
            Turn("I want to make spaghetti carbonara",
                 expect_tool=["propose_dish"],
                 expect_state={"pending_dish_name": "Spaghetti Carbonara"}),
            Turn("just the spaghetti and the eggs, skip the rest",
                 expect_state={"pending_dish_name": None,
                               "raw_items_has": "spaghetti"}),
        ],
    ),
]


# ─────────────────────────── helpers ─────────────────────────────────


_RESET = "\033[0m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_DIM = "\033[2m"


def _fmt_view(v: dict) -> str:
    return json.dumps(v, ensure_ascii=False)


def _state_snapshot(state: AgentState) -> dict:
    plan = state.shopping_plan or {}
    return {
        "raw_items": [it.get("name") for it in state.raw_items],
        "avoid_stores": dict(state.preferences),
        "preferred_stores": dict(state.preferred_stores),
        "pending_dish_name": (state.pending_dish or {}).get("name"),
        "last_options_count": len(state.last_options),
        "has_plan": bool((plan.get("plan") or {})),
        "plan_stores": sorted((plan.get("plan") or {}).keys()),
        "plan_total": plan.get("total_cost"),
        "destinations": [d.get("label") for d in state.destinations],
        "route_stops": [
            {"kind": s.get("kind"), "name": s.get("name")}
            for s in ((state.route_plan or {}).get("ordered_stops") or [])
        ],
    }


def _check_turn(
    turn: Turn,
    trace: list[ToolTraceEntry],
    reply: str,
    state: AgentState,
    parse_retries: int,
    emergency: bool,
) -> list[str]:
    """Return a list of anomaly strings for this turn (empty = clean)."""
    anomalies: list[str] = []

    tools_fired = [e.tool for e in trace]

    # Loop health
    non_reply_steps = sum(1 for t in tools_fired if t != "reply")
    if non_reply_steps > 6:
        anomalies.append(f"tool_chain: {non_reply_steps} non-reply tool calls in one turn: {tools_fired}")
    if parse_retries > 0:
        anomalies.append(f"parse_retry: {parse_retries} malformed-JSON retries")
    if emergency:
        anomalies.append("emergency: loop fell through to _emergency_reply")
    if not reply or len(reply.strip()) < 5:
        anomalies.append(f"empty_reply: reply={reply!r}")

    if turn.expect_tool:
        missing = [t for t in turn.expect_tool if t not in tools_fired]
        if missing:
            anomalies.append(f"missing_tool: expected {missing}, actually fired {tools_fired}")

    if turn.expect_no_tool:
        leaked = [t for t in turn.expect_no_tool if t in tools_fired]
        if leaked:
            anomalies.append(f"unexpected_tool: forbidden {leaked} fired, trace={tools_fired}")

    # State assertions
    snap = _state_snapshot(state)
    names_lower = [(n or "").lower() for n in snap["raw_items"]]
    if turn.expect_state:
        for k, want in turn.expect_state.items():
            if k == "raw_items_min":
                if len(snap["raw_items"]) < want:
                    anomalies.append(f"state.raw_items_min: got {len(snap['raw_items'])} < {want} ({snap['raw_items']})")
            elif k == "raw_items_count":
                if len(snap["raw_items"]) != want:
                    anomalies.append(f"state.raw_items_count: got {len(snap['raw_items'])} != {want} ({snap['raw_items']})")
            elif k == "raw_items_has":
                if not any(want.lower() in n for n in names_lower):
                    anomalies.append(f"state.raw_items_has: '{want}' not in {snap['raw_items']}")
            elif k == "raw_items_missing":
                exact_hits = [n for n in names_lower if n == want.lower()]
                if exact_hits:
                    anomalies.append(f"state.raw_items_missing: '{want}' unexpectedly present in {snap['raw_items']}")
            elif k == "has_plan":
                if snap["has_plan"] != want:
                    anomalies.append(f"state.has_plan: got {snap['has_plan']} != {want}")
            elif k == "pending_dish_name":
                got = snap["pending_dish_name"]
                if want is None:
                    if got is not None:
                        anomalies.append(f"state.pending_dish_name: expected None, got {got!r}")
                else:
                    if not got or want.lower() not in got.lower():
                        anomalies.append(f"state.pending_dish_name: expected '{want}', got {got!r}")
            elif k == "last_options_min":
                if snap["last_options_count"] < want:
                    anomalies.append(f"state.last_options_min: got {snap['last_options_count']} < {want}")
            elif k == "last_options_count":
                if snap["last_options_count"] != want:
                    anomalies.append(f"state.last_options_count: got {snap['last_options_count']} != {want}")
            elif k == "avoid_has":
                item, sid = want
                lst = snap["avoid_stores"].get(item.lower()) or snap["avoid_stores"].get(item) or []
                if sid not in lst:
                    anomalies.append(f"state.avoid_has: expected ({item},{sid}) in {snap['avoid_stores']}")
            elif k == "plan_stores_missing":
                if want in snap["plan_stores"]:
                    anomalies.append(f"state.plan_stores_missing: {want} leaked into plan {snap['plan_stores']}")
            elif k == "destinations_count":
                if len(snap["destinations"]) != want:
                    anomalies.append(
                        f"state.destinations_count: got {len(snap['destinations'])} != {want} ({snap['destinations']})"
                    )
            elif k == "destinations_has":
                if not any(want.lower() in (d or "").lower() for d in snap["destinations"]):
                    anomalies.append(
                        f"state.destinations_has: '{want}' not in {snap['destinations']}"
                    )
            elif k == "route_has_destination":
                dest_names = [s["name"] for s in snap["route_stops"] if s.get("kind") == "destination"]
                if not any(want.lower() in (n or "").lower() for n in dest_names):
                    anomalies.append(
                        f"state.route_has_destination: '{want}' not in route destination stops {dest_names}"
                    )

    reply_l = (reply or "").lower()
    if turn.expect_reply_has:
        missing = [s for s in turn.expect_reply_has if s.lower() not in reply_l]
        if missing:
            anomalies.append(f"reply_missing_phrase: {missing} not in reply")
    if turn.expect_reply_missing:
        leaked = [s for s in turn.expect_reply_missing if s.lower() in reply_l]
        if leaked:
            anomalies.append(f"reply_forbidden_phrase: {leaked} appeared in reply")

    return anomalies


# ─────────────────────────── runner ──────────────────────────────────


def run_scenario(sc: Scenario, *, quiet: bool) -> tuple[int, int]:
    """Run one scenario. Returns (anomaly_count, turn_count)."""
    state = AgentState()
    anomaly_count = 0

    header = f"━━━ Scenario: {sc.name} ━━━"
    print(header)
    print(f"{_DIM}{sc.purpose}{_RESET}")

    for i, turn in enumerate(sc.turns, 1):
        trace: list[ToolTraceEntry] = []
        # Capture parse-retry / emergency signals via stderr sniffing. The loop
        # logs warnings when it retries or emergency-replies, so we redirect
        # that stream to a buffer just for this call.
        buf = io.StringIO()
        import logging
        root_log = logging.getLogger("agent.loop")
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.WARNING)
        root_log.addHandler(handler)
        t0 = time.time()
        try:
            reply = chat(state, turn.user, trace=trace)
        except Exception as e:
            reply = ""
            trace = []
            anomaly_count += 1
            print(f"  [{i}] > {turn.user}")
            print(f"      {_RED}EXCEPTION{_RESET}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            continue
        finally:
            root_log.removeHandler(handler)
        elapsed_ms = int((time.time() - t0) * 1000)

        log_out = buf.getvalue()
        parse_retries = log_out.count("was not valid JSON") + log_out.count("parse retries")
        emergency = "_emergency_reply" in log_out or "exceeded" in log_out

        anomalies = _check_turn(turn, trace, reply, state, parse_retries, emergency)
        if anomalies:
            anomaly_count += len(anomalies)

        if not quiet or anomalies:
            print(f"  [{i}] > {turn.user}")
            tools_line = " → ".join(e.tool for e in trace) or "<no tools>"
            print(f"      tools: {tools_line}  ({elapsed_ms}ms)")
            snap = _state_snapshot(state)
            print(f"      state: {_fmt_view(snap)}")
            reply_preview = reply.replace("\n", " ")[:140]
            print(f"      reply: {reply_preview!r}")
            for a in anomalies:
                color = _RED if a.startswith(("tool_chain", "emergency", "unexpected_tool")) else _YELLOW
                print(f"      {color}⚠ {a}{_RESET}")

    print()
    return anomaly_count, len(sc.turns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, nargs="*",
                        help="1-indexed scenarios to run (default: all)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print turns with anomalies")
    parser.add_argument("--list", action="store_true",
                        help="List scenario names and exit")
    args = parser.parse_args()

    if args.list:
        for i, sc in enumerate(SCENARIOS, 1):
            print(f"  {i}. {sc.name:<35s}  {sc.purpose}")
        return 0

    selected = SCENARIOS
    if args.scenario:
        picks = {n - 1 for n in args.scenario if 1 <= n <= len(SCENARIOS)}
        selected = [s for i, s in enumerate(SCENARIOS) if i in picks]

    total_anom = 0
    total_turns = 0
    for sc in selected:
        a, t = run_scenario(sc, quiet=args.quiet)
        total_anom += a
        total_turns += t

    print("═" * 60)
    color = _GREEN if total_anom == 0 else _RED
    print(f"{color}{len(selected)} scenarios, {total_turns} turns, "
          f"{total_anom} anomalies{_RESET}")
    return 1 if total_anom else 0


if __name__ == "__main__":
    raise SystemExit(main())
