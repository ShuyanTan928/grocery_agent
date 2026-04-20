#!/usr/bin/env python3
# ============================================================
# scripts/chat.py
# Quick-start REPL for the grocery shopping agent.
#
# Reads .env from the repo root, builds a ShoppingSession, and drops
# you into a turn-by-turn prompt. Every CLI flag has an environment
# variable equivalent; CLI wins when both are set.
#
# Examples
# --------
#   # minimal — uses .env + defaults (regex router, main LLM model)
#   uv run python scripts/chat.py
#
#   # turn on the hybrid LLM intent router with a small model
#   uv run python scripts/chat.py --use-router --router-model gemini-2.5-flash-lite
#
#   # dry-run with a canned script (one message per line)
#   uv run python scripts/chat.py --replay demos/canned.txt --verbose
#
#   # show per-turn session state after every reply
#   uv run python scripts/chat.py --dump-state
#
# Special REPL commands (typed at the `>` prompt)
# -----------------------------------------------
#   /exit, /quit, :q    Leave the REPL
#   /reset              Start a fresh ShoppingSession
#   /state              Print the current session state + parsed items
#   /plan               Pretty-print the latest shopping_plan (if any)
#   /flags              Show the runtime config (model, router, flag)
#   /help               This help
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chat.py",
        description=(
            "Interactive REPL for the grocery shopping agent. "
            "CLI flags override the matching env vars in .env."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Flag reference
--------------
--provider           LLM provider: 'google' or 'openrouter'.
                     env: USE_OPENROUTER=1 | LLM_PROVIDER=openrouter.
                     default: google.

--model              Main chat LLM id. Used for list/plan/summary generation.
                     env: LLM_MODEL.
                     examples: gemma-4-26b-a4b-it (Google),
                               google/gemma-4-26b-a4b-it (OpenRouter)

--use-router         Turn on the hybrid LLM intent router.
                     When on: regex rules run first; anything they miss
                     is classified by a small model into one of
                     {list_options, recommend, closer, refinement,
                      new_list, confirm_yes, confirm_no, passthrough}.
                     env: USE_LLM_INTENT_ROUTER=1.  default: off.

--router-model       Small model for the intent router (faster/cheaper
                     than the main model). Empty → reuse --model.
                     env: LLM_ROUTER_MODEL.
                     picks: google/gemma-3-1b-it, gemini-2.5-flash-lite,
                            meta-llama/llama-3.2-1b-instruct,
                            mistralai/ministral-3b, openai/gpt-4.1-nano

--router-temp        Sampling temperature for the router only.
                     env: LLM_ROUTER_TEMPERATURE.  default: 0.0.

--llm-list-options / --no-llm-list-options
                     When on (default), "list options" replies are
                     LLM-filtered via the recommender pipeline so brand-
                     name drift (e.g. "lamb" → "Lamb Weston fries") gets
                     pruned; each call costs one extra LLM roundtrip.
                     Turn off for deterministic tier-only results.
                     env: USE_LLM_LIST_OPTIONS.  default: true.

--llm-main-optimizer / --no-llm-main-optimizer
                     When on, optimize_shopping_list runs the recommender
                     LLM once per line item to pick the best SKU from
                     cache candidates, then falls back to tier-only search
                     if the LLM returns nothing. ~N LLM calls per list.
                     env: USE_LLM_MAIN_OPTIMIZER.  default: off.

--dump-state         After every reply, print the session state
                     (current state, parsed items, prefs).

--verbose, -v        Log per-turn timing (ms) and the state transition.

--replay FILE        Read messages from FILE (one per line, blank lines
                     skipped, '#' lines skipped). After the script ends
                     you drop back into interactive mode unless --exit.

--exit-after-replay  Exit immediately after a --replay finishes. Useful
                     for CI / smoke tests.

Interactive commands
--------------------
  /exit, /quit, :q   Leave.
  /reset             Start a fresh ShoppingSession (clears all state).
  /state             Show session state + parsed items + prefs.
  /plan              Pretty-print the last executed shopping_plan.
  /flags             Show effective runtime config.
  /help              This cheatsheet.
""",
    )
    p.add_argument("--provider", choices=["google", "openrouter"])
    p.add_argument("--model", help="Main chat LLM id (LLM_MODEL).")
    p.add_argument("--use-router", action="store_true",
                   help="Enable hybrid LLM intent router.")
    p.add_argument("--router-model",
                   help="Small model id for the intent router.")
    p.add_argument("--router-temp", type=float,
                   help="Router sampling temperature.")
    p.add_argument("--llm-list-options", dest="llm_list_options",
                   action="store_true", default=None,
                   help="Force LLM-filtered list_options on.")
    p.add_argument("--no-llm-list-options", dest="llm_list_options",
                   action="store_false",
                   help="Force LLM-filtered list_options off (tier-only).")
    p.add_argument("--llm-main-optimizer", dest="llm_main_optimizer",
                   action="store_true", default=None,
                   help="Force LLM per-item picks in optimize_shopping_list on.")
    p.add_argument("--no-llm-main-optimizer", dest="llm_main_optimizer",
                   action="store_false",
                   help="Force LLM main optimizer off (cache tier-only).")
    p.add_argument("--dump-state", action="store_true",
                   help="Print session state after every reply.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Log per-turn latency and state changes.")
    p.add_argument("--replay", type=Path,
                   help="Replay messages from FILE (one per line).")
    p.add_argument("--exit-after-replay", action="store_true",
                   help="Exit immediately after --replay finishes.")
    return p


# ──────────────────────────────────────────────────────────────
# Env-var wiring (done BEFORE importing agent.agent so the settings
# module reads the right values at import time).
# ──────────────────────────────────────────────────────────────

def apply_cli_to_env(args: argparse.Namespace) -> None:
    if args.provider == "openrouter":
        os.environ["USE_OPENROUTER"] = "1"
    elif args.provider == "google":
        os.environ["USE_OPENROUTER"] = "0"
        os.environ.pop("LLM_PROVIDER", None)
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.use_router:
        os.environ["USE_LLM_INTENT_ROUTER"] = "1"
    if args.router_model is not None:
        os.environ["LLM_ROUTER_MODEL"] = args.router_model
    if args.router_temp is not None:
        os.environ["LLM_ROUTER_TEMPERATURE"] = str(args.router_temp)
    if args.llm_list_options is True:
        os.environ["USE_LLM_LIST_OPTIONS"] = "1"
    elif args.llm_list_options is False:
        os.environ["USE_LLM_LIST_OPTIONS"] = "0"
    if args.llm_main_optimizer is True:
        os.environ["USE_LLM_MAIN_OPTIMIZER"] = "1"
    elif args.llm_main_optimizer is False:
        os.environ["USE_LLM_MAIN_OPTIMIZER"] = "0"


# ──────────────────────────────────────────────────────────────
# REPL helpers
# ──────────────────────────────────────────────────────────────

def render_state(session) -> str:
    items = [
        (i.get("name"), i.get("quantity"), i.get("unit"))
        for i in (session.raw_items or [])
    ]
    return json.dumps({
        "state": session.state,
        "clarification_done": session.clarification_done,
        "items": items,
        "prefer": session.preferred_stores,
        "avoid": session.preferences,
        "has_plan": bool(session.shopping_plan and session.shopping_plan.get("plan")),
    }, ensure_ascii=False, indent=2)


def render_plan(session) -> str:
    plan = session.shopping_plan
    if not plan or not plan.get("plan"):
        return "(no plan yet — finish a shopping list first)"
    lines = [f"Total: ${plan['total_cost']}   stores: {len(plan['store_ids'])}"]
    for sid, items in plan["plan"].items():
        lines.append(f"\n  ── {sid} ──")
        for it in items:
            lines.append(f"     ${it['price']:>6.2f}   {it['item']}")
    if plan.get("not_found"):
        lines.append(f"\n  Not found: {plan['not_found']}")
    if plan.get("unfulfilled_preferences"):
        lines.append("\n  Unfulfilled preferences:")
        for u in plan["unfulfilled_preferences"]:
            lines.append(f"     - {u}")
    return "\n".join(lines)


def render_flags() -> str:
    # Import lazily so this reflects whatever env was set for this run.
    from config.settings import (
        LLM_MODEL, LLM_PROVIDER,
        LLM_ROUTER_MODEL, LLM_ROUTER_TEMPERATURE,
        USE_LLM_DISH_FALLBACK,
        USE_LLM_INTENT_ROUTER, USE_LLM_LIST_OPTIONS, USE_LLM_MAIN_OPTIMIZER,
    )
    return json.dumps({
        "provider": LLM_PROVIDER,
        "main_model": LLM_MODEL,
        "router_enabled": USE_LLM_INTENT_ROUTER,
        "router_model": LLM_ROUTER_MODEL or "(fallback → main_model)",
        "router_temperature": LLM_ROUTER_TEMPERATURE,
        "llm_list_options": USE_LLM_LIST_OPTIONS,
        "llm_main_optimizer": USE_LLM_MAIN_OPTIMIZER,
        "llm_dish_fallback": USE_LLM_DISH_FALLBACK,
    }, indent=2)


HELP_TEXT = """\
Commands at the `>` prompt
  /exit, /quit, :q   leave
  /reset             start fresh
  /state             session state + items + prefs
  /plan              pretty print the last plan
  /flags             effective runtime config
  /help              this cheatsheet

Anything else is sent to the agent.
"""


def handle_command(cmd: str, session, new_session):
    c = cmd.strip().lower()
    if c in ("/exit", "/quit", ":q"):
        return "EXIT", None
    if c == "/reset":
        return "RESET", new_session()
    if c == "/state":
        return "PRINT", render_state(session)
    if c == "/plan":
        return "PRINT", render_plan(session)
    if c == "/flags":
        return "PRINT", render_flags()
    if c == "/help":
        return "PRINT", HELP_TEXT
    return None, None


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def load_replay(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [ln for ln in (s.strip() for s in lines) if ln and not ln.startswith("#")]


def main() -> int:
    args = build_parser().parse_args()
    apply_cli_to_env(args)

    # Import AFTER apply_cli_to_env so settings pick up the env overrides.
    from agent.agent import ShoppingSession, chat

    def new_session():
        return ShoppingSession()

    session = new_session()

    print("Grocery Agent REPL   (type /help for commands, /exit to quit)")
    print(render_flags())
    print()

    messages: list[str] = []
    if args.replay:
        messages = load_replay(args.replay)
        if args.verbose:
            print(f"[replay] {len(messages)} messages from {args.replay}")

    def run_turn(msg: str) -> None:
        nonlocal session
        if not msg:
            return
        cmd_kind, cmd_payload = handle_command(msg, session, new_session)
        if cmd_kind == "EXIT":
            raise SystemExit(0)
        if cmd_kind == "RESET":
            session = cmd_payload
            print("(session reset)")
            return
        if cmd_kind == "PRINT":
            print(cmd_payload)
            return

        prev_state = session.state
        t0 = time.perf_counter()
        try:
            reply = chat(session, msg)
        except Exception as exc:      # noqa: BLE001
            print(f"[error] {type(exc).__name__}: {exc}")
            return
        dt_ms = (time.perf_counter() - t0) * 1000

        print(f"Agent : {reply}")
        if args.verbose:
            print(f"[{dt_ms:>6.0f}ms] {prev_state} → {session.state}")
        if args.dump_state:
            print(render_state(session))

    # Replay phase
    for m in messages:
        print(f"You   : {m}")
        run_turn(m)

    if args.replay and args.exit_after_replay:
        return 0

    # Interactive phase
    try:
        while True:
            try:
                msg = input("> ").strip()
            except EOFError:
                break
            if not msg:
                continue
            run_turn(msg)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
