#!/usr/bin/env python3
# ============================================================
# scripts/chat.py
# Quick-start REPL for the grocery shopping agent.
#
# The agent is now driven by an LLM tool-calling loop (see
# agent/loop.py); routing decisions are made by the orchestrator LLM
# looking at AgentState each turn. No more regex side-flows / state
# machine, so most of the old CLI flags are gone.
#
# Examples
# --------
#   # minimal - uses .env + defaults
#   uv run python scripts/chat.py
#
#   # pick a different model (slug conventions differ by provider)
#   uv run python scripts/chat.py --provider openrouter --model openai/gpt-4o-mini
#   uv run python scripts/chat.py --provider google     --model gemini-2.5-pro
#
#   # show per-turn tool-call trace (helpful when debugging weird replies)
#   uv run python scripts/chat.py --show-trace
#
#   # cap max steps per turn (default 8)
#   uv run python scripts/chat.py --max-steps 4
#
#   # replay a canned script (one message per line, # = comment)
#   uv run python scripts/chat.py --replay demos/canned.txt --verbose
#
# REPL commands (typed at the `>` prompt)
# ---------------------------------------
#   /exit, /quit, :q    Leave the REPL
#   /reset              Start a fresh session
#   /state              Print the session state (JSON)
#   /plan               Pretty-print the latest shopping plan
#   /trace              Show the tool-call trace from the last turn
#   /flags              Show the runtime config
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

--model              Orchestrator LLM id. Used for every tool-choice step.
                     env: LLM_MODEL.
                     recommended defaults:
                       gemini-2.5-flash              (Google GenAI)
                       google/gemini-2.5-flash       (OpenRouter)
                     fallback if JSON parse flakes:
                       openai/gpt-4o-mini            (OpenRouter)
                       gemini-2.5-pro                (Google GenAI)

--max-steps N        Hard cap on tool-calls per user turn (default 8).
                     env: MAX_AGENT_STEPS.

--show-trace         After each reply, dump the tool-call trace for that
                     turn (one line per step: tool, args, observation).

--llm-main-optimizer / --no-llm-main-optimizer
                     Leaf-tool flag (doesn't affect routing). When on,
                     optimize_shopping_list consults the recommender LLM
                     once per line item. ~N extra LLM calls per plan.
                     env: USE_LLM_MAIN_OPTIMIZER.  default: off.

--dump-state         Print the full session state after every reply.

--verbose, -v        Log per-turn latency.

--replay FILE        Read messages from FILE (one per line, # = comment).
                     After the script ends you drop back into interactive
                     mode unless --exit-after-replay is given.

--exit-after-replay  Exit immediately after --replay finishes.

Interactive commands
--------------------
  /exit, /quit, :q   Leave.
  /reset             Start a fresh session (clears all state).
  /state             Dump the current session state JSON.
  /plan              Pretty-print the latest shopping plan.
  /trace             Show the tool-call trace from the last turn.
  /flags             Effective runtime config.
  /help              This cheatsheet.
""",
    )
    p.add_argument("--provider", choices=["google", "openrouter"])
    p.add_argument("--model", help="Orchestrator LLM id (LLM_MODEL).")
    p.add_argument("--max-steps", type=int, dest="max_steps",
                   help="Hard cap on tool-calls per user turn.")
    p.add_argument("--show-trace", action="store_true",
                   help="Print the tool-call trace after every reply.")
    p.add_argument("--llm-main-optimizer", dest="llm_main_optimizer",
                   action="store_true", default=None,
                   help="Force LLM per-item picks in optimize_shopping_list on.")
    p.add_argument("--no-llm-main-optimizer", dest="llm_main_optimizer",
                   action="store_false",
                   help="Force LLM main optimizer off (cache tier-only).")
    p.add_argument("--dump-state", action="store_true",
                   help="Print full session state after every reply.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Log per-turn latency.")
    p.add_argument("--replay", type=Path,
                   help="Replay messages from FILE (one per line).")
    p.add_argument("--exit-after-replay", action="store_true",
                   help="Exit immediately after --replay finishes.")
    return p


# ──────────────────────────────────────────────────────────────
# Env wiring (runs BEFORE importing agent.agent so settings see it)
# ──────────────────────────────────────────────────────────────


def apply_cli_to_env(args: argparse.Namespace) -> None:
    if args.provider == "openrouter":
        os.environ["USE_OPENROUTER"] = "1"
    elif args.provider == "google":
        os.environ["USE_OPENROUTER"] = "0"
        os.environ.pop("LLM_PROVIDER", None)
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.max_steps is not None:
        os.environ["MAX_AGENT_STEPS"] = str(max(1, args.max_steps))
    if args.llm_main_optimizer is True:
        os.environ["USE_LLM_MAIN_OPTIMIZER"] = "1"
    elif args.llm_main_optimizer is False:
        os.environ["USE_LLM_MAIN_OPTIMIZER"] = "0"


# ──────────────────────────────────────────────────────────────
# REPL helpers
# ──────────────────────────────────────────────────────────────


def render_state(session) -> str:
    return json.dumps(session.to_full_dict(), ensure_ascii=False, indent=2, default=str)


def render_plan(session) -> str:
    plan = session.shopping_plan
    if not plan or not plan.get("plan"):
        return "(no plan yet — finish a shopping list first)"
    lines = [f"Total: ${plan.get('total_cost')}   stores: {len(plan['store_ids'])}"]
    for sid, items in plan["plan"].items():
        lines.append(f"\n  ── {sid} ──")
        for it in items:
            price = it.get("price") or 0
            lines.append(f"     ${price:>6.2f}   {it.get('item')}")
    if plan.get("not_found"):
        lines.append(f"\n  Not found: {plan['not_found']}")
    if plan.get("unfulfilled_preferences"):
        lines.append("\n  Unfulfilled preferences:")
        for u in plan["unfulfilled_preferences"]:
            lines.append(f"     - {u}")
    return "\n".join(lines)


def render_flags() -> str:
    from config.settings import (
        AGENT_LOOP_MODEL, LLM_MODEL, LLM_PROVIDER,
        MAX_AGENT_STEPS,
        USE_LLM_DISH_FALLBACK, USE_LLM_MAIN_OPTIMIZER,
    )
    return json.dumps({
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "agent_loop_model": AGENT_LOOP_MODEL,
        "max_agent_steps": MAX_AGENT_STEPS,
        "llm_main_optimizer": USE_LLM_MAIN_OPTIMIZER,
        "llm_dish_fallback": USE_LLM_DISH_FALLBACK,
    }, indent=2)


def render_trace(trace: list) -> str:
    if not trace:
        return "(no trace — run a message first)"
    lines = []
    for e in trace:
        args_preview = json.dumps(e.args, ensure_ascii=False)[:140]
        obs_preview = json.dumps(e.obs, ensure_ascii=False)[:200]
        lines.append(f"  step {e.step}  {e.tool}({args_preview})")
        lines.append(f"              -> {obs_preview}")
    return "\n".join(lines)


HELP_TEXT = """\
Commands at the `>` prompt
  /exit, /quit, :q   leave
  /reset             start fresh
  /state             dump full session state JSON
  /plan              pretty-print the last plan
  /trace             tool-call trace from the last turn
  /flags             effective runtime config
  /help              this cheatsheet

Anything else is sent to the agent.
"""


def handle_command(cmd, session, new_session, last_trace):
    c = cmd.strip().lower()
    if c in ("/exit", "/quit", ":q"):
        return "EXIT", None
    if c == "/reset":
        return "RESET", new_session()
    if c == "/state":
        return "PRINT", render_state(session)
    if c == "/plan":
        return "PRINT", render_plan(session)
    if c == "/trace":
        return "PRINT", render_trace(last_trace)
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

    from agent.agent import ShoppingSession, chat

    def new_session():
        return ShoppingSession()

    session = new_session()
    last_trace: list = []

    print("Grocery Agent REPL   (type /help for commands, /exit to quit)")
    print(render_flags())
    print()

    messages: list[str] = []
    if args.replay:
        messages = load_replay(args.replay)
        if args.verbose:
            print(f"[replay] {len(messages)} messages from {args.replay}")

    def run_turn(msg: str) -> None:
        nonlocal session, last_trace
        if not msg:
            return
        cmd_kind, cmd_payload = handle_command(msg, session, new_session, last_trace)
        if cmd_kind == "EXIT":
            raise SystemExit(0)
        if cmd_kind == "RESET":
            session = cmd_payload
            last_trace = []
            print("(session reset)")
            return
        if cmd_kind == "PRINT":
            print(cmd_payload)
            return

        trace: list = []
        t0 = time.perf_counter()
        try:
            reply = chat(session, msg, trace=trace)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {type(exc).__name__}: {exc}")
            return
        dt_ms = (time.perf_counter() - t0) * 1000
        last_trace = trace

        print(f"Agent : {reply}")
        if args.verbose:
            print(f"[{dt_ms:>6.0f}ms, {len(trace)} tool call(s)]")
        if args.show_trace and trace:
            print(render_trace(trace))
        if args.dump_state:
            print(render_state(session))

    for m in messages:
        print(f"You   : {m}")
        run_turn(m)

    if args.replay and args.exit_after_replay:
        return 0

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
