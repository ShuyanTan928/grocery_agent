#!/usr/bin/env python3
"""Scripted end-to-end agent conversation (uses real LLM from .env).

Runs a fixed user script through ShoppingSession.chat and prints each turn.
Useful as a regression / wiring check after changing agent or tool code.

    uv run python scripts/agent_chain_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
os.environ.setdefault("USE_MOCK_DATA", "true")

from agent.agent import ShoppingSession, chat  # noqa: E402


SCRIPT = [
    "I need milk, eggs, pork chops, and bananas",
    "2 lb pork, 1 lb bananas, and don't buy meat at Trader Joe's",
    "yes",
]


def main() -> int:
    session = ShoppingSession()
    for i, user_msg in enumerate(SCRIPT, start=1):
        print(f"\n--- Turn {i} ---")
        print(f"You: {user_msg}")
        reply = chat(session, user_msg)
        print(f"Agent: {reply}")
        print(f"[state after: {session.state}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
