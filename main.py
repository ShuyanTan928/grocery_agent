#!/usr/bin/env python3
# ============================================================
# main.py
# Interactive CLI chat loop for the grocery agent.
# Supports multi-turn conversation with clarification and
# preference collection before generating a shopping plan.
#
# Usage: uv run python main.py
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("USE_MOCK_DATA", "true")

from agent.agent import ShoppingSession, chat

BANNER = """
╔══════════════════════════════════════════════════╗
║       Pittsburgh Grocery Agent  🛒               ║
║  Find cheapest prices + plan your route          ║
║  Type 'quit' to exit, 'reset' to start over      ║
╚══════════════════════════════════════════════════╝
"""

WELCOME = "Hi! What groceries do you need today? (type your grocery list below)"


def main():
    print(BANNER)
    session = ShoppingSession()
    print(f"Agent: {WELCOME}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            session = ShoppingSession()
            print(f"\nAgent: {WELCOME}\n")
            continue

        # Get agent response
        print("\nAgent: ", end="", flush=True)
        response = chat(session, user_input)
        print(response)
        print()


if __name__ == "__main__":
    main()