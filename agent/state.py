"""
Universal inner state for the LLM tool-calling agent loop.

`AgentState` is the single source of truth for everything a tool can
read or mutate. The previous CLARIFY / CONFIRM / EXECUTE / DONE state
machine has been deleted — the LLM looks at `to_llm_view()` every turn
and decides what tool to call next.

`ShoppingSession` is kept as an alias so scripts/chat.py, main.py and
the still-live leaf-tool tests don't need to change imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentState:
    raw_items: list[dict] = field(default_factory=list)
    # AVOID: {"chicken": ["trader_joes_shadyside"]}
    preferences: dict[str, list[str]] = field(default_factory=dict)
    # PREFER: {"pork": ["trader_joes_shadyside"]}
    preferred_stores: dict[str, list[str]] = field(default_factory=dict)

    pending_dish: dict | None = None
    last_options: list[dict] = field(default_factory=list)

    shopping_plan: dict | None = None
    route_plan: dict | None = None
    errand_quote: dict | None = None

    want_errand: bool = False

    conversation_history: list[dict] = field(default_factory=list)

    # Legacy attributes — kept so back-compat tests and any caller that
    # still reads them don't blow up. The new loop doesn't read these.
    state: str = "CLARIFY"
    clarification_done: bool = False

    def add_message(self, role: str, text: str) -> None:
        self.conversation_history.append({"role": role, "text": text})

    # ────────────────────────── serialization ─────────────────────────

    def to_llm_view(self) -> dict:
        """Compact summary we pass to the orchestrator LLM each step.

        The goal is to give the model enough to reason about (what's on
        the list? is a plan already built? is there a dish awaiting
        approval?) without flooding it with 200 lines of SKU JSON.
        Tools that need details read them off `state` directly.
        """
        plan = (self.shopping_plan or {}).get("plan") or {}
        plan_summary: dict | None = None
        if plan:
            plan_summary = {
                "store_count": len(plan),
                "item_count": sum(len(v) for v in plan.values()),
                "total_cost": (self.shopping_plan or {}).get("total_cost"),
                "not_found": (self.shopping_plan or {}).get("not_found") or [],
                "stores": [
                    {
                        "store_id": sid,
                        "items": [
                            {
                                "item": e.get("item"),
                                "source_item": e.get("source_item"),
                                "price": e.get("price"),
                            }
                            for e in entries
                        ],
                    }
                    for sid, entries in plan.items()
                ],
            }

        return {
            "raw_items": [
                {
                    "name": it.get("name"),
                    "quantity": it.get("quantity"),
                    "unit": it.get("unit"),
                    "ambiguous": bool(it.get("ambiguous")),
                }
                for it in self.raw_items
            ],
            "avoid_stores": dict(self.preferences),
            "preferred_stores": dict(self.preferred_stores),
            "want_errand": self.want_errand,
            "pending_dish": (
                {
                    "name": self.pending_dish.get("name"),
                    "cuisine": self.pending_dish.get("cuisine"),
                    "ingredient_count": len(self.pending_dish.get("ingredients") or []),
                }
                if self.pending_dish
                else None
            ),
            "last_options_count": len(self.last_options),
            "shopping_plan": plan_summary,
            "has_route_plan": self.route_plan is not None,
            "has_errand_quote": self.errand_quote is not None,
        }

    def to_full_dict(self) -> dict:
        """Full serialization (used by REPL /state debug dump)."""
        return {
            "raw_items": self.raw_items,
            "preferences": self.preferences,
            "preferred_stores": self.preferred_stores,
            "pending_dish": self.pending_dish,
            "last_options": self.last_options,
            "shopping_plan": self.shopping_plan,
            "route_plan": self.route_plan,
            "errand_quote": self.errand_quote,
            "want_errand": self.want_errand,
            "conversation_history": self.conversation_history,
        }


ShoppingSession = AgentState
