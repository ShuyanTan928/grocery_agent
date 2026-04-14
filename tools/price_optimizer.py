# ============================================================
# tools/price_optimizer.py
# Given a shopping list, finds the cheapest store for each item
# and returns an optimized buy plan that minimizes total cost.
#
# Data format (mock_prices.json):
# {
#   "items": {
#     "pork": [
#       {"store": "trader joe's", "location": "...", "item_name": "...", "item_price": 7.49},
#       ...
#     ]
#   }
# }
# ============================================================

import json
from pathlib import Path
from config.settings import MOCK_DATA_DIR, USE_MOCK_DATA


def load_prices() -> dict:
    """Load product price data from mock file or live scraper."""
    if USE_MOCK_DATA:
        path = Path(MOCK_DATA_DIR) / "mock_prices.json"
        with open(path) as f:
            return json.load(f)
    else:
        raise NotImplementedError("Live scraper not implemented yet")


def load_stores() -> dict:
    """
    Load store metadata indexed by store id.
    Also builds a display_name -> id lookup for matching price data.
    Returns: { store_id: store_dict }
    """
    path = Path(MOCK_DATA_DIR) / "mock_stores.json"
    with open(path) as f:
        data = json.load(f)
    return {s["id"]: s for s in data["stores"]}


def build_display_name_index(stores: dict) -> dict:
    """
    Build a reverse lookup: display_name (lowercase) -> store_id.
    e.g. "trader joe's" -> "trader_joes_shadyside"
    Used to map the store name strings in price data back to store IDs.
    """
    return {s["display_name"].lower(): sid for sid, s in stores.items()}


def find_cheapest(item_query: str, price_data: dict) -> dict | None:
    """
    Find the cheapest store entry for a given item query.
    Matches item_query against the category keys in price_data["items"]
    using simple substring matching.

    Returns the cheapest entry dict:
      {"store": "aldi", "location": "...", "item_name": "...", "item_price": 3.19}
    or None if no match found.
    """
    query_lower = item_query.lower()
    items_db = price_data.get("items", {})

    # Try to find a category whose key is contained in or contains the query
    matched_entries = None
    for category, entries in items_db.items():
        if category in query_lower or query_lower in category:
            matched_entries = entries
            break

    if matched_entries is None:
        return None

    # Return the entry with the lowest item_price
    return min(matched_entries, key=lambda e: e["item_price"])


def optimize_shopping_list(items: list[str]) -> dict:
    """
    Main entry point for the price optimizer tool.

    Takes a list of item strings, finds the cheapest store for each,
    and groups them into a per-store buy plan.

    Returns:
    {
      "plan": {
        "aldi_greenfield": [
          {"item": "Whole Milk 1 Gallon", "price": 3.19, "store_display": "aldi"},
          ...
        ],
        ...
      },
      "total_cost": 12.45,
      "not_found": ["mystery item"],
      "store_ids": ["aldi_greenfield", "trader_joes_shadyside"],
      "stores_meta": { store_id: store_dict, ... }
    }
    """
    price_data = load_prices()
    stores = load_stores()
    name_index = build_display_name_index(stores)

    plan: dict[str, list] = {}
    not_found: list[str] = []
    total_cost = 0.0

    for item in items:
        result = find_cheapest(item, price_data)

        if result is None:
            not_found.append(item)
            continue

        # Map store display name -> store_id
        store_display = result["store"].lower()
        store_id = name_index.get(store_display)

        if store_id is None:
            # Store in price data has no matching entry in stores metadata
            not_found.append(item)
            continue

        if store_id not in plan:
            plan[store_id] = []

        plan[store_id].append({
            "item": result["item_name"],
            "price": result["item_price"],
            "store_display": result["store"],
        })
        total_cost += result["item_price"]

    return {
        "plan": plan,
        "total_cost": round(total_cost, 2),
        "not_found": not_found,
        "store_ids": list(plan.keys()),
        "stores_meta": {sid: stores[sid] for sid in plan if sid in stores},
    }


def get_all_prices_for_item(item_query: str) -> list[dict]:
    """
    Return all store prices for a given item, sorted cheapest first.
    Useful for displaying price comparison to the user.
    """
    price_data = load_prices()
    query_lower = item_query.lower()
    items_db = price_data.get("items", {})

    for category, entries in items_db.items():
        if category in query_lower or query_lower in category:
            return sorted(entries, key=lambda e: e["item_price"])

    return []