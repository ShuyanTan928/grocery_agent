# ============================================================
# scripts/sync_mock_from_cache.py
# Overwrite Trader Joe's entries in data/mock_prices.json with the
# closest real SKU from data/price_cache/trader_joes_shadyside.json.
#
# Per-category "recipe" below picks a narrow filter so we don't end
# up replacing "Organic Whole Milk 1 Gallon" with a yogurt cup.
#
# Usage:
#   uv run python scripts/sync_mock_from_cache.py          # dry-run
#   uv run python scripts/sync_mock_from_cache.py --apply  # write changes
# ============================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config.settings import MOCK_DATA_DIR, PRICE_CACHE_DIR


TJ_STORE_DISPLAY = "trader joe's"
TJ_CACHE_FILE = "trader_joes_shadyside.json"


# Per-category recipes. Each recipe filters cached TJ items down to a
# plausible equivalent of the mock category and picks the cheapest match.
#   required_any:  item_name must contain at least one of these tokens
#   required_all:  item_name must contain ALL of these tokens
#   forbidden:     reject item if any of these tokens appear
#   location:      for eggs/pork/etc. we may want "Dairy"/"Meat" etc., unused for now
# All matching is case-insensitive on item_name.
RECIPES: dict[str, dict] = {
    "milk": {
        "required_all": ["milk"],
        "forbidden": [
            "yogurt", "chocolate", "cereal", "latte", "creamer", "ice cream",
            "oatmilk", "oat milk", "almond", "soy", "coconut", "cookie", "butter",
            "smoothie", "shake", "kefir", "bar", "cake", "pudding",
            "cheese", "ricotta", "mozzarella", "cottage",
            "lactose free", "a2", "gruy",
        ],
        # TJ online catalog doesn't carry a plain whole-gallon SKU; the closest
        # is the ultra-filtered 59 fl oz bottle. Prefer that when present.
        "prefer_tokens": ["ultra-filtered", "whole milk", "organic"],
    },
    "eggs": {
        "required_all": ["egg"],
        "forbidden": [
            "chocolate", "cookie", "marshmallow", "truffle", "yogurt", "nog",
            "eggplant", "cake", "pasta", "noodle",
        ],
        "prefer_tokens": ["large", "free range", "organic"],
    },
    "bread": {
        "required_all": ["bread"],
        "forbidden": [
            "gingerbread", "cornbread", "shortbread", "pudding", "crumb",
            "flatbread", "breadcrumb", "crisp", "stuffing",
        ],
        "prefer_tokens": ["sandwich", "sliced", "loaf"],
    },
    "chicken": {
        "required_all": ["chicken", "breast"],
        "forbidden": ["broth", "stock", "soup", "flavored"],
        "prefer_tokens": ["boneless", "skinless"],
    },
    "pork": {
        "required_any": ["pork chop", "pork loin chops", "pork loin chop", "pork tenderloin"],
        "forbidden": [
            "rinds", "bao", "dumpling", "rice", "sausage", "hot dog", "bacon",
            "pepperoni",
        ],
        "prefer_tokens": ["chop", "loin"],
    },
    "bananas": {
        "required_all": ["banana"],
        "forbidden": [
            "chocolate", "fritter", "bread", "cake", "muffin", "chip",
            "sauce", "crusher", "crisp", "gone bananas",
        ],
        # Prefer whole fresh bananas (no other descriptors)
        "prefer_tokens": ["bananas 1", "organic bananas"],
    },
    "orange juice": {
        "required_all": ["orange", "juice"],
        "forbidden": ["sparkling", "soda", "marmalade", "tea", "spritz", "sauce"],
        "prefer_tokens": ["100%", "fl oz"],
    },
    "pasta": {
        "required_all": ["pasta"],
        "forbidden": [
            "sauce", "salad", "mac & cheese", "macaroni", "ravioli", "lasagna",
            "gnocchi",
        ],
        "prefer_tokens": ["spaghetti", "penne", "organic", "1 lb"],
    },
    "butter": {
        "required_all": ["butter"],
        "forbidden": [
            "peanut", "almond", "cashew", "sunflower", "cookie", "cake",
            "brownie", "granola", "bar", "caramel", "cup", "biscuit", "shortbread",
            "spread", "chocolate",
        ],
        "prefer_tokens": ["unsalted", "salted", "cultured", "1 lb", "sticks"],
    },
}


def load_cache_items(cache_path: Path) -> list[dict]:
    with cache_path.open() as f:
        data = json.load(f)
    return data.get("items") or []


def pick_tj_entry(category: str, recipe: dict, cache_items: list[dict]) -> dict | None:
    req_all = [t.lower() for t in recipe.get("required_all", [])]
    req_any = [t.lower() for t in recipe.get("required_any", [])]
    forbidden = [t.lower() for t in recipe.get("forbidden", [])]
    prefer = [t.lower() for t in recipe.get("prefer_tokens", [])]

    def score(name_lower: str) -> int:
        return sum(1 for p in prefer if p in name_lower)

    candidates: list[tuple[int, float, dict]] = []
    for item in cache_items:
        name = (item.get("item_name") or "").lower()
        if not name:
            continue
        price = item.get("item_price")
        if not isinstance(price, (int, float)) or price <= 0:
            continue

        if req_all and not all(t in name for t in req_all):
            continue
        if req_any and not any(t in name for t in req_any):
            continue
        if forbidden and any(t in name for t in forbidden):
            continue

        candidates.append((score(name), price, item))

    if not candidates:
        return None

    # Highest preference score wins; ties broken by cheapest price.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]


def sync(apply_changes: bool) -> int:
    mock_path = Path(MOCK_DATA_DIR) / "mock_prices.json"
    cache_path = Path(PRICE_CACHE_DIR) / TJ_CACHE_FILE

    if not cache_path.exists():
        print(f"ERR: Trader Joe's cache not found at {cache_path}", file=sys.stderr)
        print("     Run `uv run python -m tools.refresh_prices --store trader_joes` first.",
              file=sys.stderr)
        return 2

    cache_items = load_cache_items(cache_path)
    if not cache_items:
        print(f"ERR: Trader Joe's cache at {cache_path} is empty.", file=sys.stderr)
        return 2

    with mock_path.open() as f:
        mock = json.load(f)

    changed = 0
    skipped_no_match: list[str] = []
    skipped_no_recipe: list[str] = []

    for category, entries in mock.get("items", {}).items():
        recipe = RECIPES.get(category)
        tj_entry = next(
            (e for e in entries if e.get("store", "").strip().lower() == TJ_STORE_DISPLAY),
            None,
        )
        if tj_entry is None:
            continue

        if recipe is None:
            skipped_no_recipe.append(category)
            continue

        picked = pick_tj_entry(category, recipe, cache_items)
        if picked is None:
            skipped_no_match.append(category)
            print(f"  [SKIP]  [{category:<13}]  no confident TJ match; "
                  f"keeping mock '{tj_entry['item_name']}' @ ${tj_entry['item_price']}")
            continue

        old_name = tj_entry.get("item_name")
        old_price = tj_entry.get("item_price")
        new_name = picked["item_name"]
        new_price = picked["item_price"]

        print(f"  [SYNC]  [{category:<13}]  '{old_name}' @ ${old_price}")
        print(f"           -> '{new_name}' @ ${new_price}")

        tj_entry["item_name"] = new_name
        tj_entry["item_price"] = new_price
        # Optional extras if we have them
        if picked.get("url"):
            tj_entry["url"] = picked["url"]
        if picked.get("location"):
            tj_entry["location"] = picked["location"]
        changed += 1

    print()
    print(f"Summary: {changed} entry(ies) would change; "
          f"no-match categories: {skipped_no_match or '(none)'}; "
          f"no-recipe categories: {skipped_no_recipe or '(none)'}")

    if not apply_changes:
        print("Dry-run only. Re-run with --apply to write changes to mock_prices.json.")
        return 0

    with mock_path.open("w") as f:
        json.dump(mock, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote updates to {mock_path}.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Sync TJ entries in mock_prices from the real cache.")
    p.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = p.parse_args()
    sys.exit(sync(args.apply))


if __name__ == "__main__":
    main()
