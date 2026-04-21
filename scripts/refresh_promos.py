"""
Refresh data/promos.json from the existing per-store price caches.

Run:
    uv run python scripts/refresh_promos.py [--dry-run]

Writes atomically to ``data/promos.json``. Safe to run repeatedly; if the
underlying per-store caches are stale, the output is stale too — this
script doesn't re-scrape, it just extracts promo rows from what's already
on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the project importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.promos import build_all_promos, save_promos, PROMOS_CACHE_PATH  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild data/promos.json from per-store caches.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print a summary, don't write.")
    ap.add_argument("--stores", nargs="*", default=None,
                    help="Restrict output summary to these store_ids.")
    args = ap.parse_args()

    data = build_all_promos()
    total = data.get("total_promos", 0)

    print(f"Generated at : {data.get('generated_at')}")
    print(f"Total promos : {total}")
    stores = data.get("stores") or {}
    for sid in sorted(stores):
        if args.stores and sid not in args.stores:
            continue
        rows = stores[sid] or []
        snap = (data.get("source_snapshots") or {}).get(sid) or "?"
        print(f"  {sid:<28}  {len(rows):>5}  (snapshot: {snap})")
        for r in rows[:3]:
            sale = r.get("sale_price")
            reg = r.get("reg_price")
            pct = r.get("discount_pct")
            tag = (
                f"${sale:.2f} was ${reg:.2f} ({pct}% off)"
                if reg and pct is not None
                else f"${sale:.2f}"
            )
            print(f"      - {r.get('item_name', '')[:70]}  {tag}")

    if args.dry_run:
        print("\n(--dry-run; no file written)")
        return 0

    path = save_promos(data)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
