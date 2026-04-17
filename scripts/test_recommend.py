#!/usr/bin/env python3
"""End-to-end test: user query -> real-data search -> LLM top-K recommend.

Thin CLI over `tools.recommender.recommend_for_query`. The actual logic
(tiered relevance ranking, candidate normalization, LLM call, parsing)
lives in `tools/product_search.py` and `tools/recommender.py` so the
agent and the batch evaluator can share it.

Usage:
  uv run python scripts/test_recommend.py --q pork
  uv run python scripts/test_recommend.py --q "ground beef" --topk 3
  uv run python scripts/test_recommend.py --q yogurt --candidates 60 \\
        --store aldi_greenfield --store trader_joes_shadyside
  uv run python scripts/test_recommend.py --q milk --prefer organic --prefer largest-pack
  uv run python scripts/test_recommend.py --q chicken --prefer brand:Just\\ Bare
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from tools.recommender import (  # noqa: E402
    recommend_for_query,
    render_candidate_block,
    _build_user_prompt,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--q", "--query", dest="query", required=True)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--candidates", type=int, default=40,
                   help="max candidates sent to LLM (default 40)")
    p.add_argument("--store", dest="stores", action="append", default=None,
                   help="restrict to cache store_id (repeatable)")
    p.add_argument("--prefer", dest="preferences", action="append", default=None,
                   help="user preference flag (repeatable). "
                        "Built-ins: cheapest, organic, largest-pack, premium, "
                        "store-brand. Or 'brand:NAME', or any free-form text.")
    p.add_argument("--show-prompt", action="store_true",
                   help="print the full LLM prompt before the call")
    args = p.parse_args()

    print("=" * 72)
    n_stores = len(args.stores) if args.stores else 4
    print(f"[1] Search '{args.query}' across {n_stores} store cache(s)")
    if args.preferences:
        print(f"    preferences: {args.preferences}")
    print("=" * 72)

    result = recommend_for_query(
        args.query,
        topk=args.topk,
        preferences=args.preferences,
        store_ids=args.stores,
        max_candidates=args.candidates,
    )

    cands = result["candidates"]
    print(f"  found {len(cands)} candidates in "
          f"{result['timings']['search_ms']:.1f} ms\n")
    print(render_candidate_block(cands))

    if not cands:
        print("\n[no candidates — nothing to recommend]")
        return 1

    if args.show_prompt:
        print("\n" + "=" * 72)
        print("[2a] Prompt being sent to LLM")
        print("=" * 72)
        print(_build_user_prompt(
            args.query, cands, args.topk, args.preferences,
        ))

    print("\n" + "=" * 72)
    print(f"[2] Asking LLM for top-{args.topk} picks")
    print("=" * 72)
    print(f"  LLM took {result['timings']['llm_ms'] / 1000:.2f}s\n")

    if not result["picks"]:
        print(f"  ! No usable picks. Summary: {result['summary']}")
        if result.get("raw_llm"):
            print(f"  --- raw LLM output ---\n{result['raw_llm']}")
        return 2

    print("=" * 72)
    print(f"[3] Top-{args.topk} recommendation")
    print("=" * 72)
    for pick in result["picks"]:
        c = pick["candidate"]
        size = f" [{c['size']}]" if c["size"] else ""
        print(f"  #{pick['rank']} {c['store']:<12} ${c['price']:>6.2f}  "
              f"{c['name']}{size}")
        print(f"       reason: {pick['reason']}")
    print(f"\n  SUMMARY: {result['summary']}")

    print("\n" + "=" * 72)
    print("[4] Timing")
    print("=" * 72)
    t = result["timings"]
    print(f"  search:  {t['search_ms']:>6.1f} ms")
    print(f"  llm:     {t['llm_ms']:>6.1f} ms")
    print(f"  total:   {t['search_ms'] + t['llm_ms']:>6.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
