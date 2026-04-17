#!/usr/bin/env python3
"""Batch evaluation: run the recommender across ~15 representative queries
and emit a compact summary table so we can eyeball regressions.

What it tracks per query:
  - candidate count from real caches
  - top-1 pick (store / price / item / one-line reason)
  - search-ms vs LLM-ms
  - any "no relevant matches" misses

Usage:
  uv run python scripts/eval_recommendations.py
  uv run python scripts/eval_recommendations.py --topk 3
  uv run python scripts/eval_recommendations.py --json eval_out.json
  uv run python scripts/eval_recommendations.py --queries pork "ground beef" milk
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from tools.recommender import recommend_for_query  # noqa: E402


# Representative coverage: meat, produce, dairy, pantry, snacks, drinks,
# specialty, plus an intentionally tricky one (pork — the original "pork
# & beans crowding out fresh pork" failure mode).
DEFAULT_QUERIES: list[str] = [
    "pork",
    "ground beef",
    "chicken breast",
    "salmon",
    "milk",
    "eggs",
    "greek yogurt",
    "shredded cheese",
    "bananas",
    "spinach",
    "pasta",
    "olive oil",
    "tortilla chips",
    "sparkling water",
    "ice cream",
]


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def run_one(query: str, topk: int, max_candidates: int) -> dict:
    """Wrap recommend_for_query and capture wall-clock total separately."""
    t0 = time.perf_counter()
    res = recommend_for_query(query, topk=topk, max_candidates=max_candidates)
    res["wall_ms"] = (time.perf_counter() - t0) * 1000
    return res


def print_table(rows: list[dict]) -> None:
    """Render a fixed-width summary table."""
    header = (
        f"{'query':<18}{'cands':>6}{'srch ms':>9}{'llm ms':>9}"
        f"{'wall ms':>9}  {'top-1 store':<13}{'price':>8}  top-1 item"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        picks = r.get("picks") or []
        if picks:
            c = picks[0]["candidate"]
            top_store = c["store"]
            top_price = f"${c['price']:.2f}"
            top_name = _truncate(c["name"], 50)
        else:
            top_store = "-"
            top_price = "-"
            top_name = "(no relevant match)"
        t = r["timings"]
        print(
            f"{_truncate(r['query'], 18):<18}"
            f"{len(r.get('candidates') or []):>6}"
            f"{t['search_ms']:>9.0f}"
            f"{t['llm_ms']:>9.0f}"
            f"{r['wall_ms']:>9.0f}  "
            f"{top_store:<13}"
            f"{top_price:>8}  "
            f"{top_name}"
        )
    print(sep)


def print_picks_detail(rows: list[dict], topk: int) -> None:
    """One block per query showing the full top-K with reasons + summary."""
    for r in rows:
        print()
        print(f"### {r['query']!r}  (top-{topk})")
        picks = r.get("picks") or []
        if not picks:
            print(f"    (no picks)  -- {r.get('summary','')}")
            continue
        for p in picks:
            c = p["candidate"]
            size = f" [{c['size']}]" if c.get("size") else ""
            print(f"  #{p['rank']} {c['store']:<12} ${c['price']:>6.2f}  "
                  f"{c['name']}{size}")
            if p.get("reason"):
                print(f"        → {p['reason']}")
        if r.get("summary"):
            print(f"    summary: {r['summary']}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--topk", type=int, default=3,
                   help="picks requested per query (default 3)")
    p.add_argument("--candidates", type=int, default=30,
                   help="max candidates sent to LLM per query (default 30)")
    p.add_argument("--queries", nargs="+", default=None,
                   help="custom query list (defaults to a 15-item suite)")
    p.add_argument("--json", dest="json_out", default=None,
                   help="dump full structured results to this JSON path")
    p.add_argument("--no-detail", action="store_true",
                   help="only print the summary table, skip per-query picks")
    args = p.parse_args()

    queries = args.queries or DEFAULT_QUERIES
    print("=" * 72)
    print(f"Eval: {len(queries)} queries  topk={args.topk}  "
          f"candidates<={args.candidates}")
    print("=" * 72)

    rows: list[dict] = []
    overall_start = time.perf_counter()
    for i, q in enumerate(queries, 1):
        print(f"  [{i:>2}/{len(queries)}] {q}", flush=True)
        try:
            res = run_one(q, args.topk, args.candidates)
        except Exception as exc:
            res = {
                "query": q, "topk": args.topk, "candidates": [], "picks": [],
                "summary": f"ERROR: {exc}",
                "timings": {"search_ms": 0.0, "llm_ms": 0.0},
                "wall_ms": 0.0, "raw_llm": "",
            }
        rows.append(res)
    overall_ms = (time.perf_counter() - overall_start) * 1000

    print()
    print_table(rows)

    if not args.no_detail:
        print_picks_detail(rows, args.topk)

    # aggregate stats
    misses = [r for r in rows if not r.get("picks")]
    avg_search = sum(r["timings"]["search_ms"] for r in rows) / max(1, len(rows))
    avg_llm = sum(r["timings"]["llm_ms"] for r in rows) / max(1, len(rows))
    print()
    print("=" * 72)
    print(f"Overall: {len(rows)} queries in {overall_ms / 1000:.1f}s")
    print(f"  avg search:  {avg_search:.1f} ms")
    print(f"  avg LLM:     {avg_llm:.1f} ms")
    print(f"  misses:      {len(misses)}"
          + (f" ({', '.join(m['query'] for m in misses)})" if misses else ""))
    print("=" * 72)

    if args.json_out:
        # strip raw_llm to keep the file readable; keep everything else
        out = []
        for r in rows:
            out.append({**r, "raw_llm": r.get("raw_llm", "")[:500]})
        Path(args.json_out).write_text(
            json.dumps(out, ensure_ascii=False, indent=2)
        )
        print(f"\nWrote structured results to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
