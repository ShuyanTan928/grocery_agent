"""
Lightweight place-name → (lat, lng) resolver for destination waypoints.

The agent lets a user tack on non-shopping stops to a route ("I also need
to swing by CMU before going home"). To route through those stops we need
coordinates. Resolution order:

  1. Optional file cache (data/geocode_cache.json) — ORS hits only, keyed
     by normalized query. Lets street addresses work offline after first
     resolve.
  2. A small curated dict of Pittsburgh landmarks / neighborhoods (always
     offline).
  3. When ORS_API_KEY is set (not the placeholder) and USE_MOCK_GEOCODE is
     false, OpenRouteService geocode/search — **independent of USE_MOCK_DATA**
     so product/route mocks can coexist with real address lookup.
  4. Otherwise None — caller asks for landmark, coords, or a configured key.

Return shape for a hit:
  {"label": <echo of user query>, "address": <display address>,
   "lat": float, "lng": float, "source": "landmark" | "ors" | "geocode_cache"}
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import requests

from config.settings import (
    MOCK_DATA_DIR,
    ORS_API_KEY,
    ORS_BASE_URL,
    USE_MOCK_GEOCODE,
)

# Tests can monkeypatch to a temp file.
GEOCODE_CACHE_OVERRIDE: Path | None = None

_cache_lock = threading.Lock()


PITTSBURGH_LANDMARKS: dict[str, tuple[float, float]] = {
    # Universities
    "cmu": (40.4433, -79.9436),
    "carnegie mellon": (40.4433, -79.9436),
    "carnegie mellon university": (40.4433, -79.9436),
    "pitt": (40.4444, -79.9608),
    "university of pittsburgh": (40.4444, -79.9608),
    "duquesne": (40.4362, -79.9933),
    "duquesne university": (40.4362, -79.9933),
    "chatham": (40.4510, -79.9239),

    # Neighborhoods (centroids)
    "downtown": (40.4406, -79.9959),
    "downtown pittsburgh": (40.4406, -79.9959),
    "oakland": (40.4412, -79.9536),
    "squirrel hill": (40.4367, -79.9229),
    "shadyside": (40.4548, -79.9346),
    "east liberty": (40.4641, -79.9231),
    "lawrenceville": (40.4653, -79.9606),
    "strip district": (40.4528, -79.9833),
    "the strip": (40.4528, -79.9833),
    "south side": (40.4284, -79.9715),
    "north shore": (40.4471, -79.9964),
    "north side": (40.4557, -80.0103),
    "bloomfield": (40.4625, -79.9478),
    "greenfield": (40.4253, -79.9473),
    "highland park": (40.4802, -79.9236),
    "point breeze": (40.4514, -79.9029),

    # Landmarks / venues
    "pittsburgh airport": (40.4915, -80.2329),
    "pit airport": (40.4915, -80.2329),
    "airport": (40.4915, -80.2329),
    "heinz field": (40.4468, -80.0158),
    "acrisure stadium": (40.4468, -80.0158),
    "pnc park": (40.4469, -80.0057),
    "ppg paints arena": (40.4395, -79.9892),
    "pittsburgh zoo": (40.4836, -79.9183),
    "phipps conservatory": (40.4394, -79.9481),
    "carnegie museum": (40.4433, -79.9503),
    "frick park": (40.4369, -79.9029),
    "schenley park": (40.4358, -79.9425),
    "union station": (40.4519, -79.9953),
    "pittsburgh amtrak": (40.4519, -79.9953),
    "greyhound station": (40.4484, -79.9942),
    "the waterfront": (40.4067, -79.8889),
    "waterfront": (40.4067, -79.8889),
    "monroeville mall": (40.4331, -79.7636),
    "ross park mall": (40.5512, -80.0025),

    # Bakery Living — 6480 Living Pl (user-provided; "pi" = Pl typo)
    "bakery living": (40.45601, -79.91707),
    "bakeryliving": (40.45601, -79.91707),
    "6480 living pi": (40.45601, -79.91707),
    "6480 living pl": (40.45601, -79.91707),
    "6480 living place": (40.45601, -79.91707),
    "living pi": (40.45601, -79.91707),
}


def _normalize(query: str) -> str:
    """Lowercase + collapse whitespace; strip punctuation like `'` and `.`."""
    q = (query or "").strip().lower()
    q = re.sub(r"[\"'\.,!?]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _match_landmark(q: str) -> tuple[float, float] | None:
    """Exact key match first, then longest-substring containment either
    way. Returns (lat, lng) or None."""
    if q in PITTSBURGH_LANDMARKS:
        return PITTSBURGH_LANDMARKS[q]
    candidates: list[tuple[int, str, tuple[float, float]]] = []
    for key, coords in PITTSBURGH_LANDMARKS.items():
        if key in q or q in key:
            candidates.append((len(key), key, coords))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][2]


def _effective_cache_path() -> Path:
    return GEOCODE_CACHE_OVERRIDE or Path(MOCK_DATA_DIR) / "geocode_cache.json"


def _cache_read_all() -> dict[str, Any]:
    path = _effective_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _cache_get(normalized: str) -> dict[str, Any] | None:
    with _cache_lock:
        row = _cache_read_all().get(normalized)
    if not isinstance(row, dict):
        return None
    lat, lng = row.get("lat"), row.get("lng")
    if lat is None or lng is None:
        return None
    return {
        "lat": float(lat),
        "lng": float(lng),
        "address": str(row.get("address") or normalized),
        "source": str(row.get("source") or "geocode_cache"),
    }


def _cache_store(
    normalized: str, lat: float, lng: float, address: str, source: str
) -> None:
    path = _effective_cache_path()
    with _cache_lock:
        data = _cache_read_all()
        data[normalized] = {
            "lat": lat,
            "lng": lng,
            "address": address,
            "source": source,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def _ors_key_configured() -> bool:
    k = (ORS_API_KEY or "").strip()
    return bool(k and k != "YOUR_ORS_KEY")


def build_ors_search_text(query: str) -> str:
    """Bias free-form text toward Pittsburgh when the user omits city/state."""
    q = (query or "").strip()
    if not q:
        return q
    lower = q.lower()
    if "pittsburgh" in lower:
        return q
    if re.search(r",\s*pa\b", lower) or re.search(r"\bpa\s+[0-9]{5}\b", lower):
        return q
    return f"{q}, Pittsburgh, PA, USA"


def _ors_geocode(query: str) -> dict[str, Any] | None:
    """Call ORS geocode/search. Biased to Pittsburgh metro. Returns None
    on any failure (network, no key, no results)."""
    if not _ors_key_configured():
        return None
    try:
        resp = requests.get(
            f"{ORS_BASE_URL}/geocode/search",
            params={
                "api_key": ORS_API_KEY,
                "text": build_ors_search_text(query),
                "size": 1,
                "boundary.circle.lat": 40.4406,
                "boundary.circle.lon": -79.9959,
                "boundary.circle.radius": 30,  # km
            },
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception:
        return None
    features = (data or {}).get("features") or []
    if not features:
        return None
    feat = features[0]
    coords = (feat.get("geometry") or {}).get("coordinates") or []
    if len(coords) < 2:
        return None
    lng, lat = float(coords[0]), float(coords[1])
    props = feat.get("properties") or {}
    return {
        "lat": lat,
        "lng": lng,
        "address": props.get("label") or query,
        "source": "ors",
    }


def geocode(query: str) -> dict[str, Any] | None:
    """Resolve a free-form place name to a waypoint dict.

    Returns {label, address, lat, lng, source} or None if the query can't
    be resolved from landmarks, disk cache, or ORS."""
    label = (query or "").strip()
    if not label:
        return None
    normalized = _normalize(label)
    if not normalized:
        return None

    hit = _match_landmark(normalized)
    if hit is not None:
        lat, lng = hit
        return {
            "label": label,
            "address": label,
            "lat": lat,
            "lng": lng,
            "source": "landmark",
        }

    cached = _cache_get(normalized)
    if cached is not None:
        return {
            "label": label,
            "address": cached["address"],
            "lat": cached["lat"],
            "lng": cached["lng"],
            "source": cached["source"],
        }

    if USE_MOCK_GEOCODE or not _ors_key_configured():
        return None

    ors_hit = _ors_geocode(label)
    if ors_hit is None:
        return None
    out = {
        "label": label,
        "address": ors_hit["address"],
        "lat": ors_hit["lat"],
        "lng": ors_hit["lng"],
        "source": "ors",
    }
    _cache_store(
        normalized,
        out["lat"],
        out["lng"],
        out["address"],
        "ors",
    )
    return out
