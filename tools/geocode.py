"""
Lightweight place-name → (lat, lng) resolver for destination waypoints.

The agent lets a user tack on non-shopping stops to a route ("I also need
to swing by CMU before going home"). To route through those stops we need
coordinates. We deliberately avoid a heavy geocoding dependency:

  1. A small curated dict of Pittsburgh landmarks / neighborhoods covers
     the most common asks instantly and works entirely offline. This is
     the only path exercised in mock mode (the default for tests / demo).
  2. When USE_MOCK_DATA is false AND an ORS_API_KEY is configured, we fall
     back to the OpenRouteService geocode/search endpoint. We bias the
     query to Pittsburgh and keep the top hit.
  3. Anything else returns None — the caller is expected to ask the user
     for an address or explicit lat/lng.

Return shape for a hit:
  {"label": <echo of user query>, "address": <display address>,
   "lat": float, "lng": float, "source": "landmark" | "ors"}
"""

from __future__ import annotations

import re
from typing import Any

import requests

from config.settings import ORS_API_KEY, ORS_BASE_URL, USE_MOCK_DATA


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


def _ors_geocode(query: str) -> dict[str, Any] | None:
    """Call ORS geocode/search. Biased to Pittsburgh metro. Returns None
    on any failure (network, no key, no results)."""
    if not ORS_API_KEY or ORS_API_KEY == "YOUR_ORS_KEY":
        return None
    try:
        resp = requests.get(
            f"{ORS_BASE_URL}/geocode/search",
            params={
                "api_key": ORS_API_KEY,
                "text": f"{query}, Pittsburgh, PA",
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
    be resolved from the landmark dict or ORS (which is skipped entirely
    in mock mode)."""
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

    if USE_MOCK_DATA:
        return None

    ors_hit = _ors_geocode(label)
    if ors_hit is None:
        return None
    return {
        "label": label,
        "address": ors_hit["address"],
        "lat": ors_hit["lat"],
        "lng": ors_hit["lng"],
        "source": "ors",
    }
