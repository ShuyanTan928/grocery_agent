# ============================================================
# tools/route_planner.py
# Computes the optimal multi-stop driving route between stores
# using OpenRouteService (free) or mock distance matrix data.
#
# TSP solver: brute-force for <= 6 stops, greedy nearest-neighbor
# for larger sets. Fine for a grocery trip (usually 2-4 stores).
# ============================================================

import json
import math
import itertools
import requests
from pathlib import Path
from config.settings import (
    ORS_API_KEY, ORS_BASE_URL, USE_MOCK_DATA,
    MOCK_DATA_DIR, HOME_LAT, HOME_LNG, HOME_ADDRESS
)


# --------------- Haversine (fallback matrix) ---------------

_EARTH_RADIUS_KM = 6371.0088


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lng) points, in km."""
    lat1, lng1 = a
    lat2, lng2 = b
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def get_haversine_distance_matrix(
    stop_ids: list[str], stops_meta: dict
) -> dict:
    """Build a distance matrix from haversine straight-line distance.

    Used as a universal fallback when we have extra user-requested
    destinations (and therefore can't look up the legs in the precomputed
    mock matrix). Driving time is approximated as 2 min/km of straight-line
    distance — roughly the Pittsburgh urban average after accounting for
    detour factor."""
    n = len(stop_ids)
    durations = [[0.0] * n for _ in range(n)]
    distances = [[0.0] * n for _ in range(n)]
    coords = [
        (stops_meta[sid]["lat"], stops_meta[sid]["lng"]) for sid in stop_ids
    ]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            km = _haversine_km(coords[i], coords[j])
            distances[i][j] = km * 1000.0
            durations[i][j] = km * 120.0  # 2 min/km ≈ 30 km/h urban
    return {"durations": durations, "distances": distances, "store_ids": stop_ids}


# --------------- Distance matrix ---------------

def get_mock_distance_matrix(store_ids: list[str]) -> dict:
    """
    Load precomputed distance matrix from mock file and slice it
    down to just the requested stores.
    """
    path = Path(MOCK_DATA_DIR) / "mock_distance_matrix.json"
    with open(path) as f:
        full = json.load(f)

    order = full["store_order"]

    # Build index map: store_id -> row/col index in the full matrix
    idx = {sid: order.index(sid) for sid in store_ids if sid in order}

    n = len(store_ids)
    durations = [[0.0] * n for _ in range(n)]
    distances = [[0.0] * n for _ in range(n)]

    for i, a in enumerate(store_ids):
        for j, b in enumerate(store_ids):
            if a in idx and b in idx:
                durations[i][j] = full["durations"][idx[a]][idx[b]]
                distances[i][j] = full["distances"][idx[a]][idx[b]]

    return {"durations": durations, "distances": distances, "store_ids": store_ids}


def get_ors_distance_matrix(store_coords: list[tuple], store_ids: list[str]) -> dict:
    """
    Call the ORS Distance Matrix API to get driving time and distance
    between all pairs of stores.

    store_coords: list of (lat, lng) tuples in same order as store_ids
    """
    # ORS expects [lng, lat] order
    locations = [[lng, lat] for lat, lng in store_coords]

    url = f"{ORS_BASE_URL}/v2/matrix/driving-car"
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "locations": locations,
        "metrics": ["duration", "distance"],
        "units": "m",
    }

    response = requests.post(url, json=body, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    return {
        "durations": data["durations"],
        "distances": data["distances"],
        "store_ids": store_ids,
    }


# --------------- TSP solver ---------------

def solve_tsp_brute_force(matrix: list[list], n: int) -> list[int]:
    """
    Brute-force TSP: tries all permutations of store visit order,
    returns the sequence with minimum total travel duration.
    Only used when n <= 6 (720 permutations max).
    """
    best_order = None
    best_cost = float("inf")

    for perm in itertools.permutations(range(n)):
        cost = sum(matrix[perm[i]][perm[i + 1]] for i in range(n - 1))
        if cost < best_cost:
            best_cost = cost
            best_order = list(perm)

    return best_order


def solve_tsp_greedy(matrix: list[list], n: int) -> list[int]:
    """
    Greedy nearest-neighbor TSP: start from store 0 and always
    go to the closest unvisited store next. Fast but not optimal.
    Used as fallback when n > 6.
    """
    visited = [False] * n
    order = [0]
    visited[0] = True

    for _ in range(n - 1):
        last = order[-1]
        nearest = min(
            (j for j in range(n) if not visited[j]),
            key=lambda j: matrix[last][j]
        )
        order.append(nearest)
        visited[nearest] = True

    return order


# --------------- Main entry point ---------------

def _pseudo_destination_id(label: str, index: int) -> str:
    """Stable fake store_id for a user-provided destination waypoint."""
    slug = "".join(ch for ch in (label or "").lower() if ch.isalnum())
    slug = slug[:24] or "dest"
    return f"__dest__{index}__{slug}"


def plan_route(
    store_ids: list[str],
    stores_meta: dict,
    extra_waypoints: list[dict] | None = None,
) -> dict:
    """
    Given a list of store IDs to visit, compute the optimal driving
    order starting and ending at the user's home address.

    `extra_waypoints` lets the caller inject mandatory non-shopping stops
    (e.g. "I also want to drop by CMU on the way"). Each entry must have
    {label, address, lat, lng}. These are added to the TSP alongside the
    stores. When any extras are present we use a haversine-based distance
    matrix (mock mode) or ORS (live mode) to cover the new coordinates.

    Returns:
    {
      "ordered_stops": [
        {"store_id": "aldi_greenfield",
         "kind": "store" | "destination",
         "name": "Aldi - Greenfield",
         "address": "...",
         "leg_duration_min": 8, "leg_distance_km": 3.2},
        ...
      ],
      "total_duration_min": 35,
      "total_distance_km": 18.4,
      "ors_directions_url": "https://...",   # deep link to ORS route map
      "destinations_count": 1
    }
    """
    extras = list(extra_waypoints or [])
    if len(store_ids) == 0 and not extras:
        return {
            "ordered_stops": [],
            "total_duration_min": 0,
            "total_distance_km": 0,
            "destinations_count": 0,
        }

    # Merge stores + extras into a single unified list of stops.
    all_ids: list[str] = list(store_ids)
    combined_meta: dict = dict(stores_meta or {})
    for i, wp in enumerate(extras):
        pid = _pseudo_destination_id(wp.get("label", ""), i)
        combined_meta[pid] = {
            "name": wp.get("label", "Destination"),
            "branch": "(custom stop)",
            "address": wp.get("address") or wp.get("label") or "",
            "lat": float(wp["lat"]),
            "lng": float(wp["lng"]),
            "_is_destination": True,
        }
        all_ids.append(pid)

    if len(all_ids) == 1:
        only = combined_meta[all_ids[0]]
        return {
            "ordered_stops": [{
                "store_id": all_ids[0],
                "kind": "destination" if only.get("_is_destination") else "store",
                "name": (
                    only["name"]
                    if only.get("_is_destination")
                    else f"{only['name']} - {only['branch']}"
                ),
                "address": only["address"],
                "leg_duration_min": None,
                "leg_distance_km": None,
            }],
            "total_duration_min": None,
            "total_distance_km": None,
            "ors_directions_url": None,
            "destinations_count": len(extras),
        }

    # --- Get distance matrix ---
    # When extras are present we can't slice the precomputed mock matrix
    # (it only has stores). Fall through to haversine (mock) or ORS (live).
    if extras:
        if USE_MOCK_DATA:
            matrix_data = get_haversine_distance_matrix(all_ids, combined_meta)
        else:
            coords = [(combined_meta[sid]["lat"], combined_meta[sid]["lng"]) for sid in all_ids]
            matrix_data = get_ors_distance_matrix(coords, all_ids)
    else:
        if USE_MOCK_DATA:
            matrix_data = get_mock_distance_matrix(all_ids)
        else:
            coords = [(combined_meta[sid]["lat"], combined_meta[sid]["lng"]) for sid in all_ids]
            matrix_data = get_ors_distance_matrix(coords, all_ids)

    durations = matrix_data["durations"]
    distances = matrix_data["distances"]
    n = len(all_ids)

    # --- Solve TSP ---
    if n <= 6:
        best_order = solve_tsp_brute_force(durations, n)
    else:
        best_order = solve_tsp_greedy(durations, n)

    # --- Build result ---
    ordered_stops = []
    total_duration_sec = 0.0
    total_distance_m = 0.0

    for i, idx in enumerate(best_order):
        sid = all_ids[idx]
        meta = combined_meta[sid]

        if i > 0:
            prev_idx = best_order[i - 1]
            leg_dur = durations[prev_idx][idx]
            leg_dist = distances[prev_idx][idx]
        else:
            leg_dur = None
            leg_dist = None

        if leg_dur is not None:
            total_duration_sec += leg_dur
            total_distance_m += leg_dist

        is_dest = meta.get("_is_destination", False)
        ordered_stops.append({
            "store_id": sid,
            "kind": "destination" if is_dest else "store",
            "name": meta["name"] if is_dest else f"{meta['name']} - {meta['branch']}",
            "address": meta["address"],
            "leg_duration_min": round(leg_dur / 60, 1) if leg_dur else None,
            "leg_distance_km": round(leg_dist / 1000, 2) if leg_dist else None,
        })

    waypoints = ";".join(
        f"{combined_meta[all_ids[i]]['lng']},{combined_meta[all_ids[i]]['lat']}"
        for i in best_order
    )
    ors_url = f"https://maps.openrouteservice.org/directions?n1={HOME_LAT}&n2={HOME_LNG}&n3=14&a={waypoints}&b=0&c=0&k1=en-US&k2=km"

    return {
        "ordered_stops": ordered_stops,
        "total_duration_min": round(total_duration_sec / 60, 1),
        "total_distance_km": round(total_distance_m / 1000, 2),
        "ors_directions_url": ors_url,
        "destinations_count": len(extras),
    }
