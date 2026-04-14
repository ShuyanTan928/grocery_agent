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

def plan_route(store_ids: list[str], stores_meta: dict) -> dict:
    """
    Given a list of store IDs to visit, compute the optimal driving
    order starting and ending at the user's home address.

    Returns:
    {
      "ordered_stops": [
        {"store_id": "aldi_greenfield", "name": "Aldi - Greenfield",
         "address": "...", "leg_duration_min": 8, "leg_distance_km": 3.2},
        ...
      ],
      "total_duration_min": 35,
      "total_distance_km": 18.4,
      "ors_directions_url": "https://..."   # deep link to ORS route map
    }
    """
    if len(store_ids) == 0:
        return {"ordered_stops": [], "total_duration_min": 0, "total_distance_km": 0}

    if len(store_ids) == 1:
        # No routing needed — just one store
        store = stores_meta[store_ids[0]]
        return {
            "ordered_stops": [{
                "store_id": store_ids[0],
                "name": f"{store['name']} - {store['branch']}",
                "address": store["address"],
                "leg_duration_min": None,
                "leg_distance_km": None,
            }],
            "total_duration_min": None,
            "total_distance_km": None,
            "ors_directions_url": None,
        }

    # --- Get distance matrix ---
    if USE_MOCK_DATA:
        matrix_data = get_mock_distance_matrix(store_ids)
    else:
        coords = [(stores_meta[sid]["lat"], stores_meta[sid]["lng"]) for sid in store_ids]
        matrix_data = get_ors_distance_matrix(coords, store_ids)

    durations = matrix_data["durations"]
    distances = matrix_data["distances"]
    n = len(store_ids)

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
        store_id = store_ids[idx]
        store = stores_meta[store_id]

        # Leg = travel from previous stop to this one
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

        ordered_stops.append({
            "store_id": store_id,
            "name": f"{store['name']} - {store['branch']}",
            "address": store["address"],
            "leg_duration_min": round(leg_dur / 60, 1) if leg_dur else None,
            "leg_distance_km": round(leg_dist / 1000, 2) if leg_dist else None,
        })

    # Build an ORS directions deep-link for the browser
    waypoints = ";".join(
        f"{stores_meta[store_ids[i]]['lng']},{stores_meta[store_ids[i]]['lat']}"
        for i in best_order
    )
    ors_url = f"https://maps.openrouteservice.org/directions?n1={HOME_LAT}&n2={HOME_LNG}&n3=14&a={waypoints}&b=0&c=0&k1=en-US&k2=km"

    return {
        "ordered_stops": ordered_stops,
        "total_duration_min": round(total_duration_sec / 60, 1),
        "total_distance_km": round(total_distance_m / 1000, 2),
        "ors_directions_url": ors_url,
    }
