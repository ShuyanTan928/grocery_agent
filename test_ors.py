#!/usr/bin/env python3
# ============================================================
# test_ors.py
# Test your OpenRouteService API key and routing functionality.
# Tests geocoding, distance matrix, and directions between
# real Pittsburgh store locations.
#
# Place inside grocery_agent/ and run:
#   uv run python test_ors.py
# ============================================================

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ORS_API_KEY")

if not API_KEY or API_KEY == "YOUR_ORS_KEY":
    print("ERROR: ORS_API_KEY not found in .env")
    print("Get a free key at: https://openrouteservice.org/dev/#/signup")
    sys.exit(1)

print(f"API key loaded: {API_KEY[:8]}...{API_KEY[-4:]}\n")

BASE_URL = "https://api.openrouteservice.org"
HEADERS = {
    "Authorization": API_KEY,
    "Content-Type": "application/json",
}

# Two Pittsburgh stores as test coordinates [lng, lat]
ALDI       = [-79.9558, 40.4442]   # Aldi Greenfield
TRADER_JOE = [-79.9256, 40.4583]   # Trader Joe's Shadyside
WALMART    = [-80.0611, 40.4051]   # Walmart Crafton

# ── Test 1: Distance Matrix ──────────────────────────────────
print("=" * 55)
print("Test 1: Distance Matrix (3 stores)")
print("=" * 55)

try:
    resp = requests.post(
        f"{BASE_URL}/v2/matrix/driving-car",
        headers=HEADERS,
        json={
            "locations": [ALDI, TRADER_JOE, WALMART],
            "metrics": ["duration", "distance"],
            "units": "m",
        },
        timeout=15,
    )

    if resp.status_code == 200:
        data = resp.json()
        print("  Status: OK")
        stores = ["Aldi", "Trader Joe's", "Walmart"]
        print(f"\n  Drive times (minutes):")
        for i, row in enumerate(data["durations"]):
            for j, val in enumerate(row):
                if i != j:
                    print(f"    {stores[i]} -> {stores[j]}: {val/60:.1f} min")
        print(f"\n  Drive distances (km):")
        for i, row in enumerate(data["distances"]):
            for j, val in enumerate(row):
                if i != j:
                    print(f"    {stores[i]} -> {stores[j]}: {val/1000:.2f} km")
    else:
        print(f"  FAIL: {resp.status_code}")
        print(f"  {resp.text[:300]}")

except Exception as e:
    print(f"  FAIL: {e}")

# ── Test 2: Directions (Aldi -> Trader Joe's) ────────────────
print("\n" + "=" * 55)
print("Test 2: Turn-by-turn directions")
print("=" * 55)

try:
    resp = requests.post(
        f"{BASE_URL}/v2/directions/driving-car/json",
        headers=HEADERS,
        json={
            "coordinates": [ALDI, TRADER_JOE],
            "instructions": True,
            "units": "mi",
        },
        timeout=15,
    )

    if resp.status_code == 200:
        data = resp.json()
        route = data["routes"][0]["summary"]
        segments = data["routes"][0]["segments"][0]["steps"]
        print(f"  Status: OK")
        print(f"  Distance: {route['distance']:.2f} miles")
        print(f"  Duration: {route['duration']/60:.1f} minutes")
        print(f"\n  First 3 steps:")
        for step in segments[:3]:
            print(f"    - {step['instruction']}")
    else:
        print(f"  FAIL: {resp.status_code}")
        print(f"  {resp.text[:300]}")

except Exception as e:
    print(f"  FAIL: {e}")

# ── Test 3: Rate limit check ─────────────────────────────────
print("\n" + "=" * 55)
print("Test 3: Account info / rate limit headers")
print("=" * 55)

try:
    resp = requests.post(
        f"{BASE_URL}/v2/matrix/driving-car",
        headers=HEADERS,
        json={"locations": [ALDI, TRADER_JOE], "metrics": ["duration"]},
        timeout=10,
    )
    remaining = resp.headers.get("X-RateLimit-Remaining-Day", "n/a")
    limit     = resp.headers.get("X-RateLimit-Limit-Day", "n/a")
    print(f"  Daily requests remaining: {remaining} / {limit}")
    if remaining != "n/a" and limit != "n/a":
        used = int(limit) - int(remaining)
        print(f"  Used today: {used}")
except Exception as e:
    print(f"  Could not read rate limit headers: {e}")

print("\nDone.")