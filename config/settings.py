# ============================================================
# config/settings.py
# Central configuration for all API keys and app settings.
# Copy this file to config/settings_local.py and fill in your
# real API keys. Never commit settings_local.py to git.
# ============================================================

import os

# --- LLM ---
# Gemma 3 via Google GenAI SDK (same as your calendar agent)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_GENAI_KEY")
LLM_MODEL = "gemma-4-26b-a4b-it"

# --- Routing ---
# OpenRouteService (free tier: 2000 req/day, no credit card needed)
# Sign up at https://openrouteservice.org/dev/#/signup
ORS_API_KEY = os.getenv("ORS_API_KEY", "YOUR_ORS_KEY")
ORS_BASE_URL = "https://api.openrouteservice.org"

# --- Data ---
# Use mock data when True (no real scraping or API calls)
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "true").lower() == "true"
MOCK_DATA_DIR = "data/"

# Price cache TTL in seconds (1 hour by default)
PRICE_CACHE_TTL = 3600

# --- User home location (Pittsburgh default for testing) ---
# This is the starting point for route planning
HOME_ADDRESS = "4800 Forbes Ave, Pittsburgh, PA 15213"  # Carnegie Mellon University
HOME_LAT = 40.4444
HOME_LNG = -79.9431

# --- Routing preferences ---
# Max number of stores to visit in one trip (keep TSP tractable)
MAX_STORES_PER_TRIP = 4

# Cost per mile driven (used to weigh savings vs detour distance)
# Based on IRS standard mileage rate
COST_PER_MILE = 0.21

# --- Errand runner ---
# Base fee for errand runner service (USD)
ERRAND_BASE_FEE = 5.00
# Per-store surcharge
ERRAND_PER_STORE_FEE = 2.00