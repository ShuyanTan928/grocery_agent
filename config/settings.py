# ============================================================
# config/settings.py
# Central configuration for all API keys and app settings.
# Copy this file to config/settings_local.py and fill in your
# real API keys. Never commit settings_local.py to git.
# ============================================================

import os


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


# --- LLM ---
# Switch: USE_OPENROUTER (default off) → Google GenAI. Set true/on/1/yes to use OpenRouter.
# Back-compat: LLM_PROVIDER=openrouter also selects OpenRouter.
_use_openrouter = _env_truthy("USE_OPENROUTER")
_legacy_openrouter = os.getenv("LLM_PROVIDER", "google").strip().lower() == "openrouter"
LLM_PROVIDER = "openrouter" if (_use_openrouter or _legacy_openrouter) else "google"

# Google GenAI — default path when USE_OPENROUTER is off
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_GENAI_KEY")

# OpenRouter — when USE_OPENROUTER is on (or LLM_PROVIDER=openrouter)
# Docs: https://openrouter.ai/docs/quickstart
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
# Optional; improves attribution on openrouter.ai rankings
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "").strip()

# Model id: Google uses short names (e.g. gemma-4-26b-a4b-it); OpenRouter uses provider/slug (e.g. google/gemma-4-26b-a4b-it)
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4-26b-a4b-it")

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