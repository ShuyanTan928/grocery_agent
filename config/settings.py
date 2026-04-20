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

# --- Intent router ---
# When True, turns on the hybrid intent router: regex/state-machine runs
# first (fast, deterministic), and any leftover "didn't match" messages
# get classified by a small LLM call into one of a fixed label set
# (list_options | recommend | closer | refinement | new_list | passthrough).
# Default False keeps the pure-regex behavior — cheap and zero extra
# latency.
USE_LLM_INTENT_ROUTER = _env_truthy("USE_LLM_INTENT_ROUTER")

# Dedicated (typically smaller/cheaper/faster) model for intent
# classification. Leave empty to reuse LLM_MODEL. The router is a
# structured-output task with ≤10 labels, so a 1B-3B class model is
# plenty. Good picks:
#   Google GenAI:  gemini-2.5-flash-lite / gemma-3-1b-it
#   OpenRouter:    google/gemma-3-1b-it, meta-llama/llama-3.2-1b-instruct,
#                  mistralai/ministral-3b, openai/gpt-4.1-nano,
#                  google/gemini-2.5-flash-lite
# When unset, routing uses the main LLM_MODEL (slower + pricier).
LLM_ROUTER_MODEL = os.getenv("LLM_ROUTER_MODEL", "").strip()

# Sampling temperature for the router. Classification wants
# determinism — we override the default 0.3 to 0 so labels are stable
# across identical inputs.
LLM_ROUTER_TEMPERATURE = float(os.getenv("LLM_ROUTER_TEMPERATURE", "0.0"))

# --- list_options LLM filtering ---
# When True, handle_list_options_request routes the raw cache hits
# through the recommender's LLM-based relevance pass before showing
# them to the user. This is what stops brand-name drift (e.g. "lamb"
# returning "Lamb Weston fries" at the top). Costs one extra LLM call
# per list_options turn. Set to False to fall back to the pure regex /
# relevance-tier ordering — cheaper but noisier.
USE_LLM_LIST_OPTIONS = os.getenv("USE_LLM_LIST_OPTIONS", "true").strip().lower() in (
    "1", "true", "yes", "on",
)

# --- Routing ---
# OpenRouteService (free tier: 2000 req/day, no credit card needed)
# Sign up at https://openrouteservice.org/dev/#/signup
ORS_API_KEY = os.getenv("ORS_API_KEY", "YOUR_ORS_KEY")
ORS_BASE_URL = "https://api.openrouteservice.org"

# --- Data ---
# Use mock data when True (no real scraping or API calls)
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "true").lower() == "true"
MOCK_DATA_DIR = "data/"

# Price cache TTL in seconds (1 hour by default; refreshers also support per-day cache)
PRICE_CACHE_TTL = 3600

# Directory for per-store scraped price caches (one JSON per store)
PRICE_CACHE_DIR = os.getenv("PRICE_CACHE_DIR", "data/price_cache/")

# --- Trader Joe's ---
# Internal store code (NOT ZIP). Passed as storeCode to their GraphQL API.
# 638 = Pittsburgh Shadyside (6343 Penn Ave).
TRADER_JOES_STORE_CODE = os.getenv("TRADER_JOES_STORE_CODE", "638")

# --- Giant Eagle ---
# Internal store code used by core.shop.gianteagle.com/api/v2.
# 38 = Squirrel Hill (1901 Murray Ave, Pittsburgh PA 15217).
GIANT_EAGLE_STORE_CODE = os.getenv("GIANT_EAGLE_STORE_CODE", "38")

# --- Target ---
# Internal store_id used by redsky.target.com for store-scoped pricing.
# 2757 = East Liberty (6231 Penn Ave, Pittsburgh PA 15206), ~1 mi from CMU.
TARGET_STORE_CODE = os.getenv("TARGET_STORE_CODE", "2757")

# --- Aldi ---
# Aldi runs e-commerce on Instacart Connect; pricing/availability is
# driven by (postal_code, zone_id) rather than a store_code, but we
# keep a nominal store_code for cache/config parity with the others.
# 4061 = Murray Ave (closest Aldi to CMU at the Greenfield/Sq. Hill border).
ALDI_STORE_CODE = os.getenv("ALDI_STORE_CODE", "4061")

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