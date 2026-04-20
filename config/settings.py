# ============================================================
# config/settings.py
# Central configuration for all API keys and app settings.
# Copy this file to config/settings_local.py and fill in your
# real API keys. Never commit settings_local.py to git.
# ============================================================

import os
import warnings


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


# --- LLM provider switch ---
# Default path = Google GenAI. Set USE_OPENROUTER=true to route through
# OpenRouter's OpenAI-compatible API instead.
# Back-compat: LLM_PROVIDER=openrouter is honored too.
_use_openrouter = _env_truthy("USE_OPENROUTER")
_legacy_openrouter = os.getenv("LLM_PROVIDER", "google").strip().lower() == "openrouter"
LLM_PROVIDER = "openrouter" if (_use_openrouter or _legacy_openrouter) else "google"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_GENAI_KEY")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "").strip()

# --- LLM model ---
# Default picks Google GenAI's native short name `gemini-2.5-flash`.
# When USE_OPENROUTER=true the default switches to the OpenRouter slug
# `google/gemini-2.5-flash`. Users can always override via LLM_MODEL.
#
# Suggested fallback if you see JSON parse failures:
#   LLM_MODEL=openai/gpt-4o-mini        (OpenRouter)
#   LLM_MODEL=gemini-2.5-pro            (Google GenAI)
_default_llm_model = (
    "google/gemini-2.5-flash" if LLM_PROVIDER == "openrouter" else "gemini-2.5-flash"
)
LLM_MODEL = os.getenv("LLM_MODEL", _default_llm_model)

# Optionally override the model used by the orchestrator loop. Defaults
# to LLM_MODEL so existing setups just work.
AGENT_LOOP_MODEL = os.getenv("AGENT_LOOP_MODEL", "").strip() or LLM_MODEL

# Hard cap on tool-calling steps per user turn. Keeps cost bounded even
# if the LLM gets stuck; on exceed, an emergency-reply synth runs.
try:
    MAX_AGENT_STEPS = max(1, int(os.getenv("MAX_AGENT_STEPS", "8")))
except ValueError:
    MAX_AGENT_STEPS = 8

# --- Routing ---
ORS_API_KEY = os.getenv("ORS_API_KEY", "YOUR_ORS_KEY")
ORS_BASE_URL = "https://api.openrouteservice.org"

# --- Data ---
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "true").lower() == "true"
MOCK_DATA_DIR = "data/"
PRICE_CACHE_TTL = 3600
PRICE_CACHE_DIR = os.getenv("PRICE_CACHE_DIR", "data/price_cache/")

# --- Trader Joe's ---
TRADER_JOES_STORE_CODE = os.getenv("TRADER_JOES_STORE_CODE", "638")

# --- Giant Eagle ---
GIANT_EAGLE_STORE_CODE = os.getenv("GIANT_EAGLE_STORE_CODE", "38")

# --- Target ---
TARGET_STORE_CODE = os.getenv("TARGET_STORE_CODE", "2757")

# --- Aldi ---
ALDI_STORE_CODE = os.getenv("ALDI_STORE_CODE", "4061")

# --- User home location ---
HOME_ADDRESS = "4800 Forbes Ave, Pittsburgh, PA 15213"
HOME_LAT = 40.4444
HOME_LNG = -79.9431

# --- Routing preferences ---
MAX_STORES_PER_TRIP = 4
COST_PER_MILE = 0.21

# --- Errand runner ---
ERRAND_BASE_FEE = 5.00
ERRAND_PER_STORE_FEE = 2.00


# ────────────────────────── leaf-tool flags ──────────────────────────
#
# USE_LLM_MAIN_OPTIMIZER is a LEAF-tool flag (affects
# tools/price_optimizer.py internals, NOT the orchestrator). Keeping it
# as-is. When true, each line item in optimize_shopping_list is resolved
# via the recommender LLM (same pipeline as `recommend X`) with a cache
# fallback. Default off.
USE_LLM_MAIN_OPTIMIZER = _env_truthy("USE_LLM_MAIN_OPTIMIZER")

# --- Dish -> ingredients resolver ---
# When true, tools.dish_resolver.resolve_dish falls back to an LLM for
# dishes that aren't in data/dishes.json (or dishes_cache.json).
USE_LLM_DISH_FALLBACK = _env_truthy("USE_LLM_DISH_FALLBACK")


# ────────────────────────── deprecated flags ─────────────────────────
#
# These controlled the old regex + state-machine agent. The new
# LLM-orchestrator loop makes them redundant:
#   - USE_LLM_INTENT_ROUTER — the orchestrator IS the intent router
#   - USE_LLM_LIST_OPTIONS  — tools/recommend_products does the same work
#     when the orchestrator chooses it instead of list_options
#   - LLM_ROUTER_MODEL / LLM_ROUTER_TEMPERATURE — no separate classifier
#     pass anymore
#
# We still READ the env vars (so imports don't KeyError in transitional
# environments) and emit a DeprecationWarning when they're explicitly
# set. They don't affect runtime behavior.

def _deprecated_env(name: str, replacement_hint: str) -> str | bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return False
    warnings.warn(
        f"{name} is deprecated and ignored by the new LLM-orchestrator "
        f"loop. {replacement_hint}",
        DeprecationWarning,
        stacklevel=2,
    )
    return raw

USE_LLM_INTENT_ROUTER = bool(_deprecated_env(
    "USE_LLM_INTENT_ROUTER",
    "Routing is now handled by the orchestrator LLM itself.",
))
USE_LLM_LIST_OPTIONS = bool(_deprecated_env(
    "USE_LLM_LIST_OPTIONS",
    "Use the recommend_products tool for LLM-filtered options.",
))
LLM_ROUTER_MODEL = str(_deprecated_env(
    "LLM_ROUTER_MODEL",
    "The orchestrator uses AGENT_LOOP_MODEL / LLM_MODEL.",
) or "")
try:
    LLM_ROUTER_TEMPERATURE = float(os.getenv("LLM_ROUTER_TEMPERATURE", "0.0"))
except ValueError:
    LLM_ROUTER_TEMPERATURE = 0.0
if "LLM_ROUTER_TEMPERATURE" in os.environ and os.environ["LLM_ROUTER_TEMPERATURE"] != "0.0":
    warnings.warn(
        "LLM_ROUTER_TEMPERATURE is deprecated; tune AGENT_LOOP_TEMPERATURE instead.",
        DeprecationWarning,
        stacklevel=2,
    )
