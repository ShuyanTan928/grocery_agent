# Pittsburgh Grocery Agent 🛒

A conversational AI agent that finds the cheapest grocery prices across Pittsburgh supermarkets, plans an optimal multi-stop driving route, and optionally generates an errand runner quote.

---

## Demo

```
Agent: Hi! What groceries do you need today?

You: I need milk, eggs, pork chops, and bananas

Agent: Got it! A couple of quick questions:
       - How many lbs of pork chops are you looking for? Buying 2+ lbs
         can get you a better per-lb price at some stores.
       - Any stores you'd prefer to avoid for certain items?

You: 2 lbs of pork, and I don't want to buy meat at Trader Joe's

Agent: Here's what I'll shop for:
       - Whole milk x1, eggs x1 dozen, pork chops 2 lb (not Trader Joe's), bananas x1
       Shall I find the best prices and plan your route? (yes/no)

You: yes

Agent: 🛒 Shopping Plan
       📍 Aldi (Greenfield) — 3850 Bigelow Blvd
          • Whole Milk 1 Gallon      $3.19
          • Large Eggs 12 Count      $2.89
          • Pork Loin Chops 2 lb     $8.98
       📍 Trader Joe's (Shadyside) — 6343 Penn Ave
          • Bananas 1 lb             $0.29

       🗺  Route: Trader Joe's → Aldi  (+6.3 min, 4.2 km)
       💰 Total: $15.35
```

You can also ask mid-flow:

```
You: can you list the options?

Agent: Here are 5 "pork chops" options I have in cache:
       1. Pork Loin Chops 1 lb — Aldi — $4.49
       2. Bone-In Center Cut Pork Chops — Giant Eagle — $5.29
       3. ...

       Say the number to lock one in (e.g. "pick 2"), or keep going
       with your list.

You: pick 2
```

---

## Features

- **Natural language input** — tell the agent what you need in plain English
- **Multi-turn clarification** — agent asks about quantities and store preferences before planning
- **Cache-only price optimization** — plans against real per-store caches (`data/price_cache/*.json`), not mock data
- **Store preferences** — respects "I don't want to buy meat at Trader Joe's"-style constraints, and "must buy X at store Y"
- **Recommend (side-flow)** — `recommend chicken wings` → top-K ranked by an LLM over cached SKUs, without disturbing the current shopping list
- **List options + LLM filter (side-flow)** — `can you list the options?` returns a numbered list from cache, optionally passed through the recommender's LLM so brand-name drift (e.g. "lamb" → "Lamb Weston fries") is pruned
- **Pick by number** — after a numbered list, `pick 3` / `option 2` / `I prefer 1` locks that SKU directly — no re-search, no drift
- **Hybrid intent router** (optional) — regex first; anything it misses can fall back to a small model classifier (`tools/intent_classifier.py`)
- **Route planning** — uses OpenRouteService to compute the optimal driving order (TSP solver)
- **Errand runner quotes** — generates a service fee + tip estimate if you want someone else to shop

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | **Google GenAI** (`google-genai`) *or* **OpenRouter** (`openai` SDK) — pick one, see below |
| Routing | [OpenRouteService](https://openrouteservice.org) — free tier, 2000 req/day |
| Agent pattern | ReAct (Reason + Act) with state machine (CLARIFY → CONFIRM → EXECUTE → DONE) + side-flows |
| Package manager | `uv` |
| Tests | `pytest` |

---

## Project Structure

```
grocery_agent/
├── agent/
│   └── agent.py              # State machine + side-flows + LLM orchestration
├── config/
│   └── settings.py           # API keys, feature flags, cache dir (env-backed)
├── data/
│   ├── price_cache/          # Per-store JSON caches (primary price source)
│   ├── mock_prices.json      # Legacy fixtures, used by tests/conftest.py
│   ├── mock_stores.json      # Store metadata: id, display_name, address, lat/lng, hours
│   └── mock_distance_matrix.json  # Pre-computed ORS driving times (used when USE_MOCK_DATA=true)
├── tools/
│   ├── price_optimizer.py    # Cache-only optimize + preferred/avoid store helpers
│   ├── price_cache.py        # Cache I/O, TTL helpers
│   ├── product_search.py     # Relevance-tiered search across caches
│   ├── recommender.py        # LLM top-K over search candidates
│   ├── intent_classifier.py  # (Optional) small-model intent labels for routing
│   ├── synonyms.py
│   ├── route_planner.py      # TSP solver + ORS distance matrix + directions
│   ├── errand_runner.py      # Errand runner fee calculator
│   ├── refresh_prices.py     # CLI to refresh per-store caches
│   └── scrapers/             # Trader Joe's, Giant Eagle, Target, Aldi, …
├── scripts/
│   ├── chat.py               # Developer REPL (flags, /state, /plan, replay)
│   ├── test_recommend.py
│   ├── eval_recommendations.py
│   └── …
├── tests/
│   ├── conftest.py                # Shared cache-from-mock fixture
│   ├── test_confirm_routing.py    # State-machine + side-flow routing
│   ├── test_intent_router.py      # Hybrid LLM router
│   ├── test_product_search.py
│   ├── test_price_optimizer.py
│   ├── test_recommend_intent.py
│   ├── test_route_planner.py
│   └── test_integration.py
├── main.py                    # Minimal interactive chat loop
├── test_api.py                # Standalone Google API key tester
├── test_ors.py                # Standalone ORS API key tester
└── pyproject.toml             # Dependencies (managed by uv)
```

---

## Setup

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/ShuyanTan928/grocery_agent.git
cd grocery_agent
uv venv
uv sync
```

### 3. Configure API keys

Run the setup script to create your `.env` file:

```bash
python setup.py
```

Then open `.env` and fill in your keys. Two LLM providers are supported — pick **one** path:

```env
# --- LLM: OpenRouter path (slugs look like "provider/model") ---
USE_OPENROUTER=true
OPENROUTER_API_KEY=sk-or-v1-...
LLM_MODEL=google/gemma-2-27b-it
# Optional (improves your app's attribution on openrouter.ai)
# OPENROUTER_HTTP_REFERER=https://yourapp.example
# OPENROUTER_APP_TITLE=Grocery Agent

# --- OR: LLM: Google GenAI path (short model names) ---
# USE_OPENROUTER=false
# GOOGLE_API_KEY=your_key_here
# LLM_MODEL=gemma-2-27b-it

# --- Optional: hybrid intent router ---
# When on, regex rules run first; leftovers get classified by a small model.
USE_LLM_INTENT_ROUTER=false
LLM_ROUTER_MODEL=google/gemma-3-1b-it
LLM_ROUTER_TEMPERATURE=0.0

# --- Optional: LLM filter for "list options" (default on) ---
# Costs one extra LLM call per list_options turn; stops brand-name drift.
USE_LLM_LIST_OPTIONS=true

# --- Routing — OpenRouteService (free, no credit card needed) ---
# Get key at: https://openrouteservice.org/dev/#/signup
ORS_API_KEY=your_key_here

# --- Data ---
# Some tools still use mock distance matrix when this is true
USE_MOCK_DATA=true
PRICE_CACHE_DIR=data/price_cache/
```

**Quick switch between providers** (without editing `.env`):

```bash
uv run python scripts/chat.py --provider openrouter --model google/gemma-2-27b-it
uv run python scripts/chat.py --provider google     --model gemma-2-27b-it
```

> ⚠️ Model ids have different formats: Google uses short names (`gemma-2-27b-it`), OpenRouter uses `provider/slug` (`google/gemma-2-27b-it`). Always update `LLM_MODEL` when you switch providers.

### 4. Verify your API keys

```bash
# Test Google / Gemma API
uv run python test_api.py

# Test OpenRouteService routing API
uv run python test_ors.py
```

### 5. Populate price caches (optional, for live data)

```bash
# Refresh per-store caches under data/price_cache/
uv run python -m tools.refresh_prices --help
```

---

## Running

```bash
# Minimal interactive chat
uv run python main.py

# Developer REPL (flags, /state, /plan, /flags, replay)
uv run python scripts/chat.py --help

# Run all tests
uv run pytest -v
```

### Chat commands

| Input | Effect |
|-------|--------|
| Any grocery list | Start a new shopping session |
| `yes` / `ok` / `confirm` | Confirm the plan and execute |
| `no` / `no, <change>` | Go back (bare no asks what to change; `no, …` applies the change and executes) |
| `recommend <item>` | Ranked picks, does not disturb the current list |
| `list the options` / `any alternatives?` | Numbered options from cache; `pick N` to lock one in |
| `pick 3` / `option 2` / `I prefer 1` | Lock the Nth option from the last list |
| `no thanks` / `thanks` / `done` (after a plan) | End the session gracefully |
| `reset` (in `main.py`) / `/reset` (in `scripts/chat.py`) | Start a completely new session |
| `quit` / `/exit` | Exit |

---

## Data Format

### Price cache (primary, per store)

Each file under `data/price_cache/` holds the SKUs for one store, produced by the matching scraper in `tools/scrapers/`. `tools/price_optimizer.py` reads these directly and never touches `mock_prices.json`.

### Legacy mock format (tests / fixtures)

`data/mock_prices.json` uses this schema — still handy for unit tests, fixtures, and offline demos:

```json
{
  "last_updated": "2026-04-14T10:00:00",
  "items": {
    "pork": [
      {
        "store": "trader joe's",
        "location": "6343 Penn Ave, Pittsburgh, PA 15206",
        "item_name": "Boneless Pork Tenderloin 1 lb",
        "item_price": 7.49
      },
      {
        "store": "aldi",
        "location": "3850 Bigelow Blvd, Pittsburgh, PA 15213",
        "item_name": "Pork Loin Chops 1 lb",
        "item_price": 4.49
      }
    ]
  }
}
```

`tests/conftest.py` materializes this file into a per-store temp cache, so tests run against a realistic cache shape without hitting the network.

---

## Agent Architecture

```
                          User message
                                │
                                ▼
      ┌─────────────────────────────────────────────────────┐
      │   Side-flows (do NOT advance state)                 │
      │                                                     │
      │   1. pick N         → lock SKU from last_options    │
      │   2. recommend X    → recommender LLM → reply       │
      │   3. list options   → cache (+opt LLM filter)       │
      │   4. LLM intent     → optional fallback router      │
      └──────────────┬──────────────────────────────────────┘
                     │ all miss
                     ▼
                ┌─────────────┐
                │   CLARIFY   │  Parse items, detect ambiguous quantities,
                │   (state)   │  collect "prefer" / "avoid" preferences
                └──────┬──────┘
                       │ all info collected
                       ▼
                ┌─────────────┐
                │   CONFIRM   │  Show plan summary, wait for yes/no
                │   (state)   │  — "no" asks what to change
                └──────┬──────┘  — "no, <refinement>" auto-executes
                       │ confirmed│ "yes" executes
                       ▼
                ┌─────────────┐
                │   EXECUTE   │  price_optimizer (cache-only) →
                │   (state)   │  apply prefer/avoid → route_planner →
                └──────┬──────┘  errand_runner → LLM summary
                       │ closer ("no thanks", "done", …)
                       ▼
                ┌─────────────┐
                │    DONE     │  Session ended. Any new message re-opens
                │   (state)   │  a fresh session without repeating info.
                └─────────────┘
```

Key invariants:

- **Side-flows never advance the state machine.** A question during CLARIFY stays in CLARIFY.
- **Empty plan short-circuits.** When `optimize_shopping_list` returns no hits, the agent says "couldn't find those" instead of letting the LLM fill in a template.
- **Pick-by-number skips search.** `handle_pick_request` uses the staged `last_options` snapshot directly — this is what prevents drift like `lamb → Lamb Weston fries`.

---

## Roadmap

- [x] Per-store price caches + refresh scrapers (Trader Joe's, Giant Eagle, Target, Aldi)
- [x] Cache-only optimizer (no more mock fallback for planning)
- [x] Product search with relevance tiering
- [x] LLM recommender + `list options` LLM filter + `pick N` side-flow
- [x] Optional hybrid intent router
- [ ] Live driving matrix everywhere (drop the last mock-data fallbacks)
- [ ] Fuzzy / embedding-based item name matching
- [ ] Web UI (Streamlit)
- [ ] Integrate real errand runner marketplace (TaskRabbit / Instacart)
- [ ] "Minimize stores vs minimize cost" trade-off setting
- [ ] Constrained decoding / logit bias for intent classification (lower router cost)

---

## Contributing

Pull requests welcome. Please run `uv run pytest -v` before submitting.
