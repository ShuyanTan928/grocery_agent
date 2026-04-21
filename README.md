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
You: I want to make carbonara tonight

Agent: To make **spaghetti carbonara** (Italian), here's the shopping list:
       1. spaghetti — 200 g
       2. bacon — 150 g
       3. eggs — 2
       4. parmesan cheese — 50 g

       _Skipping pantry items you likely already have: garlic, black pepper, salt._

       Add all to your shopping list? (`yes` / `no` / `only 1 3 5`)

You: yes
# → merges into raw_items, moves to CONFIRM, then usual shopping flow.
```

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

- **LLM tool-calling agent** — a single orchestrator LLM decides which tool to run on every turn (add items, search, recommend, optimize, route, justify, …). No hand-wired regex router, no state-machine transitions. See **Agent Architecture** below.
- **Dish → ingredients** — say "I want to make carbonara" / "recipe for pad thai" / "bibimbap tonight" and the agent resolves the dish through `data/dishes.json` (67 dishes, 13 cuisines) + optional LLM fallback (cached to `data/dishes_cache.json`)
- **Natural language input** — tell the agent what you need in plain English; the LLM calls `add_items` / `parse_items` to structure it
- **Cache-only price optimization** — plans against real per-store caches (`data/price_cache/*.json`), not mock data; `optimize_and_route` is the workhorse tool
- **Optional LLM main optimizer** — set `USE_LLM_MAIN_OPTIMIZER=true` so `optimize_shopping_list` runs the recommender LLM once per line item to pick the best SKU from cache candidates; falls back to deterministic tier-only lookup on misses
- **Store preferences** — respects "I don't want to buy meat at Trader Joe's"-style constraints ("avoid") and "must buy X at store Y" ("prefer"); tools: `set_preference`, `unset_preference`
- **Recommend / list options / pick N** — `recommend chicken wings`, `can you list the options?`, `pick 3` all map to dedicated tools (`recommend_products`, `list_options` → stages on `state.last_options`, `pick_option`). No fuzzy re-search when picking.
- **Remove items** — `remove the water` / `drop the ginger` / `no I don't want the water` → `remove_items` prunes from both `raw_items` and the priced plan. Exact-name matches beat loose token matches (`remove orange` keeps `orange juice`); ambiguous targets surface a clarification observation; emptying the list auto-clears the plan.
- **Justify picks** — `why is the ginger ale on my list?` → `justify_pick` traces a SKU back to its `source_item` ingredient; returns an empty result (not a hallucination) if there's no active plan.
- **Route planning** — OpenRouteService for real driving matrix + TSP ordering (runs as part of `optimize_and_route`)
- **Custom route waypoints** — "I also need to swing by CMU on the way home" → `add_destination` geocodes the stop (offline Pittsburgh landmark dict + ORS fallback in live mode) and the next `optimize_and_route` weaves it into the TSP as a mandatory non-shopping stop. Tools: `add_destination`, `remove_destination`, `clear_destinations`.
- **User-configurable home** — "my home is in Oakland" / "I live at 419 Melwood" → `set_home` resolves the location through the same landmark-dict + ORS geocoder (or accepts explicit `lat`/`lng`). Until set, `plan_route` anchors at the default from `config/settings.py`. Tools: `set_home`, `clear_home`.
- **Errand runner quotes** — `set_errand(true)` then `optimize_and_route` attaches a service fee + tip estimate
- **Web UI** — FastAPI (`server.py`) + a single-file frontend (`web/index.html`) with chat, live shopping list, per-stop item breakdown, and an optional Leaflet map rendering the route (numbered markers, item popups, links to ORS turn-by-turn). Run `uv run uvicorn server:app --reload` and open <http://localhost:8000>.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | **Google GenAI** (`google-genai`) *or* **OpenRouter** (`openai` SDK). Default model: `google/gemini-2.5-flash` (OpenRouter) / `gemini-2.5-flash` (Google). Fallback suggestion: `openai/gpt-4o-mini`. |
| Routing | [OpenRouteService](https://openrouteservice.org) — free tier, 2000 req/day |
| Agent pattern | **LLM tool-calling loop** — one orchestrator LLM emits `{"tool": "...", "args": {...}}` JSON per step, loop runs the tool, appends the observation, asks again. Terminates when the LLM emits `reply`. |
| Package manager | `uv` |
| Tests | `pytest` + scripted self-test harness (`scripts/self_test.py`) |

---

## Project Structure

```
grocery_agent/
├── agent/
│   ├── state.py              # AgentState dataclass (universal inner state)
│   ├── tools.py              # TOOLS registry — every action the LLM can call
│   ├── prompts.py            # Orchestrator system prompt + per-step renderer
│   ├── loop.py               # ReAct loop: prompt → JSON → run_tool → observe
│   └── agent.py              # Thin facade: LLM clients, call_llm(), re-exports chat()
├── config/
│   └── settings.py           # API keys, feature flags, cache dir (env-backed)
├── data/
│   ├── price_cache/          # Per-store JSON caches (primary price source)
│   ├── dishes.json           # 67 hand-curated recipes across 13 cuisines
│   ├── dishes_cache.json     # Persisted LLM fallback answers (auto-grown)
│   ├── mock_prices.json      # Legacy fixtures, used by tests/conftest.py
│   ├── mock_stores.json      # Store metadata: id, display_name, address, lat/lng
│   └── mock_distance_matrix.json  # Pre-computed ORS driving times (for tests/offline)
├── tools/
│   ├── list_ops.py           # items_to_query_strings, apply_prefer/avoid, match helpers
│   ├── price_optimizer.py    # Cache-only optimize_shopping_list + find_at_store_in_cache
│   ├── price_cache.py        # Cache I/O, TTL helpers
│   ├── product_search.py     # Relevance-tiered search over caches
│   ├── recommender.py        # LLM top-K over search candidates
│   ├── dish_resolver.py      # dishes.json → ingredients + LLM fallback
│   ├── synonyms.py
│   ├── route_planner.py      # TSP + ORS distance matrix + haversine fallback
│   ├── geocode.py            # Pittsburgh landmark dict + ORS geocode fallback
│   ├── errand_runner.py      # Errand runner fee calculator
│   ├── refresh_prices.py     # CLI to refresh per-store caches
│   └── scrapers/             # Trader Joe's, Giant Eagle, Target, Aldi, …
├── scripts/
│   ├── chat.py               # Developer REPL (flags, /state, /plan, /trace, replay)
│   └── self_test.py          # Scripted-LLM smoke test — no API credits burned
├── tests/
│   ├── conftest.py           # Shared cache-from-mock fixture
│   ├── test_agent_loop.py    # JSON parsing, retry, emergency fallback, trace capture
│   ├── test_tools.py         # Per-tool unit tests (good + bad args path)
│   ├── test_product_search.py
│   ├── test_price_optimizer.py
│   ├── test_price_cache.py
│   ├── test_preferences.py
│   ├── test_dish_resolver.py
│   ├── test_synonyms.py
│   ├── test_route_planner.py
│   ├── test_geocode.py
│   ├── test_integration.py
│   └── test_trader_joes_scraper.py
├── server.py                  # FastAPI wrapper around agent.loop.chat()
├── web/
│   └── index.html             # Single-page frontend (chat + list + stops + Leaflet map)
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
# Default orchestrator model. gemini-2.5-flash is fast, JSON-stable, and cheap.
# Tested fallback: openai/gpt-4o-mini.
LLM_MODEL=google/gemini-2.5-flash
# Optional (improves your app's attribution on openrouter.ai)
# OPENROUTER_HTTP_REFERER=https://yourapp.example
# OPENROUTER_APP_TITLE=Grocery Agent

# --- OR: LLM: Google GenAI path (short model names) ---
# USE_OPENROUTER=false
# GOOGLE_API_KEY=your_key_here
# LLM_MODEL=gemini-2.5-flash

# --- Agent loop tunables ---
# Max steps the orchestrator can take in a single turn before emergency reply.
MAX_AGENT_STEPS=10
# Temperature for the orchestrator LLM. Low — we want stable tool choices.
AGENT_LOOP_TEMPERATURE=0.2

# --- Optional: override loop model independently of LLM_MODEL ---
# AGENT_LOOP_MODEL=google/gemini-2.5-flash

# --- Optional: LLM picks each shopping-list line item (default off) ---
# ~1 LLM call per item inside optimize_shopping_list (leaf tool, not the loop).
USE_LLM_MAIN_OPTIMIZER=false

# --- Optional: LLM fallback for unknown dish names (default off) ---
# Hits data/dishes.json first; on miss, asks the LLM and caches the answer.
USE_LLM_DISH_FALLBACK=false

# --- Routing — OpenRouteService (free, no credit card needed) ---
# Get key at: https://openrouteservice.org/dev/#/signup
ORS_API_KEY=your_key_here

# --- Data ---
# Some tools still use mock distance matrix when this is true
USE_MOCK_DATA=true
PRICE_CACHE_DIR=data/price_cache/
```

> **Deprecated.** `USE_LLM_INTENT_ROUTER`, `LLM_ROUTER_MODEL`, `LLM_ROUTER_TEMPERATURE`, and `USE_LLM_LIST_OPTIONS` were part of the pre-refactor regex/state-machine router. They're still read for backwards compatibility but emit a warning — the orchestrator LLM now makes those decisions natively.

**Quick switch between providers** (without editing `.env`):

```bash
uv run python scripts/chat.py --provider openrouter --model google/gemini-2.5-flash
uv run python scripts/chat.py --provider google     --model gemini-2.5-flash
```

> ⚠️ Model ids have different formats: Google uses short names (`gemini-2.5-flash`), OpenRouter uses `provider/slug` (`google/gemini-2.5-flash`). Always update `LLM_MODEL` when you switch providers.

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
# Web UI — chat + shopping list + per-stop breakdown + route map
uv run uvicorn server:app --reload --port 8000
# → open http://localhost:8000

# Minimal interactive chat (CLI)
uv run python main.py

# Developer REPL (flags, /state, /plan, /trace, /history, replay)
uv run python scripts/chat.py --help

# Show the tool-call trace on every turn (super useful while iterating on prompts)
uv run python scripts/chat.py --show-trace

# Bump the per-turn step cap
uv run python scripts/chat.py --max-steps 12

# Run all tests
uv run pytest -v
```

### Web UI

`server.py` is a thin FastAPI wrapper around `agent.loop.chat()`; the frontend lives in `web/index.html` (single file, Leaflet via CDN, no build step). Sessions are kept in memory keyed by a UUID the browser stores in `localStorage`.

Panels:
- **Left** — chat log + quick-prompt chips + Reset.
- **Right top** — `raw_items` pills (ambiguous rows are yellow) and the running total.
- **Right middle** — Route & stops. Each card shows the stop index, address, per-item SKUs + prices (clickable to the scraped URL), and inter-stop drive time. Destinations are green-accented; stores are blue-accented.
- **Right bottom** — Leaflet map (optional, appears once a route exists). Home is `H`, each stop is numbered; popups show the items at that stop. "Open turn-by-turn ↗" links out to the ORS directions deep-link.

API:

| Endpoint | Purpose |
|----------|---------|
| `POST /api/chat` | `{session_id, message}` → `{reply, state}`. Drives `agent.loop.chat()`. |
| `GET /api/state?session_id=…` | Current `AgentState` flattened for the frontend. |
| `POST /api/reset` | Wipe that session's state. |
| `GET /api/new-session` | Mint a UUID server-side (optional — client can also generate one). |
| `GET /` | Serves `web/index.html`. |

### Chat commands

The orchestrator LLM interprets free-form input and calls tools accordingly — you don't have to memorize any trigger phrases. Some examples:

| Example input | What the LLM typically does |
|---------------|-----------------------------|
| `I need milk, 2 lb pork chops, bananas` | `add_items` → clarify if quantities are ambiguous → `reply` |
| `yes` / `sounds good` | `optimize_and_route` → `reply` with the plan summary |
| `no, use Aldi for pork` | `set_preference(item="pork", store="aldi_…", kind="prefer")` → `optimize_and_route` → `reply` |
| `recommend the best ice cream` | `recommend_products` → `reply` |
| `can you list the options for lamb?` | `list_options(query="lamb")` → `reply` with a numbered list |
| `pick 3` / `option 2` / `I prefer 1` | `pick_option(n=3)` → `reply` |
| `remove the water` / `drop ginger` | `remove_items(target="water")` → `reply` (or ask for clarification if ambiguous) |
| `why is the ginger ale on my list?` | `justify_pick(target="ginger ale")` → `reply` |
| `I want to make carbonara tonight` | `propose_dish` → `reply` with ingredients → on `yes`, `apply_pending_dish` |
| `clear everything` / `reset` | `clear_list` → `reply` |

Meta-commands (handled by the REPL itself, never sent to the LLM):

| REPL command | Effect |
|--------------|--------|
| `/help`  | Show all meta-commands |
| `/state` | Dump the full `AgentState` as JSON |
| `/plan`  | Pretty-print the current `shopping_plan` |
| `/trace` | Print the tool-call trace for the last turn |
| `/history` | Print the recent conversation history |
| `/flags` | Show effective settings (provider, model, step cap, …) |
| `/reset` | New empty `AgentState` |
| `/exit` / `/quit` | Exit |

### Testing — three layers

The agent has **three complementary test harnesses**, each tuned to a different feedback loop. Use them in this order when iterating:

| Layer | Command | What it checks | LLM calls? | Runtime |
|-------|---------|----------------|-----------|---------|
| **1. Unit tests** | `uv run pytest` | Pure logic — tool functions, JSON parsing, prompt rendering, price optimizer, route planner, dish resolver. Catches ~90% of refactor regressions. | No (mocked) | < 1 s |
| **2. Scripted self-test** | `uv run python scripts/self_test.py` | Conversation-level flows with a `ScriptedLLM` (pre-written JSON tool calls). Verifies the *loop plumbing* + state mutations without depending on real LLM reasoning quality. | No (stubbed) | < 1 s |
| **3. Live probe** | `uv run python scripts/live_probe.py` | End-to-end against the **real** orchestrator LLM. Flags anomalies per turn (too many tool-chain steps, JSON retries, missing/unexpected tools, wrong state, suspicious reply text). Good for catching prompt-regression and "LLM picked weird tool" bugs. | Yes (real API) | 1–3 min |

#### Layer 1 — `pytest`

```bash
uv run pytest                 # all 148 tests
uv run pytest tests/test_tools.py -v    # just the tool-registry unit tests
uv run pytest -k remove_items   # filter by substring
```

- `tests/test_agent_loop.py` — JSON extraction / retry / emergency fallback / trace capture.
- `tests/test_tools.py` — per-tool good-args + bad-args path for all 19 tools.
- `tests/test_{price_optimizer,route_planner,dish_resolver,preferences,…}.py` — leaf-tool regressions.

#### Layer 2 — `scripts/self_test.py`

Uses `ScriptedLLM` (queue of pre-written JSON tool calls) + stubbed `optimize_shopping_list` / `plan_route`. Drives `chat()` through ~16 scripted scenarios (happy path, dish flow, remove/justify edge cases, ambiguous target disambiguation, empty-list rescue, pick-N guards, malformed-JSON retry, unknown-tool recovery, chained preferences, etc.). Each turn asserts `raw_items_count` / `plan_size` / `plan_total` / `has_plan` / `pending_dish_name` / `last_options_count`.

```bash
uv run python scripts/self_test.py
```

Add new flows by appending 3–5 lines to `SCENARIOS` instead of writing a fresh pytest.

#### Layer 3 — `scripts/live_probe.py`

Drives the **real** orchestrator LLM through 13 diverse scenarios and auto-flags anomalies. This is how we catch bugs that only show up when the model starts reasoning, e.g. "`remove orange` silently deletes orange juice" (caught + fixed — see `tools/list_ops.py::_singularize`).

```bash
uv run python scripts/live_probe.py                 # all 13 scenarios (~2 min)
uv run python scripts/live_probe.py --list          # list scenarios and exit
uv run python scripts/live_probe.py --scenario 5 6  # run just scenarios 5 & 6
uv run python scripts/live_probe.py --quiet         # print ONLY turns with anomalies
```

Anomaly classes flagged:

| Tag | What it means |
|-----|---------------|
| `tool_chain` | > 6 non-reply tool calls in one turn — potential runaway loop |
| `parse_retry` | Loop had to re-prompt because the model emitted malformed JSON |
| `emergency` | Emergency fallback fired (max-steps or unrecoverable parse failure) |
| `empty_reply` | The `reply` tool produced < 5 chars |
| `missing_tool` / `unexpected_tool` | Expected-tool checklist diverged from what the LLM called |
| `state.*` | Post-turn state mismatched the scenario's expectation (item count, plan store missing, avoid-store set, etc.) |
| `reply_missing_phrase` / `reply_forbidden_phrase` | Reply text failed a substring check |

Re-run after prompt changes, model swaps, or any edit that touches `agent/tools.py` or `agent/prompts.py`. Non-zero exit if any anomaly fires.

#### Interactive debugging

When something misbehaves, the REPL with `--show-trace` is the fastest way to see **exactly what tools the LLM called** and **how state changed each step**:

```bash
uv run python scripts/chat.py --show-trace
> I need 2 lb pork chops, avoid Trader Joe's
```

Then use `/state` (full AgentState dump), `/plan` (pretty plan), `/trace` (last-turn trace) as needed.

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

The agent is a classic **ReAct loop** driven by a single orchestrator LLM. There's no state machine, no regex router, no hand-wired intent detection — every per-turn decision is made by the LLM emitting one JSON tool call at a time, looking at observations, and deciding what to do next.

### The loop

```
                            User message
                                 │
                                 ▼
      ┌──────────────────────────────────────────────────────────┐
      │                 agent/loop.py :: chat()                  │
      │                                                          │
      │   render_loop_prompt(state, user_msg, observations)      │
      │   └── system prompt + tool registry + AgentState view    │
      │       + conversation history + observations so far       │
      │                                                          │
      │                          │                               │
      │                          ▼                               │
      │                  call_llm(prompt)   ◄── retries on       │
      │                          │             malformed JSON    │
      │                          ▼                               │
      │             _extract_json(raw)  ──►  {"tool": "X",       │
      │                                        "args": {...}}    │
      │                          │                               │
      │             ┌────────────┴───────────────┐               │
      │             ▼                            ▼               │
      │   tool == "reply"          run_tool(state, X, args)      │
      │     └─► return text        └─► mutate state, return obs  │
      │         (turn ends)        └─► append obs, loop again    │
      │                                                          │
      │   MAX_AGENT_STEPS hit ? ──► _emergency_reply()           │
      └──────────────────────────────────────────────────────────┘
```

Key points:

- **One orchestrator, many tools.** `agent/loop.py` owns the ReAct loop; it never decides *what* to do, just *whether* to keep going. All behavior is in tools.
- **Observations drive follow-up steps.** After each tool runs, its return dict is spliced into the next prompt. That's how the LLM knows, e.g., that `remove_items` returned `{"ambiguous": true, "matches": ["orange juice", "orange mango"]}` and decides to `reply` with a clarification question instead of running more tools.
- **`reply` is the only terminator.** A turn ends when (and only when) the LLM emits `{"tool": "reply", "args": {"text": "..."}}` — enforced via a `ReplySignal` exception.
- **Malformed JSON → retry, then emergency reply.** `_extract_json` strips code fences / prose. Up to `MAX_AGENT_PARSE_RETRIES` (default 2) retries with a schema reminder before the loop gives up and synthesizes an emergency reply.
- **Step cap + emergency reply.** `MAX_AGENT_STEPS` (default 8) prevents runaway tool-call chains. If the LLM still hasn't replied after the cap, the loop makes one more constrained call asking it to summarize whatever it observed.

### Universal inner state (`AgentState`)

Everything the agent remembers lives in one dataclass (`agent/state.py`). The LLM sees a compact view of it (`to_llm_view()`) on every step; the REPL's `/state` command dumps the full thing.

| Field | Purpose |
|-------|---------|
| `raw_items` | The user's active shopping list (name / quantity / unit / ambiguous) |
| `preferences` | Per-item **avoid** store constraints (`{"meat": ["trader_joes_shadyside"]}`) |
| `preferred_stores` | Per-item **must-buy-at** store constraints |
| `pending_dish` | Dish proposal awaiting user confirm (`{"name", "ingredients": [...]}`) |
| `last_options` | Staged numbered options from the most recent `list_options` call |
| `shopping_plan` | Output of `optimize_and_route` (`plan` / `total_cost` / `not_found` / `store_ids`) |
| `route_plan` | Driving order + distance matrix |
| `errand_quote` | Service-fee / tip estimate if `want_errand=true` |
| `want_errand` | Flip via `set_errand` tool |
| `destinations` | Non-shopping waypoints from `add_destination` (`[{label, address, lat, lng}]`). `plan_route` weaves them into the TSP as mandatory stops. |
| `home` | Optional user-configured home anchor (`{label, address, lat, lng, source}`) set via `set_home`. When `None`, `plan_route` falls back to `config.HOME_*`. |
| `conversation_history` | Running list of `{role, text}` entries |

### Tool registry

Every action is implemented as a `(state, args) -> observation` function in `agent/tools.py` and registered in the `TOOLS` dict. The full list (24 tools):

| Tool | What it does |
|------|--------------|
| `add_items` | Append items to `raw_items`, dedupe by lowercase name |
| `parse_items` | Fallback: LLM-parse a natural-language phrase → structured items |
| `remove_items` | Prune items from `raw_items` + `shopping_plan` (exact-first, `source_item`-authoritative, ambiguity surfaced) |
| `update_quantity` | Set quantity/unit on a raw item |
| `clear_list` | Wipe everything (items, plan, preferences, dish proposal) |
| `set_preference` | `kind="avoid"` or `"prefer"` a store for a given item |
| `unset_preference` | Drop a previously-set preference |
| `set_errand` | Toggle the `want_errand` flag |
| `add_destination` | Register a non-shopping waypoint on the route (geocoded via landmark dict / ORS; accepts explicit `lat`/`lng` to skip geocoding) |
| `remove_destination` | Drop a destination by label (case-insensitive) |
| `clear_destinations` | Wipe all destinations |
| `set_home` | Set the route's home anchor. Accepts a free-form `query` (goes through landmark dict + ORS), or explicit `lat`+`lng`. Returns `{ok:false,...}` if unresolvable — LLM then asks user for a neighborhood name. Invalidates `route_plan` + `errand_quote`. |
| `clear_home` | Revert to the default home from `config/settings.py` |
| `search_products` | Read-only relevance-ranked search over cache |
| `recommend_products` | LLM top-K over cache candidates, with reasons |
| `find_at_store` | Cheapest `item` at a specific `store_id` |
| `list_options` | Like `search_products` but stages results on `state.last_options` |
| `pick_option` | Lock in the Nth option from the last `list_options` (builds a 1-item plan directly) |
| `lookup_dish` | Read-only: resolve a dish to ingredients (no state mutation) |
| `propose_dish` | Stage a dish on `pending_dish` + return ingredient breakdown |
| `apply_pending_dish` | Splice pending ingredients into `raw_items` (supports `only=[1,3,5]` cherry-picking) |
| `cancel_pending_dish` | Drop the pending dish without adding anything |
| `optimize_and_route` | **THE main pipeline** — price-optimize + preference apply + TSP route + optional errand quote |
| `justify_pick` | Trace a SKU in the plan back to the raw item that produced it |
| `reply` | Terminator — emits final user-facing text; ends the turn |

### Invariants preserved from the old architecture

- **Empty plan short-circuits.** `optimize_and_route` returns `{"ok": false, "reason": "no items could be priced"}` instead of letting the LLM hallucinate a plan.
- **Pick-by-number skips re-search.** `pick_option` uses the `state.last_options` snapshot directly — this is why `list options for lamb` → `pick 3` can't drift into `Lamb Weston fries`.
- **Remove uses `source_item` back-pointers.** Every plan entry carries the original ingredient query that produced it. `remove_items` drops plan entries whose `source_item` matches the raw item being removed, so `remove orange` deletes `Navel Oranges` but spares `Simply Orange Juice`.
- **Auto-clear plan on empty list.** When `remove_items` empties both `raw_items` and `shopping_plan.plan`, the tool resets `state.shopping_plan = None` (and clears route + errand), so the orchestrator sees a fresh canvas next turn.
- **Destinations invalidate the route, not the plan.** Adding or removing a destination clears `state.route_plan` + `errand_quote` but leaves `shopping_plan` intact — the LLM just needs to re-run `optimize_and_route` to re-solve the TSP with the new waypoint set. Store selection (and therefore price) is unchanged.
- **Main-flow LLM optimizer (optional, leaf-tool flag).** `USE_LLM_MAIN_OPTIMIZER=true` makes `optimize_shopping_list` resolve each line via `recommend_for_query(..., topk=1)` before `apply_preferred_stores` / `apply_avoid_stores` run.
- **Dish → ingredients.** `tools/dish_resolver.py` walks `data/dishes.json` → `data/dishes_cache.json` → optional LLM fallback (`USE_LLM_DISH_FALLBACK=true`). Ingredients tagged `pantry: true` (salt, oil, common spices) are filtered out by default; `apply_pending_dish(include_pantry=true)` overrides.

---

## Roadmap

- [x] Per-store price caches + refresh scrapers (Trader Joe's, Giant Eagle, Target, Aldi)
- [x] Cache-only optimizer (no more mock fallback for planning)
- [x] Product search with relevance tiering
- [x] LLM recommender + `list_options` / `pick_option` tools
- [x] `remove_items` / `justify_pick` with `source_item` back-pointers and auto-clear
- [x] Refactor to LLM tool-calling loop (retire regex router + state machine)
- [x] Scripted self-test harness (`scripts/self_test.py`) + per-tool unit tests
- [x] Custom route waypoints (`add_destination` → TSP re-routes through non-shopping stops)
- [x] User-configurable home anchor (`set_home` → `plan_route` uses it as route start/end and web map H marker)
- [ ] Provider-native structured outputs (Gemini `response_schema`, OpenAI `tool_choice`) to drop JSON-parsing retries entirely
- [ ] Live driving matrix everywhere (drop the last mock-data fallbacks)
- [ ] Fuzzy / embedding-based item name matching
- [ ] Web UI (Streamlit)
- [ ] Integrate real errand runner marketplace (TaskRabbit / Instacart)
- [ ] "Minimize stores vs minimize cost" trade-off setting

---

## Contributing

Pull requests welcome. Please run `uv run pytest -v` before submitting.
