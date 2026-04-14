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

---

## Features

- **Natural language input** — tell the agent what you need in plain English
- **Multi-turn clarification** — agent asks about quantities and store preferences before planning
- **Price optimization** — finds the cheapest store for each item across 5 Pittsburgh stores
- **Store preferences** — respects "I don't want to buy meat at Trader Joe's"-style constraints
- **Route planning** — uses OpenRouteService to compute the optimal driving order (TSP solver)
- **Errand runner quotes** — generates a service fee + tip estimate if you want someone else to shop

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Gemma 3 27B via Google GenAI SDK (`google-genai`) |
| Routing | [OpenRouteService](https://openrouteservice.org) — free tier, 2000 req/day |
| Agent pattern | ReAct (Reason + Act) with state machine (CLARIFY → CONFIRM → EXECUTE) |
| Package manager | `uv` |
| Tests | `pytest` |

---

## Project Structure

```
grocery_agent/
├── agent/
│   └── agent.py              # ReAct agent: state machine, LLM calls, tool orchestration
├── config/
│   └── settings.py           # API keys and runtime settings (loaded from .env)
├── data/
│   ├── mock_prices.json       # Price data: {category: [{store, location, item_name, item_price}]}
│   ├── mock_stores.json       # Store metadata: id, display_name, address, lat/lng, hours
│   └── mock_distance_matrix.json  # Pre-computed ORS driving times (used when USE_MOCK_DATA=true)
├── tools/
│   ├── price_optimizer.py     # Finds cheapest store per item, applies store preferences
│   ├── price_scraper.py       # Scraper stubs (Giant Eagle, Aldi, Walmart — TODO)
│   ├── route_planner.py       # TSP solver + ORS distance matrix + directions
│   └── errand_runner.py       # Errand runner fee calculator
├── tests/
│   ├── test_price_optimizer.py
│   ├── test_route_planner.py
│   └── test_integration.py
├── main.py                    # Interactive CLI chat loop
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
git clone https://github.com/YOUR_USERNAME/grocery_agent.git
cd grocery_agent
uv venv
uv sync
```

### 3. Configure API keys

Run the setup script to create your `.env` file:

```bash
python setup.py
```

Then open `.env` and fill in your keys:

```env
# LLM — Google AI Studio (for Gemma 3)
# Get key at: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your_key_here

# Routing — OpenRouteService (free, no credit card needed)
# Get key at: https://openrouteservice.org/dev/#/signup
ORS_API_KEY=your_key_here

# Set to "false" when you have real scrapers ready
USE_MOCK_DATA=true
```

### 4. Verify your API keys

```bash
# Test Google / Gemma API
uv run python test_api.py

# Test OpenRouteService routing API
uv run python test_ors.py
```

---

## Running

```bash
# Start the interactive chat agent
uv run python main.py

# Run all tests
uv run pytest -v
```

### Chat commands

| Input | Effect |
|-------|--------|
| Any grocery list | Start a new shopping session |
| `yes` / `ok` / `confirm` | Confirm the plan and execute |
| `no` / `change` | Go back and modify the plan |
| `reset` | Start a completely new session |
| `quit` | Exit |

---

## Data Format

Price data in `data/mock_prices.json` follows this schema — the same format your real scrapers should output:

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

To add a new store or item, just add entries to the appropriate category array. The optimizer automatically picks the cheapest entry per category.

---

## Agent Architecture

```
User message
     │
     ▼
┌─────────────┐
│   CLARIFY   │  Parse items, detect ambiguous quantities,
│   (state)   │  ask for store preferences
└──────┬──────┘
       │ all info collected
       ▼
┌─────────────┐
│   CONFIRM   │  Show plan summary, wait for yes/no
│   (state)   │
└──────┬──────┘
       │ confirmed
       ▼
┌─────────────┐
│   EXECUTE   │  price_optimizer → route_planner → errand_runner
│   (state)   │  → LLM summary
└─────────────┘
```

---

## Roadmap

- [ ] Implement live price scrapers (Giant Eagle, Aldi, Walmart)
- [ ] Add price cache with TTL to avoid re-scraping every session
- [ ] Switch `USE_MOCK_DATA=false` for end-to-end live run
- [ ] Add fuzzy / embedding-based item name matching
- [ ] Build web UI (Streamlit)
- [ ] Integrate real errand runner marketplace (TaskRabbit / Instacart)
- [ ] "Minimize stores vs minimize cost" trade-off setting

---

## Contributing

Pull requests welcome. Please run `uv run pytest -v` before submitting.