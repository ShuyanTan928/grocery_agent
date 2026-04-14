# ============================================================
# tools/price_scraper.py
# Scrapes or fetches live product prices from Pittsburgh stores.
#
# Currently returns mock data. Real scrapers will be added
# per-store as they are implemented.
#
# Each scraper must return a list of:
# { "canonical_name": str, "price": float, "unit": str }
# ============================================================

import json
from pathlib import Path
from datetime import datetime
from config.settings import USE_MOCK_DATA, MOCK_DATA_DIR


def scrape_giant_eagle(items: list[str]) -> list[dict]:
    """
    Scrape Giant Eagle website for item prices.
    TODO: Implement using requests + BeautifulSoup or Playwright.
    Giant Eagle has a public search at gianteagle.com/foods/grocery
    """
    raise NotImplementedError("Giant Eagle scraper not yet implemented")


def scrape_aldi(items: list[str]) -> list[dict]:
    """
    Scrape Aldi weekly specials page for item prices.
    TODO: Aldi does not have a live price API; use weekly ad PDF parser
    or community price database.
    """
    raise NotImplementedError("Aldi scraper not yet implemented")


def scrape_walmart(items: list[str]) -> list[dict]:
    """
    Fetch Walmart prices using their unofficial product search endpoint.
    TODO: Use walmart.com/search?q={item} with BeautifulSoup.
    """
    raise NotImplementedError("Walmart scraper not yet implemented")


def get_prices_for_store(store_id: str, items: list[str]) -> list[dict]:
    """
    Dispatcher: routes to the correct scraper based on store_id,
    or returns mock data if USE_MOCK_DATA is True.
    """
    if USE_MOCK_DATA:
        # Return all mock prices — the optimizer will pick the right store
        path = Path(MOCK_DATA_DIR) / "mock_prices.json"
        with open(path) as f:
            data = json.load(f)

        results = []
        for product in data["products"]:
            if store_id in product["prices"]:
                results.append({
                    "canonical_name": product["canonical_name"],
                    "price": product["prices"][store_id],
                    "unit": product["unit"],
                    "scraped_at": datetime.utcnow().isoformat(),
                })
        return results

    # Live scraper dispatch
    scraper_map = {
        "giant_eagle_squirrel_hill": scrape_giant_eagle,
        "aldi_greenfield": scrape_aldi,
        "walmart_crafton": scrape_walmart,
    }

    if store_id not in scraper_map:
        raise ValueError(f"No scraper registered for store: {store_id}")

    return scraper_map[store_id](items)
