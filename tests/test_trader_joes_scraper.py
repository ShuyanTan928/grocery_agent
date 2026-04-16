from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.scrapers import trader_joes as tj


def _gql_response(items, total_pages, current_page):
    return {
        "data": {
            "products": {
                "items": items,
                "total_count": sum(len([items]) for _ in range(total_pages)),
                "page_info": {
                    "current_page": current_page,
                    "page_size": len(items),
                    "total_pages": total_pages,
                },
            }
        }
    }


def _mock_resp(payload):
    m = MagicMock()
    m.json.return_value = payload
    m.raise_for_status = MagicMock()
    return m


def test_normalize_item_prefers_retail_price_and_unit():
    raw = {
        "sku": "078384",
        "url_key": "sliced-truffle-jack",
        "name": "Sliced Black Truffle Monterey Jack",
        "item_title": "Sliced Black Truffle Monterey Jack Cheese",
        "sales_size": "6",
        "sales_uom_description": "OZ",
        "availability": "1",
        "retail_price": "4.99",
        "price_range": {"minimum_price": {"final_price": {"currency": "USD", "value": 4.99}}},
    }
    normalized = tj.normalize_item(raw, "6343 Penn Ave")
    assert normalized["item_price"] == 4.99
    assert "Sliced Black Truffle Monterey Jack Cheese" in normalized["item_name"]
    assert "6 OZ" in normalized["item_name"]
    assert normalized["store"] == "trader joe's"
    assert normalized["_raw"]["sku"] == "078384"
    assert normalized["url"] == "https://www.traderjoes.com/home/products/pdp/sliced-truffle-jack-078384"


def test_normalize_item_url_none_when_sku_missing():
    assert tj.normalize_item({"item_title": "x", "retail_price": "1.0"}, "loc")["url"] is None


def test_build_store_meta_contains_brand_urls():
    meta = tj.build_store_meta(
        "638",
        store_id="trader_joes_shadyside",
        branch="Shadyside",
        address="6343 Penn Ave, Pittsburgh, PA 15206",
        lat=40.4583,
        lng=-79.9256,
        hours="8:00-21:00",
    )
    assert meta["store_code"] == "638"
    assert meta["website"] == "https://www.traderjoes.com"
    assert meta["products_url"].endswith("/home/products")
    assert meta["api_url"].endswith("/api/graphql")
    assert meta["branch"] == "Shadyside"


def test_normalize_item_falls_back_to_price_range():
    raw = {
        "sku": "x",
        "item_title": "Weird Item",
        "retail_price": None,
        "price_range": {"minimum_price": {"final_price": {"value": 1.23}}},
    }
    assert tj.normalize_item(raw, "loc")["item_price"] == 1.23


def test_normalize_item_handles_missing_price():
    raw = {"item_title": "NoPrice", "retail_price": None, "price_range": None}
    assert tj.normalize_item(raw, "loc")["item_price"] is None


def _mock_session_with_responses(responses):
    """Build a fake Session whose .post returns queued responses, .get is no-op."""
    session = MagicMock()
    session.post.side_effect = [_mock_resp(r) for r in responses]
    session.get = MagicMock()
    session.close = MagicMock()
    return session


@patch("tools.scrapers.trader_joes.time.sleep", lambda *_: None)
def test_fetch_all_products_paginates():
    page1 = _gql_response(
        [{"sku": "a", "item_title": "A", "retail_price": "1.0"}] * 2,
        total_pages=2,
        current_page=1,
    )
    page2 = _gql_response(
        [{"sku": "b", "item_title": "B", "retail_price": "2.0"}],
        total_pages=2,
        current_page=2,
    )
    session = _mock_session_with_responses([page1, page2])

    items = tj.fetch_all_products("638", page_size=2, session=session)

    assert len(items) == 3
    assert session.post.call_count == 2
    first_body = session.post.call_args_list[0].kwargs["json"]
    assert first_body["variables"] == {
        "storeCode": "638",
        "published": "1",
        "currentPage": 1,
        "pageSize": 2,
    }


@patch("tools.scrapers.trader_joes.time.sleep", lambda *_: None)
@patch("tools.scrapers.trader_joes._build_session")
def test_fetch_trader_joes_filters_unpriced_and_normalizes(mock_build):
    page = _gql_response(
        [
            {"sku": "1", "item_title": "Cheap", "sales_size": "1", "sales_uom_description": "LB", "retail_price": "1.50"},
            {"sku": "2", "item_title": "NoPrice", "retail_price": None, "price_range": None},
            {"sku": "3", "item_title": "", "retail_price": "9.99"},
        ],
        total_pages=1,
        current_page=1,
    )
    mock_build.return_value = _mock_session_with_responses([page])

    store_meta = tj.build_store_meta(
        "638",
        store_id="trader_joes_shadyside",
        branch="Shadyside",
        address="6343 Penn Ave, Pittsburgh, PA 15206",
    )
    payload = tj.fetch_trader_joes(
        "638",
        "6343 Penn Ave, Pittsburgh, PA 15206",
        store_meta=store_meta,
    )

    assert payload["store_code"] == "638"
    assert payload["item_count"] == 1
    assert payload["items"][0]["item_name"].startswith("Cheap")
    assert payload["items"][0]["item_price"] == 1.50
    assert payload["store"]["store_code"] == "638"
    assert payload["store"]["website"] == "https://www.traderjoes.com"


def test_post_raises_on_graphql_errors():
    session = _mock_session_with_responses([{"errors": [{"message": "oops"}]}])
    with pytest.raises(RuntimeError, match="GraphQL errors"):
        tj._post(session, "638", 1, 10, timeout=1.0)
