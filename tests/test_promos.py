"""Unit tests for tools.promos (extraction, cache IO, tool-facing API)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import promos as promos_mod


# -------- extractor --------

class TestExtractTarget:
    def test_reg_retail_drop_is_promo(self):
        items = [
            {
                "item_name": "Almonds",
                "item_price": 6.99,
                "url": "https://example.com/almonds",
                "_raw": {"reg_retail": 8.99, "promo_count": 0, "brand": "Acme"},
            },
        ]
        out = promos_mod.extract_target_promos(items)
        assert len(out) == 1
        row = out[0]
        assert row["sale_price"] == 6.99
        assert row["reg_price"] == 8.99
        assert row["discount_pct"] == pytest.approx(22.2, abs=0.1)
        assert row["reason"] == "reg_retail_drop"
        assert row["store"] == "target"

    def test_promo_flag_without_regdrop(self):
        items = [
            {
                "item_name": "Coffee",
                "item_price": 9.99,
                "_raw": {"reg_retail": 9.99, "promo_count": 2},
            },
        ]
        out = promos_mod.extract_target_promos(items)
        assert len(out) == 1
        assert out[0]["reason"] == "promo_flag"

    def test_no_signal_skipped(self):
        items = [
            {
                "item_name": "Bread",
                "item_price": 3.49,
                "_raw": {"reg_retail": 3.49, "promo_count": 0},
            },
        ]
        assert promos_mod.extract_target_promos(items) == []

    def test_sorted_by_discount_desc(self):
        items = [
            {"item_name": "A", "item_price": 9, "_raw": {"reg_retail": 10}},
            {"item_name": "B", "item_price": 5, "_raw": {"reg_retail": 10}},
            {"item_name": "C", "item_price": 8, "_raw": {"reg_retail": 10}},
        ]
        out = promos_mod.extract_target_promos(items)
        assert [r["item_name"] for r in out] == ["B", "C", "A"]


# -------- cache IO --------

def test_save_and_load_roundtrip(tmp_path: Path):
    data = {"generated_at": "2026-01-01T00:00:00+00:00", "stores": {"x": []}}
    target = tmp_path / "promos.json"
    promos_mod.save_promos(data, path=target)
    assert target.exists()
    got = promos_mod.load_promos(path=target)
    assert got == data


def test_load_missing_returns_empty(tmp_path: Path):
    assert promos_mod.load_promos(path=tmp_path / "nope.json") == {}


def test_load_corrupt_returns_empty(tmp_path: Path):
    p = tmp_path / "promos.json"
    p.write_text("{not json", encoding="utf-8")
    assert promos_mod.load_promos(path=p) == {}


# -------- tool-facing API --------

class TestGetDailyPromos:
    def _seed(self, tmp_path: Path, monkeypatch):
        cache = {
            "generated_at": "2026-04-21T00:00:00+00:00",
            "total_promos": 3,
            "stores": {
                "target_east_liberty": [
                    {"item_name": "A", "sale_price": 5, "reg_price": 10,
                     "discount_pct": 50.0, "reason": "reg_retail_drop",
                     "url": "https://ex/a"},
                    {"item_name": "B", "sale_price": 8, "reg_price": 10,
                     "discount_pct": 20.0, "reason": "reg_retail_drop",
                     "url": "https://ex/b"},
                ],
                "aldi_greenfield": [
                    {"item_name": "C", "sale_price": 2, "reg_price": 4,
                     "discount_pct": 50.0, "reason": "reg_retail_drop",
                     "url": None},
                ],
            },
        }
        path = tmp_path / "promos.json"
        path.write_text(json.dumps(cache), encoding="utf-8")
        monkeypatch.setattr(promos_mod, "PROMOS_CACHE_PATH", path)
        return path

    def test_empty_when_no_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(promos_mod, "PROMOS_CACHE_PATH", tmp_path / "x.json")
        out = promos_mod.get_daily_promos()
        assert out["empty"] is True
        assert out["total"] == 0
        assert out["per_store"] == {}

    def test_topk_caps(self, tmp_path, monkeypatch):
        self._seed(tmp_path, monkeypatch)
        out = promos_mod.get_daily_promos(topk_per_store=1)
        assert out["total"] == 2  # 1 per store × 2 stores
        assert len(out["per_store"]["target_east_liberty"]) == 1
        assert out["per_store"]["target_east_liberty"][0]["item_name"] == "A"

    def test_stores_filter(self, tmp_path, monkeypatch):
        self._seed(tmp_path, monkeypatch)
        out = promos_mod.get_daily_promos(stores=["aldi_greenfield"])
        assert list(out["per_store"].keys()) == ["aldi_greenfield"]

    def test_min_discount_filter(self, tmp_path, monkeypatch):
        self._seed(tmp_path, monkeypatch)
        out = promos_mod.get_daily_promos(min_discount_pct=30.0)
        assert "aldi_greenfield" in out["per_store"]
        # B is 20% — should drop
        names = {
            r["item_name"]
            for rows in out["per_store"].values()
            for r in rows
        }
        assert "B" not in names
        assert "A" in names and "C" in names


# -------- greeting helper --------

class TestGetGreetingPromos:
    def _write(self, tmp_path: Path, stores: dict) -> Path:
        cache = {
            "generated_at": "2026-04-21T00:00:00+00:00",
            "total_promos": sum(len(v) for v in stores.values()),
            "stores": stores,
        }
        p = tmp_path / "promos.json"
        p.write_text(json.dumps(cache), encoding="utf-8")
        return p

    def test_returns_grocery_items_sorted_by_discount(self, tmp_path):
        p = self._write(tmp_path, {
            "target_east_liberty": [
                {"item_name": "Almond Milk - 64oz", "sale_price": 3, "reg_price": 5,
                 "discount_pct": 40.0, "reason": "reg_retail_drop", "url": "u1"},
                {"item_name": "Heart Shaped Lollipop", "sale_price": 1, "reg_price": 5,
                 "discount_pct": 80.0, "reason": "reg_retail_drop", "url": None},
                {"item_name": "Bread Loaf", "sale_price": 2, "reg_price": 4,
                 "discount_pct": 50.0, "reason": "reg_retail_drop", "url": "u3"},
            ],
        })
        out = promos_mod.get_greeting_promos(limit=3, min_discount_pct=20.0, path=p)
        names = [r["item_name"] for r in out["items"]]
        assert "Heart Shaped Lollipop" not in names
        # Sorted by discount_pct desc: Bread (50%) then Almond Milk (40%)
        assert names == ["Bread Loaf", "Almond Milk - 64oz"]
        assert out["items"][0]["store_id"] == "target_east_liberty"
        assert out["empty"] is False

    def test_respects_min_discount_and_limit(self, tmp_path):
        p = self._write(tmp_path, {
            "target_east_liberty": [
                {"item_name": f"Item {i}", "sale_price": 9, "reg_price": 10,
                 "discount_pct": 10.0, "reason": "reg_retail_drop", "url": None}
                for i in range(5)
            ],
        })
        out = promos_mod.get_greeting_promos(limit=3, min_discount_pct=15.0, path=p)
        assert out["empty"] is True
        assert out["items"] == []

    def test_empty_when_cache_missing(self, tmp_path):
        out = promos_mod.get_greeting_promos(path=tmp_path / "nope.json")
        assert out == {"generated_at": None, "items": [], "empty": True}
