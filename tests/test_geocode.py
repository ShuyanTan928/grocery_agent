"""Unit tests for the thin landmark + ORS geocoder (tools/geocode.py).

ORS calls are monkeypatched; disk cache uses tmp_path overrides.
"""
from __future__ import annotations

from tools.geocode import (
    PITTSBURGH_LANDMARKS,
    _match_landmark,
    _normalize,
    build_ors_search_text,
    geocode,
)


class TestNormalize:
    def test_strips_punctuation_and_lowercases(self):
        assert _normalize("  Carnegie Mellon, University!  ") == "carnegie mellon university"

    def test_empty_handling(self):
        assert _normalize("") == ""
        assert _normalize(None) == ""


class TestMatchLandmark:
    def test_exact_key_hit(self):
        coords = _match_landmark("cmu")
        assert coords == PITTSBURGH_LANDMARKS["cmu"]

    def test_substring_match_prefers_longest(self):
        # "carnegie mellon university" contains "carnegie mellon" — both
        # are in the dict and point to the same coords, so either is fine.
        coords = _match_landmark("carnegie mellon university")
        assert coords is not None

    def test_miss_returns_none(self):
        assert _match_landmark("antarctica ice station zebra") is None


class TestGeocode:
    def test_common_pittsburgh_landmark_hits(self):
        for phrase in ["CMU", "Downtown Pittsburgh", "Strip District", "Airport"]:
            hit = geocode(phrase)
            assert hit is not None, f"{phrase!r} should resolve"
            assert -90 <= hit["lat"] <= 90
            assert -180 <= hit["lng"] <= 180
            assert hit["source"] == "landmark"
            assert hit["label"] == phrase

    def test_bakery_living_landmark_coords(self):
        for phrase in ["bakery living", "6480 living pi", "6480 Living Pl"]:
            hit = geocode(phrase)
            assert hit is not None
            assert hit["source"] == "landmark"
            assert abs(hit["lat"] - 40.45601) < 1e-5
            assert abs(hit["lng"] - (-79.91707)) < 1e-5

    def test_unknown_place_is_none_without_ors_or_cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr("tools.geocode.USE_MOCK_GEOCODE", True)
        monkeypatch.setattr("tools.geocode.GEOCODE_CACHE_OVERRIDE", tmp_path / "gc.json")
        assert geocode("some obscure place 12345") is None

    def test_unknown_place_is_none_when_ors_key_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr("tools.geocode.ORS_API_KEY", "")
        monkeypatch.setattr("tools.geocode.USE_MOCK_GEOCODE", False)
        monkeypatch.setattr("tools.geocode.GEOCODE_CACHE_OVERRIDE", tmp_path / "gc.json")
        assert geocode("some obscure place 12345") is None

    def test_ors_path_works_independent_of_product_mock_flag(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr("tools.geocode.GEOCODE_CACHE_OVERRIDE", tmp_path / "gc.json")
        monkeypatch.setattr("tools.geocode.ORS_API_KEY", "test-key")
        monkeypatch.setattr(
            "tools.geocode._ors_geocode",
            lambda q: {
                "lat": 40.5,
                "lng": -80.0,
                "address": "Resolved St, Pittsburgh",
                "source": "ors",
            },
        )
        hit = geocode("999 nowhere lane")
        assert hit is not None
        assert hit["lat"] == 40.5
        assert hit["source"] == "ors"

    def test_second_call_reads_disk_cache_without_ors(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr("tools.geocode.GEOCODE_CACHE_OVERRIDE", tmp_path / "gc.json")
        monkeypatch.setattr("tools.geocode.ORS_API_KEY", "test-key")
        calls: list[str] = []

        def fake_ors(q: str):
            calls.append(q)
            return {
                "lat": 40.1,
                "lng": -79.9,
                "address": "First Hit",
                "source": "ors",
            }

        monkeypatch.setattr("tools.geocode._ors_geocode", fake_ors)
        first = geocode("unique cache probe street")
        assert first["source"] == "ors"
        assert len(calls) == 1

        monkeypatch.setattr("tools.geocode.ORS_API_KEY", "")
        second = geocode("unique cache probe street")
        assert second["lat"] == 40.1
        assert second["source"] == "ors"

    def test_blank_input_is_none(self):
        assert geocode("") is None
        assert geocode("   ") is None


class TestBuildOrsSearchText:
    def test_appends_pittsburgh_when_no_context(self):
        assert "Pittsburgh" in build_ors_search_text("419 melwood ave")

    def test_no_double_append_when_pa_zip_present(self):
        t = build_ors_search_text("4800 Forbes Ave, PA 15213")
        assert t == "4800 Forbes Ave, PA 15213"

    def test_no_append_when_pittsburgh_in_query(self):
        assert build_ors_search_text("CMU Pittsburgh") == "CMU Pittsburgh"
