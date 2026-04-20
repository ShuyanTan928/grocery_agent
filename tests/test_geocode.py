"""Unit tests for the thin landmark + ORS geocoder (tools/geocode.py).

We only test the offline landmark path here — ORS fallback needs the
network and a key so it's guarded by USE_MOCK_DATA in the helper itself.
"""
from __future__ import annotations

import pytest

from tools.geocode import (
    PITTSBURGH_LANDMARKS,
    _match_landmark,
    _normalize,
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

    def test_unknown_place_is_none_in_mock_mode(self, monkeypatch):
        monkeypatch.setattr("tools.geocode.USE_MOCK_DATA", True)
        assert geocode("some obscure place 12345") is None

    def test_blank_input_is_none(self):
        assert geocode("") is None
        assert geocode("   ") is None
