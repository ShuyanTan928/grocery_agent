# ============================================================
# tests/test_synonyms.py
# Tests for tools/synonyms.py and its integration with
# price_optimizer.find_cheapest / find_at_store.
# ============================================================

import pytest
from tools.synonyms import canonicalize, expand_query, matches_any
from tools.price_optimizer import find_cheapest, find_at_store, load_prices


class TestCanonicalize:
    def test_direct_surface_form(self):
        assert canonicalize("pork chop") == "pork"

    def test_plural_surface_form(self):
        assert canonicalize("pork chops") == "pork"

    def test_multi_word_phrase_matches_bigram(self):
        assert canonicalize("bone-in pork chops") == "pork"

    def test_single_token_match(self):
        assert canonicalize("milk") == "milk"

    def test_ground_beef(self):
        assert canonicalize("ground beef") == "beef"

    def test_unknown_phrase(self):
        assert canonicalize("martian tacos") is None

    def test_empty(self):
        assert canonicalize("") is None


class TestExpandQuery:
    def test_original_first(self):
        out = expand_query("pork chop")
        assert out[0] == "pork chop"
        assert "pork" in out
        assert "pork loin chops" in out or "pork loin chop" in out

    def test_unknown_returns_only_original(self):
        out = expand_query("martian tacos")
        assert out == ["martian tacos"]

    def test_deduplicates(self):
        out = expand_query("milk")
        assert len(out) == len(set(out))


class TestMatchesAny:
    def test_case_insensitive(self):
        assert matches_any("All Natural Pork Loin Chops 1 Lb", ["pork loin"])

    def test_no_match(self):
        assert not matches_any("Whole Milk Gallon", ["pork"])


class TestFindCheapestWithSynonyms:
    def test_pork_chop_matches_pork_category(self):
        # "pork chop" is not a mock-data category key, but synonyms map it to "pork".
        data = load_prices()
        result = find_cheapest("pork chop", data)
        assert result is not None
        assert "pork" in result["item_name"].lower() or "pork" in result.get("location", "").lower()

    def test_bone_in_pork_chops_still_resolves(self):
        data = load_prices()
        result = find_cheapest("bone-in pork chops", data)
        assert result is not None

    def test_sourdough_returns_none_when_no_actual_sourdough_listed(self):
        # Strict tier check: "sourdough" is a synonym surface form for the
        # "bread" category, but none of the mock bread entries actually
        # have "sourdough" in their item_name. We now refuse to silently
        # serve "Whole Wheat Bread" when the user asked for sourdough —
        # the cache fallback (or recommend flow) will handle that case.
        data = load_prices()
        assert find_cheapest("sourdough", data) is None

    def test_spaghetti_resolves_to_pasta(self):
        data = load_prices()
        result = find_cheapest("spaghetti", data)
        assert result is not None

    def test_unknown_still_returns_none(self):
        data = load_prices()
        assert find_cheapest("dragon fruit", data) is None


class TestFindAtStoreWithSynonyms:
    def test_pork_chop_at_trader_joes(self):
        data = load_prices()
        result = find_at_store("pork chop", "trader_joes_shadyside", data)
        assert result is not None
        assert "trader joe" in result["store"].lower()
