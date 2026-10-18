import pytest

from knowledge_assistant.text import (
    answer_verdict,
    contains_fact,
    normalize,
    normalize_for_matching,
    numbers_in,
    tokenize,
)


@pytest.mark.parametrize("raw", ["€50,000", "50 000 €", "EUR 50.000", "50 000 €"])
def test_thousands_separators_are_removed(raw):
    assert "50000" in normalize(raw)


def test_decimal_comma_and_percent_spacing():
    assert normalize("2,5 %") == "2.5%"


def test_tokenize_removes_stop_words_and_stems():
    assert tokenize("The suppliers are approved") == ["supplier", "approv"]


def test_number_words_become_digits_in_english_and_french():
    assert normalize_for_matching("three written quotes") == "3 written quotes"
    assert normalize_for_matching("trois devis") == "3 devis"


@pytest.mark.parametrize("raw", ["at least €2 million per claim", "€2M per claim", "2 millions €"])
def test_millions_are_expanded(raw):
    assert "2000000" in normalize_for_matching(raw)


@pytest.mark.parametrize(
    ("answer", "fact", "expected"),
    [
        # phrases and plurals
        ("You need at least three written quotes.", "3 quote | 3 written quote", True),
        ("Il faut au moins trois devis écrits.", "3 quote | 3 devis", True),
        ("Approved by the Site Director and the Category Manager.", "site director", True),
        ("The director of the site approves.", "site director", False),
        # numbers never match a longer number
        ("The limit is €5,000 per month.", "5000", True),
        ("The limit is €50,000 per month.", "5000", False),
        ("The limit is €15,000 per month.", "5000", False),
        ("Gifts up to €500 are allowed.", "50", False),
        ("You need 30 quotes.", "3 quote", False),
        ("The tolerance is 2 % and €100.", "2%", True),
        ("The tolerance is 12%.", "2%", False),
        # whole words unless the fact asks for a prefix with *
        ("Getting the continuity plan", "tin", False),
        ("Tin, tantalum and tungsten", "tin", True),
        ("Please note the rule.", "not", False),
        ("Splitting is prohibited.", "prohibit*", True),
        ("Splitting is prohibited.", "prohibit", False),
        # apostrophes
        ("Le directeur d’usine approuve.", "directeur d'usine", True),
    ],
)
def test_contains_fact(answer, fact, expected):
    assert contains_fact(answer, fact) is expected


@pytest.mark.parametrize(
    ("answer", "verdict"),
    [
        ("Yes, the Board must approve it.", "yes"),
        ("**No.** Gift cards are never accepted.", "no"),
        ("Non, les cartes cadeaux ne sont jamais acceptées.", "no"),
        ("Oui, avec l'accord du CFO.", "yes"),
        ("A €30 gift card does not exceed the limit, so you can accept it.", None),
        ("Nothing in the policies covers it.", None),
    ],
)
def test_answer_verdict(answer, verdict):
    assert answer_verdict(answer) == verdict


def test_numbers_are_extracted_after_normalisation():
    assert numbers_in("€50,000 or 50 000 € within three days") == {50000.0, 3.0}
