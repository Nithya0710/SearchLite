from hypothesis import given
from hypothesis import strategies as st

from engine.tokenizer import Token, terms, tokenize


def test_lowercases_and_splits_on_punctuation():
    assert terms("Hello, World!", stem=False) == ["hello", "world"]


def test_hyphens_and_underscores_split():
    assert terms("state-of-the-art snake_case", stem=False, remove_stop_words=False) == [
        "state", "of", "the", "art", "snake", "case",
    ]


def test_apostrophes():
    assert terms("Don't", stem=False) == ["dont"]
    assert terms("John's book", stem=False) == ["john", "book"]
    assert terms("Don\u2019t", stem=False) == ["dont"]  # curly apostrophe


def test_numbers_are_kept():
    assert terms("COVID-19 in 2020", stem=False) == ["covid", "19", "2020"]


def test_unicode_casefolding():
    assert terms("STRASSE", stem=False) == terms("stra\u00dfe", stem=False)
    assert terms("Caf\u00e9", stem=False) == ["caf\u00e9"]


def test_empty_and_symbol_only_input():
    assert tokenize("") == []
    assert tokenize("   \n\t ") == []
    assert tokenize("!!! ... ???") == []


def test_stop_words_removed_but_positions_keep_gaps():
    assert tokenize("state of the art", stem=False) == [
        Token("state", 0),
        Token("art", 3),
    ]


def test_stop_words_can_be_kept():
    assert terms("to be or not", stem=False, remove_stop_words=False) == [
        "to", "be", "or", "not",
    ]


def test_negations_are_not_stop_words():
    assert "not" in terms("not good", stem=False)


def test_stemming_conflates_word_forms():
    assert len({*terms("run runs running")}) == 1
    assert terms("connection connected connecting") == ["connect"] * 3


def test_stemming_can_be_disabled():
    assert terms("running", stem=False) == ["running"]


@given(st.text())
def test_tokenize_never_crashes_and_positions_increase(text):
    tokens = tokenize(text)
    positions = [t.position for t in tokens]
    assert positions == sorted(set(positions))  # strictly increasing
    for t in tokens:
        assert t.term != ""
        assert not any(c.isspace() for c in t.term)