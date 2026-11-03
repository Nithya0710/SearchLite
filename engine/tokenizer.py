"""Turn raw text into a list of (term, position) tokens.

Pipeline: normalize -> split -> drop stop words -> stem.

Design decisions (also record these in docs/decisions.md):
  * Positions are counted over the raw token stream BEFORE stop-word removal,
    so removed stop words leave gaps. This keeps phrase queries correct:
    "state of the art" -> [("state", 0), ("art", 3)].
  * Hyphens and underscores split tokens: "state-of-the-art" -> 4 tokens.
  * Apostrophes: possessive 's is dropped ("John's" -> "john"); other inner
    apostrophes are removed ("don't" -> "dont").
  * Numbers are kept as tokens ("covid-19" -> "covid", "19").
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import snowballstemmer

# Small, conservative list. Negations ("no", "not") are deliberately kept.
STOP_WORDS: frozenset[str] = frozenset(
    """
    a an and are as at be but by for if in into is it of on or such that the
    their then there these they this to was will with
    """.split()
)

# Curly / modifier apostrophes -> plain ASCII apostrophe.
_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'"})

# A token is a run of letters/digits (any script, no underscore), optionally
# joined by inner apostrophes.
_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
_POSSESSIVE_RE = re.compile(r"(?<=[^\W_])'s\b")

_stemmer = snowballstemmer.stemmer("english")  # Porter2 ("Snowball English")


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    # Stemming is pure and the same words repeat constantly, so cache it.
    return _stemmer.stemWord(word)


@dataclass(frozen=True, slots=True)
class Token:
    term: str
    position: int  # index in the raw token stream (stop words count)


def normalize(text: str) -> str:
    """Unicode-normalize, unify apostrophes, casefold, drop possessive 's."""
    text = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).casefold()
    return _POSSESSIVE_RE.sub("", text)


def tokenize(
    text: str, *, stem: bool = True, remove_stop_words: bool = True
) -> list[Token]:
    """Return the tokens of `text` in order, each with its raw position."""
    tokens: list[Token] = []
    for position, raw in enumerate(_TOKEN_RE.findall(normalize(text))):
        word = raw.replace("'", "")
        if remove_stop_words and word in STOP_WORDS:
            continue
        tokens.append(Token(_stem(word) if stem else word, position))
    return tokens


def terms(text: str, **kwargs: bool) -> list[str]:
    """Convenience: just the terms, no positions."""
    return [t.term for t in tokenize(text, **kwargs)]