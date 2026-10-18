"""Text normalisation shared by keyword search, the grounding checks and the evaluation."""

from __future__ import annotations

import re
import unicodedata

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")

# Small English stop-word list: enough for BM25 on short policy sections.
_STOP_WORDS_TEXT = (
    "a about above after again all also am an and any are as at be been before being "
    "below between both but by can could did do does doing down during each few for from "
    "further had has have having he her here hers him his how i if in into is it its "
    "itself just me more most my no nor not now of off on once only or other our ours out "
    "over own same she should so some such than that the their theirs them then there "
    "these they this those through to too under until up very was we were what when where "
    "which while who whom why will with would you your"
)
STOP_WORDS = frozenset(_STOP_WORDS_TEXT.split())

NUMBER_WORDS = {
    # English
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "twelve": "12",
    "fifteen": "15",
    "twenty": "20",
    "thirty": "30",
    "sixty": "60",
    "ninety": "90",
    # French ("un" and "une" are left out: they are also articles)
    "deux": "2",
    "trois": "3",
    "quatre": "4",
    "cinq": "5",
    "sept": "7",
    "huit": "8",
    "neuf": "9",
    "dix": "10",
    "douze": "12",
    "quinze": "15",
    "vingt": "20",
    "trente": "30",
    "soixante": "60",
    "quatre-vingt-dix": "90",
}

_SPACES = re.compile(r"[   ]")
# 50,000 / 50 000 / 50.000 -> 50000 (a separator followed by exactly three digits)
_THOUSANDS = re.compile(r"(?<=\d)[,. ](?=\d{3}(?!\d))")
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")
_SPACE_BEFORE_PERCENT = re.compile(r"(?<=\d)\s+%")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize(text: str) -> str:
    """Lower-case text with unified spaces, numbers and percentages.

    `€50,000`, `50 000 €` and `EUR 50.000` all contain `50000` after
    normalisation, and `2 %` becomes `2%`.
    """
    text = _SPACES.sub(" ", unicodedata.normalize("NFKC", text))
    text = strip_accents(text).lower()
    text = _THOUSANDS.sub("", text)
    text = _DECIMAL_COMMA.sub(".", text)
    return _SPACE_BEFORE_PERCENT.sub("%", text)


def tokenize(text: str) -> list[str]:
    """Tokens for keyword search: normalised, stop words removed, stemmed."""
    words = [word for word in _TOKEN.findall(normalize(text)) if word not in STOP_WORDS]
    return _STEMMER.stemWords(words)


_MILLIONS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:millions?|mn|m)\b")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_LEADING_WORD = re.compile(r"[^a-z0-9]*([a-z]+)")


def _expand_millions(match: re.Match[str]) -> str:
    return str(round(float(match.group(1)) * 1_000_000))


def normalize_for_matching(text: str) -> str:
    """Normalisation used to check that an answer contains an expected fact:
    curly quotes become straight, number words become digits ("three" -> "3",
    "trois" -> "3") and "€2 million" or "€2M" becomes "€2000000"."""
    normalized = normalize(text.translate(_QUOTES))
    # Longest words first, so that "quatre-vingt-dix" is replaced before "vingt".
    for word in sorted(NUMBER_WORDS, key=len, reverse=True):
        normalized = re.sub(rf"\b{re.escape(word)}\b", NUMBER_WORDS[word], normalized)
    return _MILLIONS.sub(_expand_millions, normalized)


def _term_pattern(term: str) -> str:
    """Regex for one term of a fact.

    - `word` matches the whole word, plural included ("quote" -> "quotes"),
      but not a longer word ("not" does not match "note").
    - `stem*` matches any word starting with the stem ("prohibit*" -> "prohibited").
    - a number never matches a longer number ("5000" does not match "50000").
    """
    if term.endswith("*"):
        return re.escape(term[:-1])
    pattern = re.escape(term)
    if term[-1].isdigit():
        return pattern + r"(?![0-9]|[.,][0-9])"
    if term[-1].isalpha():
        return pattern + r"(?:s|es)?(?![a-z0-9])"
    return pattern


def _fact_patterns(fact: str) -> list[re.Pattern[str]]:
    patterns = []
    for alternative in fact.split("|"):
        terms = normalize_for_matching(alternative).split()
        if terms:
            body = r"[\s\-]+".join(_term_pattern(term) for term in terms)
            patterns.append(re.compile(r"(?<![a-z0-9])" + body))
    return patterns


def contains_fact(answer: str, fact: str) -> bool:
    """True when the answer states the expected fact.

    A fact lists alternatives separated by `|`; each alternative is a phrase
    whose words must appear in this order: `3 quote | 3 written quote` matches
    "three written quotes", and `site director` matches "the Site Director".
    """
    normalized_answer = normalize_for_matching(answer)
    return any(pattern.search(normalized_answer) for pattern in _fact_patterns(fact))


def answer_verdict(answer: str) -> str | None:
    """'yes' or 'no' when the answer opens with Yes/No (or Oui/Non), else None."""
    match = _LEADING_WORD.match(normalize_for_matching(answer))
    if not match:
        return None
    return {"yes": "yes", "oui": "yes", "no": "no", "non": "no"}.get(match.group(1))


_NUMBER = re.compile(r"(?<![a-z0-9])\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    """Numeric values written in a text, after normalisation: "€50,000",
    "50 000 €" and "fifty thousand" are not all equal, but the first two both
    give 50000.0, and "three" gives 3.0."""
    return {float(value) for value in _NUMBER.findall(normalize_for_matching(text))}
