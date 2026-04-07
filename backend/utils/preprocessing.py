"""
Text and image preprocessing utilities.

No third-party NLP libraries are used here; all text helpers are
implemented in pure Python to avoid dependency vulnerabilities.
"""
import io
import logging
import re
import unicodedata

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# English stop-words (standard list, no external dependency)
# ---------------------------------------------------------------------------
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "get", "got", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "will", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves",
})


# ---------------------------------------------------------------------------
# Minimal Porter stemmer (pure Python, no external dependency)
# ---------------------------------------------------------------------------

def _is_vowel(ch: str) -> bool:
    return ch in "aeiou"


def _measure(stem: str) -> int:
    """Count VC sequences (Porter 'm') in *stem*."""
    n, in_vowel = 0, False
    for ch in stem:
        v = _is_vowel(ch)
        if v:
            in_vowel = True
        elif in_vowel:
            n += 1
            in_vowel = False
    return n


def _has_vowel(stem: str) -> bool:
    return any(_is_vowel(c) for c in stem)


def _ends_double_consonant(word: str) -> bool:
    return (
        len(word) >= 2
        and word[-1] == word[-2]
        and not _is_vowel(word[-1])
    )


def _ends_cvc(word: str) -> bool:
    """True if word ends consonant-vowel-consonant where last c ∉ {w,x,y}."""
    if len(word) < 3:
        return False
    c1, v, c2 = word[-3], word[-2], word[-1]
    return (
        not _is_vowel(c1)
        and _is_vowel(v)
        and not _is_vowel(c2)
        and c2 not in "wxy"
    )


def _porter_stem(word: str) -> str:
    """
    Simplified Porter stemmer (steps 1a–5b).
    Handles the most common English inflections without any dependencies.
    """
    if len(word) <= 2:
        return word

    # Step 1a
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # Step 1b
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith("ed"):
        stem = word[:-2]
        if _has_vowel(stem):
            word = stem
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
            elif _ends_double_consonant(word) and word[-1] not in "lsz":
                word = word[:-1]
            elif _measure(word) == 1 and _ends_cvc(word):
                word += "e"
    elif word.endswith("ing"):
        stem = word[:-3]
        if _has_vowel(stem):
            word = stem
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
            elif _ends_double_consonant(word) and word[-1] not in "lsz":
                word = word[:-1]
            elif _measure(word) == 1 and _ends_cvc(word):
                word += "e"

    # Step 1c
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"

    # Step 2
    _step2 = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
        ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
        ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
        ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
        ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
        ("biliti", "ble"),
    ]
    for suffix, replacement in _step2:
        if word.endswith(suffix) and _measure(word[: -len(suffix)]) > 0:
            word = word[: -len(suffix)] + replacement
            break

    # Step 3
    _step3 = [
        ("icate", "ic"), ("ative", ""), ("alize", "al"),
        ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
    ]
    for suffix, replacement in _step3:
        if word.endswith(suffix) and _measure(word[: -len(suffix)]) > 0:
            word = word[: -len(suffix)] + replacement
            break

    # Step 4
    _step4 = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
        "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
    ]
    for suffix in _step4:
        stem = word[: -len(suffix)]
        if word.endswith(suffix) and _measure(stem) > 1:
            if suffix == "ion" and stem and stem[-1] in "st":
                word = stem
            elif suffix != "ion":
                word = stem
            break

    # Step 5a
    if word.endswith("e"):
        stem = word[:-1]
        if _measure(stem) > 1:
            word = stem
        elif _measure(stem) == 1 and not _ends_cvc(stem):
            word = stem

    # Step 5b
    if _measure(word) > 1 and _ends_double_consonant(word) and word.endswith("l"):
        word = word[:-1]

    return word


# ---------------------------------------------------------------------------
# Fashion-domain synonym map for query expansion
# ---------------------------------------------------------------------------
FASHION_SYNONYMS: dict[str, list[str]] = {
    "dress": ["gown", "frock", "outfit"],
    "shirt": ["top", "blouse", "tee", "kurta"],
    "shoes": ["footwear", "sneakers", "boots", "sandals", "heels"],
    "pants": ["trousers", "jeans", "leggings", "palazzos"],
    "bag": ["handbag", "purse", "tote", "clutch", "backpack"],
    "saree": ["sari", "lehenga", "dupatta"],
    "red": ["crimson", "maroon", "scarlet"],
    "blue": ["navy", "cobalt", "indigo", "denim"],
    "black": ["dark", "jet"],
    "white": ["ivory", "cream", "off-white"],
    "jacket": ["blazer", "coat", "hoodie"],
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, remove special characters, strip extra whitespace."""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Whitespace tokenizer with optional stop-word removal."""
    tokens = clean_text(text).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS]
    return tokens


def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply the built-in Porter stemmer to a list of tokens."""
    return [_porter_stem(t) for t in tokens]


def expand_query(query: str) -> str:
    """
    Expand a query by appending fashion-domain synonyms.
    Returns the augmented query string.
    """
    tokens = tokenize(query, remove_stopwords=False)
    expanded = list(tokens)
    for token in tokens:
        if token in FASHION_SYNONYMS:
            expanded.extend(FASHION_SYNONYMS[token])
    return " ".join(expanded)


def load_image_from_path(path: str) -> Image.Image | None:
    """Load a PIL Image from a local file path."""
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        logger.debug("Cannot open local image %s: %s", path, exc)
        return None


def load_image_from_url(url: str, timeout: int = 10) -> Image.Image | None:
    """Download and return a PIL Image from a URL."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        logger.debug("Cannot fetch image from %s: %s", url, exc)
        return None


def load_image_from_bytes(data: bytes) -> Image.Image | None:
    """Load a PIL Image from raw bytes."""
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        logger.debug("Cannot load image from bytes: %s", exc)
        return None
