"""
Text and image preprocessing utilities.
"""
import io
import logging
import re
import unicodedata

import nltk
import requests
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from PIL import Image

logger = logging.getLogger(__name__)

# Download required NLTK data (silent if already present)
for _pkg in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

_stemmer = PorterStemmer()

try:
    _stop_words = set(stopwords.words("english"))
except Exception:
    _stop_words = set()

# Fashion-domain synonym map for query expansion
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


def clean_text(text: str) -> str:
    """
    Lowercase, remove special characters, strip extra spaces.
    """
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Simple whitespace tokenizer with optional stopword removal."""
    tokens = clean_text(text).split()
    if remove_stopwords and _stop_words:
        tokens = [t for t in tokens if t not in _stop_words]
    return tokens


def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply Porter stemming to a list of tokens."""
    return [_stemmer.stem(t) for t in tokens]


def expand_query(query: str) -> str:
    """
    Expand a query by appending domain synonyms and WordNet synonyms.
    Returns the augmented query string.
    """
    tokens = tokenize(query, remove_stopwords=False)
    expanded = list(tokens)

    for token in tokens:
        # Domain-specific synonyms
        if token in FASHION_SYNONYMS:
            expanded.extend(FASHION_SYNONYMS[token])

        # WordNet synonyms (first synset only to keep it focused)
        try:
            synsets = wordnet.synsets(token)
            if synsets:
                for lemma in synsets[0].lemmas()[:3]:
                    word = lemma.name().replace("_", " ").lower()
                    if word != token:
                        expanded.append(word)
        except Exception:
            pass

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
