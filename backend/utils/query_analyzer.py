"""
Heuristic query analyzer for adaptive alpha selection in hybrid search.
"""
from __future__ import annotations

from typing import Any

COLOR_WORDS = {
    "black", "blue", "brown", "cream", "gold", "green", "grey", "gray",
    "maroon", "navy", "orange", "pink", "purple", "red", "silver",
    "white", "yellow",
}
SIZE_FIT_WORDS = {
    "xs", "s", "m", "l", "xl", "xxl", "slim", "oversized", "regular",
    "loose", "fitted",
}
MATERIAL_WORDS = {
    "cotton", "denim", "linen", "polyester", "rayon", "satin", "silk",
    "wool",
}
OCCASION_WORDS = {
    "casual", "ethnic", "formal", "office", "party", "wedding",
}


def _tokenize(query: str) -> list[str]:
    return [t.strip(" ,.-_()[]{}").lower() for t in query.split() if t.strip()]


def _detect(tokens: list[str], vocab: set[str]) -> list[str]:
    return sorted({t for t in tokens if t in vocab})


def analyze_query_details(query: str, has_image: bool) -> dict[str, Any]:
    """
    Return alpha and human-readable reasoning based on query characteristics.
    """
    normalized_query = (query or "").strip()
    tokens = _tokenize(normalized_query)
    query_len = len(normalized_query)

    colors = _detect(tokens, COLOR_WORDS)
    sizes = _detect(tokens, SIZE_FIT_WORDS)
    materials = _detect(tokens, MATERIAL_WORDS)
    occasions = _detect(tokens, OCCASION_WORDS)

    has_color = bool(colors)
    has_size = bool(sizes)
    has_material = bool(materials)
    has_occasion = bool(occasions)

    # Base by modality
    alpha = 0.50 if has_image else 0.85
    reason = "Balanced multimodal search baseline."

    # Length-sensitive adjustments
    if has_image and 0 < query_len <= 10:
        alpha = max(alpha, 0.80)
        reason = "Short query with image; increasing text weight for specificity."
    elif not has_image and query_len > 20:
        alpha = max(alpha, 0.90)
        reason = "Detailed text-only query; strongly text-dominant."
    elif has_image and query_len > 20:
        alpha = max(alpha, 0.60)
        reason = "Detailed query with image; keeping both modalities active."

    # Attribute nudges
    if has_color:
        alpha -= 0.10
    if has_size:
        alpha += 0.10
    if has_material:
        alpha += 0.05
    if has_occasion:
        alpha += 0.05

    # Final prioritisation rules
    if not has_image:
        alpha = max(alpha, 0.85)
        reason = "No image uploaded; text-dominant weighting."
    elif query_len < 5:
        alpha = max(alpha, 0.85)
        reason = "Very short query with image; using image-heavy guidance profile."
    elif has_size:
        alpha = max(alpha, 0.70)
        reason = "Size/fit terms detected; prioritizing text semantics."
    elif has_color and has_material:
        alpha = max(alpha, 0.65)
        reason = "Color + material detected; using balanced text/image weighting."
    elif has_color and not (has_material or has_occasion):
        alpha = max(alpha, 0.60)
        reason = "Primarily visual color intent detected; keeping balance."
    elif query_len > 20:
        alpha = max(alpha, 0.55)
        reason = "Detailed query with image; both modalities are informative."
    else:
        alpha = max(alpha, 0.50)
        reason = "Default balanced multimodal weighting."

    alpha = max(0.0, min(1.0, round(alpha, 2)))

    detected = {
        "colors": colors,
        "sizes_or_fit": sizes,
        "materials": materials,
        "occasions": occasions,
    }

    return {
        "alpha": alpha,
        "reason": reason,
        "detected": detected,
        "query": normalized_query,
        "has_image": has_image,
    }


def analyze_query(query: str, has_image: bool) -> float:
    """Return only the adaptive alpha score."""
    return float(analyze_query_details(query, has_image)["alpha"])
