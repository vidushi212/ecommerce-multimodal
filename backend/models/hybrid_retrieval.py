"""
Hybrid (text + image) retrieval model.

Combines normalised scores from TextRetrieval and ImageRetrieval using a
weighted sum.  This allows a single query to leverage both modalities
simultaneously.
"""
import logging
from PIL import Image

from config import DEFAULT_TOP_K
from models.text_retrieval import TextRetrieval
from models.image_retrieval import ImageRetrieval

logger = logging.getLogger(__name__)


class HybridRetrieval:
    """
    Fuses text and image retrieval scores.

    score_hybrid = alpha * score_text + (1 - alpha) * score_image
    """

    def __init__(
        self,
        text_model: TextRetrieval | None = None,
        image_model: ImageRetrieval | None = None,
    ) -> None:
        self.text_model = text_model or TextRetrieval()
        self.image_model = image_model or ImageRetrieval()

    def initialize(self) -> None:
        """Initialise both underlying models."""
        self.text_model.initialize()
        self.image_model.initialize()

    def search(
        self,
        query: str,
        image: Image.Image,
        top_k: int = DEFAULT_TOP_K,
        alpha: float = 0.5,
        brand_filter: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
    ) -> list[dict]:
        """
        Hybrid search using both text and image queries.

        Args:
            query:        Text query string.
            image:        Query PIL Image.
            top_k:        Number of results to return.
            alpha:        Weight for text score (0–1).
            brand_filter: Optional brand filter.
            min_price:    Optional minimum sold price.
            max_price:    Optional maximum sold price.

        Returns:
            Merged and re-ranked list of product dicts.
        """
        fetch_k = top_k * 3  # over-fetch so fusion has enough candidates

        text_results = self.text_model.search(
            query,
            top_k=fetch_k,
            brand_filter=brand_filter,
            min_price=min_price,
            max_price=max_price,
        )
        image_results = self.image_model.search(
            image,
            top_k=fetch_k,
            brand_filter=brand_filter,
            min_price=min_price,
            max_price=max_price,
        )

        # Build score maps
        text_score: dict[str, float] = {
            p["id"]: p["score"] for p in text_results
        }
        image_score: dict[str, float] = {
            p["id"]: p["score"] for p in image_results
        }

        # Union of all candidate IDs
        all_ids = set(text_score.keys()) | set(image_score.keys())

        # Build lookup of product metadata
        product_map: dict[str, dict] = {}
        for p in text_results + image_results:
            product_map[p["id"]] = p

        # Normalise scores to [0, 1]
        def _norm(scores: dict[str, float]) -> dict[str, float]:
            if not scores:
                return {}
            max_s = max(scores.values()) or 1.0
            return {k: v / max_s for k, v in scores.items()}

        t_norm = _norm(text_score)
        i_norm = _norm(image_score)

        # Fuse
        fused: list[tuple[str, float]] = []
        for pid in all_ids:
            t = t_norm.get(pid, 0.0)
            i = i_norm.get(pid, 0.0)
            combined = alpha * t + (1.0 - alpha) * i
            fused.append((pid, combined))

        fused.sort(key=lambda x: x[1], reverse=True)

        results = []
        for pid, score in fused[:top_k]:
            product = dict(product_map[pid])
            product["score"] = round(score, 4)
            results.append(product)

        return results
