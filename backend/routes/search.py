"""
Search routes: text search, image search, hybrid search, and relevance feedback.
"""
import logging

from flask import Blueprint, jsonify, request

from config import DEFAULT_TOP_K
from models.text_retrieval import TextRetrieval
from models.image_retrieval import ImageRetrieval
from models.hybrid_retrieval import HybridRetrieval
from utils.evaluation import evaluate_retrieval
from utils.preprocessing import load_image_from_bytes

logger = logging.getLogger(__name__)

search_bp = Blueprint("search", __name__, url_prefix="/search")

# Shared model instances (lazy-initialised)
_text_model: TextRetrieval | None = None
_image_model: ImageRetrieval | None = None
_hybrid_model: HybridRetrieval | None = None


def _get_text_model() -> TextRetrieval:
    global _text_model
    if _text_model is None:
        _text_model = TextRetrieval()
        _text_model.initialize()
    return _text_model


def _get_image_model() -> ImageRetrieval:
    global _image_model
    if _image_model is None:
        _image_model = ImageRetrieval()
        _image_model.initialize()
    return _image_model


def _get_hybrid_model() -> HybridRetrieval:
    global _hybrid_model
    if _hybrid_model is None:
        _hybrid_model = HybridRetrieval(
            text_model=_get_text_model(),
            image_model=_get_image_model(),
        )
    return _hybrid_model


def _parse_filters(args) -> dict:
    """Extract common filter parameters from request args."""
    filters: dict = {}
    brand = args.get("brand") or args.get("brand_filter")
    if brand:
        filters["brand_filter"] = brand

    min_p = args.get("min_price")
    max_p = args.get("max_price")
    if min_p is not None:
        try:
            filters["min_price"] = float(min_p)
        except ValueError:
            pass
    if max_p is not None:
        try:
            filters["max_price"] = float(max_p)
        except ValueError:
            pass
    return filters


def _parse_top_k(args) -> int:
    try:
        return int(args.get("top_k", DEFAULT_TOP_K))
    except (TypeError, ValueError):
        return DEFAULT_TOP_K


# ------------------------------------------------------------------
# POST /search/text
# ------------------------------------------------------------------

@search_bp.route("/text", methods=["POST"])
def text_search():
    """
    Text-based product search using TF-IDF.

    Request body (JSON):
        {
            "query": "black dress",
            "top_k": 12,
            "expand": true,
            "brand_filter": "Nike",
            "min_price": 100,
            "max_price": 2000
        }

    Returns:
        { "results": [...], "query": "...", "count": N }
    """
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    top_k = int(data.get("top_k", DEFAULT_TOP_K))
    expand = bool(data.get("expand", True))
    filters = {
        k: v
        for k, v in {
            "brand_filter": data.get("brand_filter"),
            "min_price": data.get("min_price"),
            "max_price": data.get("max_price"),
        }.items()
        if v is not None
    }

    try:
        results = _get_text_model().search(
            query, top_k=top_k, expand=expand, **filters
        )
        return jsonify({"results": results, "query": query, "count": len(results)})
    except Exception:
        logger.exception("Text search error")
        return jsonify({"error": "An internal error occurred during text search."}), 500

# In backend/routes/search.py, add this new route after @search_bp.route("/image")

# ------------------------------------------------------------------
# POST /search/image-with-description (DEPRECATED)
# ------------------------------------------------------------------

@search_bp.route("/image-with-description", methods=["POST"])
def image_with_description_search():
    return jsonify({
        "error": "This endpoint is deprecated. Use /search/hybrid with query + image and optional description."
    }), 410

# ------------------------------------------------------------------
# POST /search/image
# ------------------------------------------------------------------

@search_bp.route("/image", methods=["POST"])
def image_search():
    """
    Image-based product search using CLIP + FAISS.

    Request: multipart/form-data with field "image".
    Optional query params: top_k, brand, min_price, max_price.

    Returns:
        { "results": [...], "count": N }
    """
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "uploaded file is empty"}), 400

    top_k = _parse_top_k(request.args)
    filters = _parse_filters(request.args)

    try:
        results = _get_image_model().search_from_bytes(
            image_bytes, top_k=top_k, **filters
        )
        return jsonify({"results": results, "count": len(results)})
    except Exception:
        logger.exception("Image search error")
        return jsonify({"error": "An internal error occurred during image search."}), 500


# ------------------------------------------------------------------
# POST /search/hybrid
# ------------------------------------------------------------------

@search_bp.route("/hybrid", methods=["POST"])
def hybrid_search():
    """
    Hybrid text + image search.

    Request: multipart/form-data with:
        - "image": image file
        - "query": text field
        - "description": optional text field
        - "alpha": float 0-1 (default 0.5)
    Optional query params: top_k, brand, min_price, max_price.

    Returns:
        { "results": [...], "query": "...", "count": N, "method": "..." }
    """
    query = (request.form.get("query") or "").strip()
    description = (request.form.get("description") or "").strip()
    if "image" not in request.files or not query:
        return jsonify({"error": "Both 'query' and 'image' are required"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "uploaded file is empty"}), 400

    img = load_image_from_bytes(image_bytes)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    top_k = _parse_top_k(request.args)
    try:
        alpha = float(request.form.get("alpha", 0.5))
    except (TypeError, ValueError):
        alpha = 0.5
    alpha = max(0.0, min(1.0, alpha))
    filters = _parse_filters(request.args)
    effective_query = query if not description else f"{query} {description}"
    method = "hybrid_refined" if description else "hybrid"

    try:
        results = _get_hybrid_model().search(
            effective_query, img, top_k=top_k, alpha=alpha, **filters
        )
        response = {
            "results": results,
            "query": query,
            "description": description or None,
            "alpha": alpha,
            "count": len(results),
            "method": method,
            "filters": {
                "brand": filters.get("brand_filter"),
                "min_price": filters.get("min_price"),
                "max_price": filters.get("max_price"),
            },
        }
        return jsonify(response)
    except Exception:
        logger.exception("Hybrid search error")
        return jsonify({"error": "An internal error occurred during hybrid search."}), 500


# ------------------------------------------------------------------
# POST /search/feedback
# ------------------------------------------------------------------

@search_bp.route("/feedback", methods=["POST"])
def relevance_feedback():
    """
    Rocchio relevance feedback for text search.

    Request body (JSON):
        {
            "query": "blue dress",
            "positive_ids": ["123", "456"],
            "negative_ids": ["789"]
        }

    Returns:
        { "results": [...], "count": N }
    """
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    positive_ids = data.get("positive_ids") or []
    negative_ids = data.get("negative_ids") or []

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        results = _get_text_model().rocchio_update(
            query, positive_ids, negative_ids
        )
        return jsonify({"results": results, "count": len(results)})
    except Exception:
        logger.exception("Relevance feedback error")
        return jsonify({"error": "An internal error occurred during relevance feedback."}), 500


# ------------------------------------------------------------------
# POST /search/evaluate
# ------------------------------------------------------------------

@search_bp.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Evaluate retrieval quality for a text query.

    Request body (JSON):
        {
            "query": "black kurta",
            "relevant_ids": ["1", "2", "3"],
            "k_values": [1, 5, 10]
        }

    Returns:
        { "metrics": { "precision@5": 0.4, ... } }
    """
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    relevant_ids = data.get("relevant_ids") or []
    k_values = data.get("k_values") or [1, 5, 10, 20]

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        results = _get_text_model().search(query, top_k=max(k_values))
        retrieved_ids = [r["id"] for r in results]
        metrics = evaluate_retrieval(retrieved_ids, relevant_ids, k_values)
        return jsonify({"metrics": metrics, "retrieved_ids": retrieved_ids})
    except Exception:
        logger.exception("Evaluation error")
        return jsonify({"error": "An internal error occurred during evaluation."}), 500
