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
# POST /search/image-with-description (NEW ENDPOINT)
# ------------------------------------------------------------------

@search_bp.route("/image-with-description", methods=["POST"])
def image_with_description_search():
    """
    🆕 NOVEL FEATURE: Combined Image Search with optional TEXT DESCRIPTION.
    
    This endpoint enables users to:
      1. Upload an image (required)
      2. Add a text description to refine results (optional)
      3. Get results ranked by: 70% image similarity + 30% description match
    
    Example use case:
      - User uploads photo of a dress
      - User adds description: "I want it in BLACK COLOR, SILK MATERIAL"
      - System combines visual similarity + text attribute matching
    
    Request: multipart/form-data
      Fields:
        - "image": image file (required)
        - "description": text description (optional)
      Query params:
        - "top_k": number of results (default 12)
        - "brand": brand filter
        - "min_price": minimum price filter
        - "max_price": maximum price filter
    
    Returns:
      {
        "results": [...],
        "count": 12,
        "description": "black color, silk material" (if provided),
        "method": "combined" (if description provided) or "image_only"
      }
    """
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "uploaded file is empty"}), 400

    # Get optional description
    description = request.form.get("description", "").strip()
    top_k = _parse_top_k(request.args)
    filters = _parse_filters(request.args)

    try:
        image_model = _get_image_model()
        text_model = _get_text_model()
        
        # Get image search results (fetch 2x to have enough candidates)
        image_results = image_model.search_from_bytes(
            image_bytes, top_k=top_k * 2, **filters
        )
        
        if not image_results:
            return jsonify({
                "results": [],
                "count": 0,
                "description": description if description else None,
                "method": "combined" if description else "image_only"
            })
        
        # If description provided, also search by text and combine
        if description:
            logger.info("🔍 Image+Description search: description='%s'", description)
            
            text_results = text_model.search(
                description, top_k=top_k * 2, **filters
            )
            
            # Build score maps
            image_score_map = {p["id"]: p["score"] for p in image_results}
            text_score_map = {p["id"]: p["score"] for p in text_results}
            
            # Get union of all IDs
            all_ids = set(image_score_map.keys()) | set(text_score_map.keys())
            
            # Normalize scores to [0, 1]
            def normalize_scores(score_map):
                if not score_map:
                    return {}
                max_score = max(score_map.values()) or 1.0
                return {k: v / max_score for k, v in score_map.items()}
            
            img_norm = normalize_scores(image_score_map)
            txt_norm = normalize_scores(text_score_map)
            
            # Build combined scores: 70% image + 30% text
            combined_scores = {}
            for pid in all_ids:
                img_score = img_norm.get(pid, 0.0)
                txt_score = txt_norm.get(pid, 0.0)
                combined_score = 0.7 * img_score + 0.3 * txt_score
                combined_scores[pid] = combined_score
            
            # Build product metadata map
            product_map = {}
            for p in image_results + text_results:
                if p["id"] not in product_map:
                    product_map[p["id"]] = p
            
            # Sort by combined score and take top-K
            sorted_ids = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            results = []
            for pid, score in sorted_ids[:top_k]:
                product = dict(product_map[pid])
                product["score"] = round(score, 4)
                results.append(product)
            
            logger.info(
                "✅ Image+Description: combined %d image + %d text results -> %d final",
                len(image_results),
                len(text_results),
                len(results)
            )
            
            return jsonify({
                "results": results,
                "count": len(results),
                "description": description,
                "method": "combined"
            })
        else:
            # No description, just return image results
            return jsonify({
                "results": image_results[:top_k],
                "count": len(image_results[:top_k]),
                "description": None,
                "method": "image_only"
            })
    
    except Exception:
        logger.exception("Image+Description search error")
        return jsonify({"error": "An internal error occurred during search."}), 500

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
        - "alpha": float 0-1 (default 0.5)
    Optional query params: top_k, brand, min_price, max_price.

    Returns:
        { "results": [...], "query": "...", "count": N }
    """
    query = (request.form.get("query") or "").strip()
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
    alpha = float(request.form.get("alpha", 0.5))
    filters = _parse_filters(request.args)

    try:
        results = _get_hybrid_model().search(
            query, img, top_k=top_k, alpha=alpha, **filters
        )
        return jsonify({"results": results, "query": query, "count": len(results)})
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
