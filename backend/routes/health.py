"""
Health-check route.
"""
import logging
from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    """Simple liveness probe."""
    return jsonify({"status": "ok"})


@health_bp.route("/", methods=["GET"])
def index():
    """Root endpoint with API overview."""
    return jsonify(
        {
            "name": "Multimodal E-Commerce Search API",
            "version": "1.0.0",
            "endpoints": {
                "GET  /health": "Liveness probe",
                "POST /search/text": "Text-based product search (TF-IDF)",
                "POST /search/image": "Image-based product search (CLIP + FAISS)",
                "POST /search/hybrid": "Hybrid text + image search",
                "POST /search/feedback": "Rocchio relevance feedback",
                "POST /search/evaluate": "Evaluate retrieval quality",
                "GET  /api/images/<id>": "Serve local product image",
            },
        }
    )
