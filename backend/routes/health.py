"""
Health-check route.
"""
import logging
from pathlib import Path
from flask import Blueprint, jsonify
from config import CSV_PATH, EMBEDDINGS_FILE, EMBEDDINGS_IDS_FILE, IMAGES_DIR
from models.image_retrieval import ImageRetrieval
from utils.data_loader import load_data

logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)
_health_image_model: ImageRetrieval | None = None


def _get_health_image_model() -> ImageRetrieval:
    global _health_image_model
    if _health_image_model is None:
        _health_image_model = ImageRetrieval()
        _health_image_model.initialize()
    return _health_image_model


@health_bp.route("/health", methods=["GET"])
def health():
    """Liveness + readiness diagnostics."""
    images_dir = Path(IMAGES_DIR)
    image_files = []
    if images_dir.exists():
        image_files = [
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]

    products_count = 0
    dataset_error = False
    try:
        products_count = len(load_data())
    except Exception:
        dataset_error = True
        logger.exception("Health check dataset load failed.")

    image_status = {
        "ready": False,
        "products_count": products_count,
        "embeddings_count": 0,
        "clip_loaded": False,
        "faiss_enabled": False,
        "has_init_error": False,
    }
    try:
        internal_status = _get_health_image_model().status()
        image_status = {
            "ready": bool(internal_status.get("ready")),
            "products_count": int(internal_status.get("products_count", 0)),
            "embeddings_count": int(internal_status.get("embeddings_count", 0)),
            "clip_loaded": bool(internal_status.get("clip_loaded")),
            "faiss_enabled": bool(internal_status.get("faiss_enabled")),
            "has_init_error": bool(internal_status.get("init_error")),
        }
    except Exception:
        logger.exception("Health check image model initialization failed.")
        image_status["has_init_error"] = True

    issues = []
    if dataset_error:
        issues.append("Dataset load error. Check backend logs.")
    if not Path(CSV_PATH).exists():
        issues.append(f"Dataset CSV missing: {CSV_PATH}")
    if not images_dir.exists():
        issues.append(f"Images directory missing: {IMAGES_DIR}")
    elif not image_files:
        issues.append(f"No local images found in: {IMAGES_DIR}")
    if not Path(EMBEDDINGS_FILE).exists() or not Path(EMBEDDINGS_IDS_FILE).exists():
        issues.append("Embeddings cache files are missing.")
    if image_status.get("has_init_error"):
        issues.append("Image retrieval initialization error. Check backend logs.")

    overall = "ok" if not issues else "degraded"
    return jsonify({
        "status": overall,
        "issues": issues,
        "checks": {
            "csv_exists": Path(CSV_PATH).exists(),
            "images_dir_exists": images_dir.exists(),
            "local_images_count": len(image_files),
            "embeddings_cache_exists": (
                Path(EMBEDDINGS_FILE).exists() and Path(EMBEDDINGS_IDS_FILE).exists()
            ),
            "products_count": products_count,
        },
        "image_retrieval": image_status,
    })


@health_bp.route("/", methods=["GET"])
def index():
    """Root endpoint with API overview."""
    return jsonify(
        {
            "name": "Multimodal E-Commerce Search API",
            "version": "1.0.0",
            "endpoints": {
                "GET  /health": "Liveness + readiness diagnostics",
                "POST /search/text": "Text-based product search (TF-IDF)",
                "POST /search/image": "Image-based product search (CLIP + FAISS)",
                "POST /search/hybrid": "Hybrid text + image search",
                "POST /search/feedback": "Rocchio relevance feedback",
                "POST /search/evaluate": "Evaluate retrieval quality",
                "GET  /api/images/<id>": "Serve local product image",
            },
        }
    )
