"""
Health-check route.
"""
import logging
from pathlib import Path
from flask import Blueprint, jsonify
from config import CSV_PATH, EMBEDDINGS_FILE, EMBEDDINGS_IDS_FILE, IMAGES_DIR
from utils.data_loader import load_data

logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)


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
    dataset_error = None
    try:
        products_count = len(load_data())
    except Exception as exc:
        dataset_error = str(exc)

    image_status = {
        "ready": False,
        "products_count": products_count,
        "embeddings_count": 0,
        "clip_loaded": False,
        "faiss_enabled": False,
        "init_error": "Image model not initialized yet.",
    }
    try:
        from routes.search import _get_image_model
        image_status = _get_image_model().status()
    except Exception as exc:
        image_status["init_error"] = f"Initialization failed: {exc}"

    issues = []
    if dataset_error:
        issues.append(f"Dataset load error: {dataset_error}")
    if not Path(CSV_PATH).exists():
        issues.append(f"Dataset CSV missing: {CSV_PATH}")
    if not images_dir.exists():
        issues.append(f"Images directory missing: {IMAGES_DIR}")
    elif not image_files:
        issues.append(f"No local images found in: {IMAGES_DIR}")
    if not Path(EMBEDDINGS_FILE).exists() or not Path(EMBEDDINGS_IDS_FILE).exists():
        issues.append("Embeddings cache files are missing.")
    if image_status.get("init_error"):
        issues.append(f"Image retrieval init error: {image_status['init_error']}")

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
                "GET  /health": "Liveness probe",
                "GET  /health (extended)": "Liveness + image-search readiness diagnostics",
                "POST /search/text": "Text-based product search (TF-IDF)",
                "POST /search/image": "Image-based product search (CLIP + FAISS)",
                "POST /search/hybrid": "Hybrid text + image search",
                "POST /search/feedback": "Rocchio relevance feedback",
                "POST /search/evaluate": "Evaluate retrieval quality",
                "GET  /api/images/<id>": "Serve local product image",
            },
        }
    )
