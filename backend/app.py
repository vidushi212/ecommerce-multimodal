"""
Flask application entry point.

Usage
-----
    python app.py                  # development server
    gunicorn app:app               # production (gunicorn)
"""
import logging
import os
import sys
from pathlib import Path

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# Make sure backend/ is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CORS_ORIGINS,
    CSV_PATH,
    EMBEDDINGS_FILE,
    EMBEDDINGS_IDS_FILE,
    FLASK_DEBUG,
    FLASK_HOST,
    FLASK_PORT,
    IMAGES_DIR,
)
from routes.health import health_bp
from routes.search import search_bp
from utils.data_loader import load_data

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def _log_startup_status() -> None:
    csv_exists = Path(CSV_PATH).exists()
    images_dir = Path(IMAGES_DIR)
    embeddings_exist = (
        Path(EMBEDDINGS_FILE).exists() and Path(EMBEDDINGS_IDS_FILE).exists()
    )
    image_count = (
        len([
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])
        if images_dir.exists()
        else 0
    )

    logger.info(
        "Startup data status | csv_exists=%s csv_path=%s images_dir_exists=%s local_images=%d embeddings_cache=%s",
        csv_exists,
        CSV_PATH,
        images_dir.exists(),
        image_count,
        embeddings_exist,
    )

    try:
        products_count = len(load_data())
        logger.info("Loaded product metadata: %d rows", products_count)
    except Exception:
        logger.exception("Failed to load product metadata at startup.")

    try:
        from routes.search import _get_image_model
        image_status = _get_image_model().status()
        logger.info(
            "Image retrieval status | ready=%s products=%d embeddings=%d clip_loaded=%s faiss=%s init_error=%s",
            image_status["ready"],
            image_status["products_count"],
            image_status["embeddings_count"],
            image_status["clip_loaded"],
            image_status["faiss_enabled"],
            image_status["init_error"],
        )
    except Exception:
        logger.exception("Image retrieval model initialization failed at startup.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)

    # CORS
    origins = (
        CORS_ORIGINS.split(",")
        if "," in CORS_ORIGINS
        else CORS_ORIGINS
    )
    CORS(app, resources={r"/*": {"origins": origins}})

    # Blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(search_bp, url_prefix="/search")

    # Serve local product images
    @app.route("/api/images/<product_id>")
    def serve_image(product_id: str):
        """Serve a local product image by product ID.

        Path traversal is prevented at two layers:
        1. The product_id is sanitised to alphanumeric chars and hyphens only
           before being used as a filename, so no directory separators or
           relative path components can be injected.
        2. send_from_directory() independently verifies that the resolved
           file path stays within the declared images directory and raises
           werkzeug.exceptions.NotFound (→ 404) if not.
        """
        # Sanitise: only alphanumeric chars and hyphens to prevent path traversal
        safe_id = "".join(c for c in product_id if c.isalnum() or c == "-")
        if not safe_id:
            return jsonify({"error": "invalid id"}), 400
        # send_from_directory handles both path-traversal prevention and 404s.
        from werkzeug.exceptions import NotFound
        try:
            return send_from_directory(
                str(IMAGES_DIR), f"{safe_id}.png", mimetype="image/png"
            )
        except NotFound:
            return jsonify({"error": "image not found"}), 404

    _log_startup_status()
    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
    )
