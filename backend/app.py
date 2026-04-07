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

from flask import Flask, send_file, jsonify
from flask_cors import CORS

# Make sure backend/ is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from config import CORS_ORIGINS, FLASK_DEBUG, FLASK_HOST, FLASK_PORT, IMAGES_DIR
from routes.health import health_bp
from routes.search import search_bp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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
        """Serve a local product image by product ID."""
        # Sanitise: only alphanumeric ids to prevent path traversal
        safe_id = "".join(c for c in product_id if c.isalnum() or c == "-")
        img_path = Path(IMAGES_DIR) / f"{safe_id}.png"
        if img_path.exists():
            return send_file(str(img_path), mimetype="image/png")
        return jsonify({"error": "image not found"}), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
    )
