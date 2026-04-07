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

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
    )
