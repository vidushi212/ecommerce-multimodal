"""
Application configuration.
"""
import os
from pathlib import Path

# Base directory (backend/)
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# CSV dataset – fall back to the repo-root copy when the data/ copy is absent
_local_csv = DATA_DIR / "products.csv"
_root_csv = BASE_DIR.parent / "Data - Copy.csv"
CSV_PATH = str(_local_csv if _local_csv.exists() else _root_csv)

# CLIP model name
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

# Image embedding cache
EMBEDDINGS_FILE = str(EMBEDDINGS_DIR / "image_embeddings.npy")
EMBEDDINGS_IDS_FILE = str(EMBEDDINGS_DIR / "embedding_ids.npy")

# TF-IDF cache
TFIDF_CACHE_FILE = str(EMBEDDINGS_DIR / "tfidf_matrix.npz")
TFIDF_VECTORIZER_FILE = str(EMBEDDINGS_DIR / "tfidf_vectorizer.pkl")

# Default top-K results
DEFAULT_TOP_K = 12

# Maximum image dimension for CLIP pre-processing
MAX_IMAGE_SIZE = (224, 224)

# FLASK settings
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# CORS allowed origins (comma-separated)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# Maximum number of products to embed images for (set 0 for all)
MAX_IMAGE_EMBED = int(os.getenv("MAX_IMAGE_EMBED", "0"))
