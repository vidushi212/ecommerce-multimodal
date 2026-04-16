"""
CLIP-based image retrieval model.

Architecture
------------
* Encodes product images (local files or remote URLs) with OpenAI CLIP.
* Stores embeddings in a FAISS flat-L2 index (or falls back to numpy).
* At query time, encodes the uploaded image with the same CLIP encoder
  and performs approximate nearest-neighbour search.

Caching
-------
Embeddings and product IDs are cached to disk so that subsequent restarts
do not require re-encoding the entire catalogue.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from config import (
    CLIP_MODEL_NAME,
    DEFAULT_TOP_K,
    EMBEDDINGS_FILE,
    EMBEDDINGS_IDS_FILE,
    IMAGES_DIR,
    MAX_IMAGE_EMBED,
)
from utils.data_loader import load_data, product_row_to_dict
from utils.preprocessing import (
    load_image_from_bytes,
    load_image_from_path,
    load_image_from_url,
)

logger = logging.getLogger(__name__)


def _load_clip():
    """Lazy-load the CLIP processor and model."""
    from transformers import CLIPModel, CLIPProcessor

    try:
        logger.info("Loading CLIP model: %s", CLIP_MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        model.eval()
        return processor, model
    except Exception as exc:
        logger.exception("Failed to load CLIP model '%s'", CLIP_MODEL_NAME)
        raise RuntimeError(f"Failed to load CLIP model '{CLIP_MODEL_NAME}'") from exc


def _encode_image(image: Image.Image, processor, model) -> np.ndarray:
    """
    Encode a single PIL image with CLIP.
    Returns a normalised 512-dim numpy vector.
    """
    import torch

    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    vec = features.squeeze().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            f"Invalid CLIP embedding norm ({norm}) for image. Expected a positive finite value."
        )
    normalised = vec / norm
    if not np.all(np.isfinite(normalised)):
        raise ValueError(
            "CLIP embedding contains non-finite values (NaN/Inf). Check input image integrity."
        )
    return normalised


class ImageRetrieval:
    """CLIP + FAISS image similarity search."""

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._embeddings: np.ndarray | None = None  # (N, D)
        self._ids: list[str] = []
        self._faiss_index = None
        self._df: pd.DataFrame | None = None
        self._init_error: str | None = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _normalise_embeddings(self) -> None:
        """L2-normalise embedding matrix in-place with zero-safe handling."""
        if self._embeddings is None or self._embeddings.size == 0:
            return
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        zero_norms = int(np.sum(norms == 0))
        if zero_norms:
            logger.warning("Found %d zero-norm embeddings during normalisation.", zero_norms)
        norms[norms == 0] = 1.0
        self._embeddings = (self._embeddings / norms).astype(np.float32)

    # Replace lines 107-160 with this:

    def _build_embeddings(self, df: pd.DataFrame) -> None:
        """
        Load pre-generated embeddings from Colab or build them if needed.
        
        IMPORTANT: Pre-generated embeddings were created with OpenAI CLIP 
        on Google Colab GPU and are stored in backend/data/embeddings/
        """
        if df.empty:
            logger.warning("Dataset is empty; cannot build image embeddings.")
            self._embeddings = np.empty((0, 512), dtype=np.float32)
            self._ids = []
            return
        
        # Try to load pre-generated embeddings first
        embeddings_path = Path(EMBEDDINGS_FILE)
        ids_path = Path(EMBEDDINGS_IDS_FILE)
        
        if embeddings_path.exists() and ids_path.exists():
            try:
                self._embeddings = np.load(embeddings_path).astype(np.float32)
                self._ids = [str(i) for i in np.load(ids_path, allow_pickle=True)]
                self._normalise_embeddings()
                logger.info(
                    "✅ Loaded pre-generated embeddings from Colab: %d vectors",
                    len(self._ids),
                )
                return
            except Exception as exc:
                logger.warning("Could not load pre-generated embeddings: %s. Building from scratch...", exc)
        
        # If pre-generated embeddings not found, fall back to CLIP encoding
        logger.info("⚙️ Building embeddings from scratch using CLIP...")
        self._processor, self._model = _load_clip()

        rows = df if MAX_IMAGE_EMBED <= 0 else df.head(MAX_IMAGE_EMBED)
        vectors, ids = [], []
        local_missing_attempts = 0
        remote_unavailable_or_failed = 0
        encode_errors = 0

        for _, row in rows.iterrows():
            local_path = str(IMAGES_DIR / f"{row['id']}.png")
            img = load_image_from_path(local_path)
            if img is None:
                local_missing_attempts += 1
                img_url = row.get("img", "")
                if img_url:
                    img = load_image_from_url(img_url)
                else:
                    remote_unavailable_or_failed += 1
            if img is None:
                if row.get("img", ""):
                    remote_unavailable_or_failed += 1
                continue
            try:
                vec = _encode_image(img, self._processor, self._model)
                vectors.append(vec)
                ids.append(str(row["id"]))
            except Exception as exc:
                encode_errors += 1
                logger.debug("Skipping product %s: %s", row["id"], exc)

        if not vectors:
            logger.warning("No image embeddings were generated.")
            self._embeddings = np.empty((0, 512), dtype=np.float32)
            self._ids = []
            return

        self._embeddings = np.stack(vectors, axis=0).astype(np.float32)
        self._normalise_embeddings()
        self._ids = ids
        logger.info(
            "Built %d image embeddings (rows=%d, local_misses=%d, remote_misses=%d, encode_errors=%d).",
            len(ids),
            len(rows),
            local_missing_attempts,
            remote_unavailable_or_failed,
            encode_errors,
        )

    def _save_embeddings(self) -> None:
        try:
            Path(EMBEDDINGS_FILE).parent.mkdir(parents=True, exist_ok=True)
            np.save(EMBEDDINGS_FILE, self._embeddings)
            np.save(EMBEDDINGS_IDS_FILE, np.array(self._ids))
            logger.info("Image embeddings saved.")
        except Exception as exc:
            logger.warning("Could not save embeddings: %s", exc)

    def _load_embeddings(self) -> bool:
        try:
            if not (
                Path(EMBEDDINGS_FILE).exists()
                and Path(EMBEDDINGS_IDS_FILE).exists()
            ):
                return False
            embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
            ids = list(np.load(EMBEDDINGS_IDS_FILE, allow_pickle=True))
            if embeddings.ndim != 2:
                raise ValueError(
                    f"Invalid embeddings shape {embeddings.shape}; expected 2-D matrix."
                )
            if embeddings.shape[0] != len(ids):
                raise ValueError(
                    f"Embeddings count {embeddings.shape[0]} does not match id count {len(ids)}."
                )
            self._embeddings = embeddings
            self._normalise_embeddings()
            self._ids = [str(i) for i in ids]
            logger.info(
                "Image embeddings loaded from cache: %d vectors.", len(self._ids)
            )
            return True
        except Exception as exc:
            logger.warning("Could not load embeddings cache: %s", exc)
            return False

    def _build_faiss_index(self) -> None:
        """Build an in-memory FAISS index (inner-product on normalised vecs)."""
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            return
        try:
            import faiss

            dim = self._embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)  # inner-product = cosine for L2-normed vecs
            index.add(self._embeddings)
            self._faiss_index = index
            logger.info("FAISS index built with %d vectors.", index.ntotal)
        except ImportError:
            logger.info("FAISS not available, will use numpy cosine similarity.")

    def initialize(self) -> None:
        """Prepare CLIP model and embeddings index."""
        self._init_error = None
        try:
            self._df = load_data()
        except Exception as exc:
            self._df = pd.DataFrame()
            self._init_error = f"Dataset load failed: {exc}"
            logger.exception("ImageRetrieval initialization failed during dataset load.")
            return

        try:
            if not self._load_embeddings():
                self._build_embeddings(self._df)
                self._save_embeddings()
            # Load CLIP for query encoding if not already loaded
            if self._processor is None:
                self._processor, self._model = _load_clip()
            self._build_faiss_index()
            if self._embeddings is None or self._embeddings.shape[0] == 0:
                self._init_error = "No image embeddings available. Populate backend/data/images first."
            logger.info(
                "ImageRetrieval initialized (products=%d, embeddings=%d, faiss=%s).",
                len(self._df),
                len(self._ids),
                self._faiss_index is not None,
            )
        except Exception as exc:
            self._init_error = str(exc)
            logger.exception("ImageRetrieval initialization failed: %s", exc)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _numpy_search(
        self, query_vec: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback cosine similarity via numpy."""
        sims = (self._embeddings @ query_vec).flatten()
        indices = np.argsort(sims)[::-1][:top_k]
        return sims[indices], indices

    def search(
        self,
        image: Image.Image,
        top_k: int = DEFAULT_TOP_K,
        brand_filter: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
    ) -> list[dict]:
        """
        Return top-K products visually similar to the query image.

        Args:
            image:        Query PIL Image.
            top_k:        Number of results to return.
            brand_filter: Optional brand filter.
            min_price:    Optional minimum price filter.
            max_price:    Optional maximum price filter.

        Returns:
            List of product dicts with a "score" key.
        """
        if self._processor is None:
            self.initialize()

        if self._init_error:
            logger.warning("Image retrieval not ready: %s", self._init_error)
            return []

        if self._embeddings is None or self._embeddings.shape[0] == 0:
            logger.warning("No embeddings available for image search.")
            return []

        query_vec = _encode_image(image, self._processor, self._model)

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(
                query_vec.reshape(1, -1), min(top_k * 4, len(self._ids))
            )
            scores = scores.flatten()
            indices = indices.flatten()
        else:
            scores, indices = self._numpy_search(query_vec, top_k * 4)

        # Map back to product rows
        df = self._df
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._ids):
                continue
            pid = self._ids[idx]
            rows = df[df["id"] == pid]
            if rows.empty:
                continue
            row = rows.iloc[0]

            # Apply optional filters
            if brand_filter and brand_filter.lower() not in str(
                row.get("brand", "")
            ).lower():
                continue
            if min_price is not None and row.get("sold_price_num", 0) < min_price:
                continue
            if max_price is not None and row.get("sold_price_num", 0) > max_price:
                continue

            product = product_row_to_dict(row)
            product["score"] = round(float(score), 4)
            results.append(product)
            if len(results) >= top_k:
                break

        return results

    def search_from_bytes(
        self,
        image_bytes: bytes,
        top_k: int = DEFAULT_TOP_K,
        **kwargs,
    ) -> list[dict]:
        """Convenience wrapper: search from raw image bytes."""
        img = load_image_from_bytes(image_bytes)
        if img is None:
            return []
        return self.search(img, top_k=top_k, **kwargs)

    def status(self) -> dict:
        """Return lightweight diagnostics for health checks and startup logs."""
        return {
            "ready": self.is_ready(),
            "products_count": len(self._df) if self._df is not None else 0,
            "embeddings_count": len(self._ids),
            "clip_loaded": self._processor is not None and self._model is not None,
            "faiss_enabled": self._faiss_index is not None,
            "init_error": self._init_error,
        }

    def is_ready(self) -> bool:
        return (
            self._init_error is None
            and self._embeddings is not None
            and self._embeddings.shape[0] > 0
            and self._processor is not None
            and self._model is not None
        )
