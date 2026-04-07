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

    logger.info("Loading CLIP model: %s", CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    return processor, model


def _encode_image(image: Image.Image, processor, model) -> np.ndarray:
    """
    Encode a single PIL image with CLIP.
    Returns a normalised 512-dim numpy vector.
    """
    import torch

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    vec = features.squeeze().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class ImageRetrieval:
    """CLIP + FAISS image similarity search."""

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._embeddings: np.ndarray | None = None  # (N, D)
        self._ids: list[str] = []
        self._faiss_index = None
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_embeddings(self, df: pd.DataFrame) -> None:
        """Encode product images and store embeddings."""
        self._processor, self._model = _load_clip()

        rows = df if MAX_IMAGE_EMBED <= 0 else df.head(MAX_IMAGE_EMBED)
        vectors, ids = [], []

        for _, row in rows.iterrows():
            local_path = str(IMAGES_DIR / f"{row['id']}.png")
            img = load_image_from_path(local_path)
            if img is None:
                img_url = row.get("img", "")
                if img_url:
                    img = load_image_from_url(img_url)
            if img is None:
                continue
            try:
                vec = _encode_image(img, self._processor, self._model)
                vectors.append(vec)
                ids.append(str(row["id"]))
            except Exception as exc:
                logger.debug("Skipping product %s: %s", row["id"], exc)

        if not vectors:
            logger.warning("No image embeddings were generated.")
            self._embeddings = np.empty((0, 512), dtype=np.float32)
            self._ids = []
            return

        self._embeddings = np.stack(vectors, axis=0).astype(np.float32)
        self._ids = ids
        logger.info("Built %d image embeddings.", len(ids))

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
            self._embeddings = np.load(EMBEDDINGS_FILE)
            self._ids = list(np.load(EMBEDDINGS_IDS_FILE, allow_pickle=True))
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
        self._df = load_data()
        if not self._load_embeddings():
            self._build_embeddings(self._df)
            self._save_embeddings()
        # Load CLIP for query encoding if not already loaded
        if self._processor is None:
            self._processor, self._model = _load_clip()
        self._build_faiss_index()

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
