"""
TF-IDF based text retrieval model.

Features
--------
* Builds a TF-IDF matrix over "title + brand" text.
* Supports optional query expansion via preprocessing.expand_query().
* Caches the fitted vectorizer and matrix to disk for fast restarts.
* Supports brand and price-range filtering.
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    DEFAULT_TOP_K,
    TFIDF_CACHE_FILE,
    TFIDF_VECTORIZER_FILE,
)
from utils.data_loader import load_data, product_row_to_dict
from utils.preprocessing import clean_text, expand_query

logger = logging.getLogger(__name__)


class TextRetrieval:
    """TF-IDF retrieval over product titles and brands."""

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix: sp.csr_matrix | None = None
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _build_index(self, df: pd.DataFrame) -> None:
        """Fit TF-IDF vectorizer on the product corpus."""
        logger.info("Building TF-IDF index for %d products …", len(df))
        corpus = df["search_text"].tolist()
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        logger.info("TF-IDF index built: shape %s", self._tfidf_matrix.shape)

    def _save_index(self) -> None:
        """Persist the vectorizer and matrix to disk."""
        try:
            Path(TFIDF_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
            sp.save_npz(TFIDF_CACHE_FILE, self._tfidf_matrix)
            with open(TFIDF_VECTORIZER_FILE, "wb") as f:
                pickle.dump(self._vectorizer, f)
            logger.info("TF-IDF index saved to disk.")
        except Exception as exc:
            logger.warning("Could not save TF-IDF index: %s", exc)

    def _load_index(self) -> bool:
        """Try to load a cached index. Returns True on success."""
        try:
            if not (
                Path(TFIDF_CACHE_FILE).exists()
                and Path(TFIDF_VECTORIZER_FILE).exists()
            ):
                return False
            self._tfidf_matrix = sp.load_npz(TFIDF_CACHE_FILE)
            with open(TFIDF_VECTORIZER_FILE, "rb") as f:
                self._vectorizer = pickle.load(f)
            logger.info("TF-IDF index loaded from cache.")
            return True
        except Exception as exc:
            logger.warning("Could not load TF-IDF cache: %s", exc)
            return False

    def initialize(self) -> None:
        """Load data and prepare the TF-IDF index."""
        self._df = load_data()
        if not self._load_index():
            self._build_index(self._df)
            self._save_index()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        expand: bool = True,
        brand_filter: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
    ) -> list[dict]:
        """
        Return top-K products matching the text query.

        Args:
            query:        Free-text search query.
            top_k:        Number of results to return.
            expand:       Whether to apply query expansion.
            brand_filter: Optional brand name filter (case-insensitive).
            min_price:    Minimum sold price filter.
            max_price:    Maximum sold price filter.

        Returns:
            List of product dicts, each with an additional "score" key.
        """
        if self._vectorizer is None or self._tfidf_matrix is None:
            self.initialize()

        df = self._df

        # Apply filters
        mask = pd.Series([True] * len(df), index=df.index)
        if brand_filter:
            mask &= df["brand"].str.lower().str.contains(
                brand_filter.lower(), na=False
            )
        if min_price is not None:
            mask &= df["sold_price_num"] >= min_price
        if max_price is not None:
            mask &= df["sold_price_num"] <= max_price

        filtered_df = df[mask]
        if filtered_df.empty:
            return []

        # Optionally expand query
        effective_query = expand_query(query) if expand else query
        effective_query = clean_text(effective_query)

        # Encode query
        query_vec = self._vectorizer.transform([effective_query])

        # Compute similarity only against filtered rows
        filtered_indices = filtered_df.index.tolist()
        filtered_matrix = self._tfidf_matrix[filtered_indices]
        sims = cosine_similarity(query_vec, filtered_matrix).flatten()

        # Get top-K indices (within filtered set)
        top_local_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for local_idx in top_local_indices:
            score = float(sims[local_idx])
            if score < 1e-6:
                continue
            row = filtered_df.iloc[local_idx]
            product = product_row_to_dict(row)
            product["score"] = round(score, 4)
            results.append(product)

        return results

    # ------------------------------------------------------------------
    # Relevance feedback (Rocchio)
    # ------------------------------------------------------------------

    def rocchio_update(
        self,
        query: str,
        positive_ids: list[str],
        negative_ids: list[str],
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.15,
    ) -> list[dict]:
        """
        Apply Rocchio relevance feedback and return refined results.

        Args:
            query:        Original query string.
            positive_ids: IDs of products marked as relevant.
            negative_ids: IDs of products marked as not relevant.
            alpha:        Weight for original query.
            beta:         Weight for positive centroid.
            gamma:        Weight for negative centroid.

        Returns:
            Refined top-K results.
        """
        if self._vectorizer is None or self._tfidf_matrix is None:
            self.initialize()

        df = self._df
        q_vec = self._vectorizer.transform([clean_text(query)]).toarray()

        def _centroid(ids: list[str]) -> np.ndarray:
            indices = df.index[df["id"].isin(ids)].tolist()
            if not indices:
                return np.zeros_like(q_vec)
            vecs = self._tfidf_matrix[indices].toarray()
            return vecs.mean(axis=0, keepdims=True)

        pos_centroid = _centroid(positive_ids)
        neg_centroid = _centroid(negative_ids)

        updated_q = (
            alpha * q_vec
            + beta * pos_centroid
            - gamma * neg_centroid
        )

        sims = cosine_similarity(updated_q, self._tfidf_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:DEFAULT_TOP_K]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score < 1e-6:
                continue
            row = df.iloc[idx]
            product = product_row_to_dict(row)
            product["score"] = round(score, 4)
            results.append(product)

        return results
