"""
Data loading utilities.

Reads the Flipkart fashion CSV and exposes a clean DataFrame used by all
retrieval models.
"""
import logging
import re
from pathlib import Path

import pandas as pd

from config import CSV_PATH, IMAGES_DIR

logger = logging.getLogger(__name__)

_df: pd.DataFrame | None = None


def _clean_price(value: str) -> float:
    """Convert a price string like '₹1,299' to a float."""
    if pd.isna(value):
        return 0.0
    cleaned = re.sub(r"[^\d.]", "", str(value))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def load_data() -> pd.DataFrame:
    """
    Load and cache the product DataFrame.

    Columns returned:
        id, brand, title, sold_price, actual_price, url, img,
        sold_price_num, actual_price_num, search_text, local_image
    """
    global _df
    if _df is not None:
        return _df

    logger.info("Loading dataset from %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH, dtype={"id": str})

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Drop rows without id or title
    df = df.dropna(subset=["id", "title"]).reset_index(drop=True)

    # Clean id – ensure string without decimals
    df["id"] = df["id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    # Numeric prices
    df["sold_price_num"] = df["sold_price"].apply(_clean_price)
    df["actual_price_num"] = df["actual_price"].apply(_clean_price)

    # Combined text for TF-IDF
    df["brand"] = df["brand"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["search_text"] = (df["title"] + " " + df["brand"]).str.strip()

    # Local image path (may not exist during development)
    df["local_image"] = df["id"].apply(
        lambda pid: str(IMAGES_DIR / f"{pid}.png")
    )

    logger.info("Dataset loaded: %d products", len(df))
    _df = df
    return df


def get_product_by_id(product_id: str) -> dict | None:
    """Return a single product record as a dict, or None if not found."""
    df = load_data()
    rows = df[df["id"] == str(product_id)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def product_row_to_dict(row: pd.Series) -> dict:
    """Serialise a DataFrame row to a JSON-safe dict."""
    local_img = row.get("local_image", "")
    img_url = row.get("img", "")

    # Use remote image URL from CSV (local images don't exist)
    # The img column in the CSV has the product image URLs
    display_image = img_url if img_url else f"/api/images/{row['id']}"

    return {
        "id": str(row["id"]),
        "title": str(row.get("title", "")),
        "brand": str(row.get("brand", "")),
        "sold_price": str(row.get("sold_price", "")),
        "actual_price": str(row.get("actual_price", "")),
        "sold_price_num": float(row.get("sold_price_num", 0)),
        "actual_price_num": float(row.get("actual_price_num", 0)),
        "url": str(row.get("url", "")),
        "image": display_image,
        "img_url": str(img_url),
    }
