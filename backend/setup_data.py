"""Dataset setup and image-embedding bootstrap utility.

Usage:
    python setup_data.py
    python setup_data.py --skip-download
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

from config import DATA_DIR, IMAGES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("setup_data")

DEFAULT_DATASET = "kuchhbhi/flipkart-fashion-products-65k-dataset"
REQUIRED_COLUMNS = {"id", "brand", "title", "sold_price", "actual_price", "url", "img"}


def _find_first_csv(directory: Path) -> Path | None:
    csv_files = sorted(directory.rglob("*.csv"))
    return csv_files[0] if csv_files else None


def _find_images_dir(directory: Path) -> Path | None:
    candidates = sorted([p for p in directory.rglob("*") if p.is_dir() and p.name.lower() == "images"])
    if candidates:
        return candidates[0]
    return None


def _copy_images(source_images: Path, destination_images: Path) -> int:
    destination_images.mkdir(parents=True, exist_ok=True)
    image_paths = [
        p for p in source_images.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    copied = 0
    for idx, path in enumerate(image_paths, start=1):
        target = destination_images / path.name
        if not target.exists():
            shutil.copy2(path, target)
            copied += 1
        if idx % 5000 == 0:
            logger.info("Image copy progress: %d/%d", idx, len(image_paths))
    logger.info("Image sync complete. Total source=%d, newly copied=%d", len(image_paths), copied)
    return len(image_paths)


def _validate_csv(csv_path: Path) -> tuple[bool, int, set[str]]:
    if not csv_path.exists():
        return False, 0, set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        rows = sum(1 for _ in reader)
    return REQUIRED_COLUMNS.issubset(headers), rows, headers


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/prepare dataset and build image embeddings cache.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset slug (owner/name)")
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle download and use local files only")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    source_root: Path | None = None
    if not args.skip_download:
        try:
            import kagglehub
            logger.info("Downloading dataset from Kaggle: %s", args.dataset)
            source_root = Path(kagglehub.dataset_download(args.dataset))
            logger.info("Dataset downloaded to: %s", source_root)
        except ModuleNotFoundError:
            logger.warning("kagglehub is not installed. Run: pip install -r requirements.txt")
        except Exception as exc:
            logger.warning("Dataset download failed: %s", exc)
            logger.warning(
                "If this is an auth issue, verify Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)."
            )
            logger.warning("Proceeding with local files only. Ensure CSV/images already exist.")

    target_csv = DATA_DIR / "products.csv"
    if source_root is not None:
        source_csv = _find_first_csv(source_root)
        source_images = _find_images_dir(source_root)

        if source_csv:
            target_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_csv, target_csv)
            logger.info("Copied CSV: %s -> %s", source_csv, target_csv)
        else:
            logger.warning("No CSV found in downloaded dataset: %s", source_root)

        if source_images:
            _copy_images(source_images, IMAGES_DIR)
        else:
            logger.warning("No images directory found in downloaded dataset: %s", source_root)

    csv_ok, row_count, headers = _validate_csv(target_csv)
    local_images_count = len([
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])

    logger.info("Validation | csv_exists=%s path=%s", target_csv.exists(), target_csv)
    logger.info("Validation | csv_rows=%d required_columns_present=%s", row_count, csv_ok)
    logger.info("Validation | local_images=%d dir=%s", local_images_count, IMAGES_DIR)

    if not csv_ok:
        missing = sorted(REQUIRED_COLUMNS - headers)
        logger.error("CSV is invalid or missing required columns: %s", missing)
        return 1
    if local_images_count == 0:
        logger.error("No local images found. Image search will return 0 results.")
        return 1

    from models.image_retrieval import ImageRetrieval
    retriever = ImageRetrieval()
    retriever.initialize()
    status = retriever.status()
    logger.info("Image retrieval status after setup: %s", status)

    if not status["ready"]:
        logger.error("Image retrieval is not ready: %s", status["init_error"])
        return 1

    logger.info("Setup completed successfully. Image search is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
