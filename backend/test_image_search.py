"""Manual CLI for debugging image search.

Usage:
    python test_image_search.py --image /path/to/query.png --top-k 5
    python test_image_search.py  # auto-picks first local image
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import IMAGES_DIR


def _pick_sample_image() -> Path | None:
    if not IMAGES_DIR.exists():
        return None
    candidates = sorted([
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual CLIP image-search smoke test")
    parser.add_argument("--image", help="Path to query image")
    parser.add_argument("--top-k", type=int, default=5, help="Results to return")
    args = parser.parse_args()

    try:
        from models.image_retrieval import ImageRetrieval, _encode_image
        from utils.preprocessing import load_image_from_path
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc}. Install backend requirements first.")
        return 1

    image_path = Path(args.image) if args.image else _pick_sample_image()
    if image_path is None or not image_path.exists():
        print("No query image available. Pass --image or populate backend/data/images.")
        return 1

    img = load_image_from_path(str(image_path))
    if img is None:
        print(f"Failed to read image: {image_path}")
        return 1

    model = ImageRetrieval()
    model.initialize()
    status = model.status()
    print("Image model status:", status)
    if not status["clip_loaded"]:
        print("CLIP model did not load correctly.")
        return 1

    vec = _encode_image(img, model._processor, model._model)
    print(f"CLIP encoding OK | dim={vec.shape[0]} | norm={float((vec**2).sum()**0.5):.4f}")

    results = model.search(img, top_k=max(1, args.top_k))
    print(f"Search results count: {len(results)}")
    for idx, item in enumerate(results, start=1):
        print(f"{idx:>2}. id={item['id']} score={item.get('score')} title={item.get('title', '')[:80]}")

    if not results:
        print("No results returned. Run `python setup_data.py` and check /health diagnostics.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
