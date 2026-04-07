# ecommerce-multimodal

A full-stack **Multimodal Information Retrieval System** for fashion e-commerce that lets users search 65 000+ Flipkart products using free-text queries *and* image uploads.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Backend API](#backend-api)
5. [Frontend](#frontend)
6. [Configuration](#configuration)
7. [Evaluation Metrics](#evaluation-metrics)
8. [cURL Examples](#curl-examples)
9. [Architecture Notes](#architecture-notes)

---

## Features

| Capability | Detail |
|---|---|
| **Text Search** | TF-IDF (1-2 gram) + cosine similarity over title + brand |
| **Image Search** | OpenAI CLIP (ViT-B/32) + FAISS flat index |
| **Hybrid Search** | Weighted fusion of text + image scores |
| **Query Expansion** | Domain synonyms + WordNet synsets |
| **Relevance Feedback** | Rocchio algorithm |
| **Filters** | Brand, min/max price |
| **Evaluation** | Precision@K, Recall@K, NDCG@K, MAP |
| **Caching** | TF-IDF matrix & CLIP embeddings serialised to disk |

---

## Project Structure

```
.
├── backend/
│   ├── app.py                  # Flask entry point
│   ├── config.py               # All configuration constants
│   ├── requirements.txt
│   ├── models/
│   │   ├── text_retrieval.py   # TF-IDF retrieval + Rocchio
│   │   ├── image_retrieval.py  # CLIP + FAISS retrieval
│   │   └── hybrid_retrieval.py # Score fusion
│   ├── utils/
│   │   ├── data_loader.py      # CSV loading + product serialisation
│   │   ├── preprocessing.py    # Text cleaning, query expansion, image loading
│   │   └── evaluation.py       # IR metrics
│   ├── routes/
│   │   ├── search.py           # /search/* endpoints
│   │   └── health.py           # /health, /
│   └── data/
│       ├── images/             # Product images (<id>.png)
│       ├── products.csv        # Optional local CSV copy
│       └── embeddings/         # Auto-generated cache (git-ignored)
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── Data - Copy.csv             # Source dataset (repo root)
└── README.md
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- Any static file server (for the frontend)
- *(Optional)* GPU with CUDA for faster CLIP encoding

### 1 — Backend

```bash
# Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Start the server (development)
cd backend
python app.py
```

The API will be available at `http://localhost:5000`.

On first run the server will:
1. Build the TF-IDF index (~5 s for 62 k products).
2. Download the CLIP model weights from Hugging Face (~340 MB, cached in `~/.cache`).
3. Encode any product images found in `backend/data/images/`.

### 2 — Frontend

Open `frontend/index.html` directly in a browser, **or** serve it:

```bash
cd frontend
python -m http.server 8080
# Then open http://localhost:8080
```

> **CORS**: The backend allows all origins by default.  
> Set `CORS_ORIGINS=http://localhost:8080` for stricter security.

### 3 — Dataset Images (optional)

```python
import kagglehub, shutil, pathlib

path = kagglehub.dataset_download("kuchhbhi/flipkart-fashion-products-65k-dataset")
src  = pathlib.Path(path) / "images"
dst  = pathlib.Path("backend/data/images")
dst.mkdir(parents=True, exist_ok=True)
for img in src.glob("*.png"):
    shutil.copy(img, dst / img.name)
```

Restart the backend after copying images. Embeddings are cached after the first run.

---

## Backend API

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /search/text`

```json
{
  "query": "silk saree",
  "top_k": 12,
  "expand": true,
  "brand_filter": "Vaidehi",
  "min_price": 500,
  "max_price": 5000
}
```

### `POST /search/image`

Multipart `image` file. Optional query params: `top_k`, `brand`, `min_price`, `max_price`.

### `POST /search/hybrid`

Multipart: `query` (text), `image` (file), `alpha` (0–1).

### `POST /search/feedback`

```json
{ "query": "blue kurta", "positive_ids": ["1","2"], "negative_ids": ["3"] }
```

### `POST /search/evaluate`

```json
{ "query": "red dress", "relevant_ids": ["1","5","12"], "k_values": [1,5,10] }
```

### `GET /api/images/<id>`

Serves local product image (404 if not found).

---

## Frontend

Four-tab single-page UI:

| Tab | Description |
|---|---|
| 🔍 Text Search | Query with optional filters |
| 🖼️ Image Search | Drag-and-drop or browse |
| ✨ Hybrid Search | Text + image combined |
| 📊 Evaluate | Interactive IR metric computation |

Set `window.API_BASE` before loading `script.js` to override the default `http://localhost:5000`.

---

## Configuration

`backend/config.py` — override via environment variables:

| Variable | Default | Description |
|---|---|---|
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch32` | HuggingFace CLIP model |
| `FLASK_HOST` | `0.0.0.0` | Bind host |
| `FLASK_PORT` | `5000` | Bind port |
| `FLASK_DEBUG` | `false` | Debug mode |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `MAX_IMAGE_EMBED` | `0` (all) | Limit images to embed |

---

## Evaluation Metrics

`backend/utils/evaluation.py`:

- **Precision@K** — fraction of top-K that are relevant  
- **Recall@K** — fraction of relevant items found in top-K  
- **NDCG@K** — ranking-aware metric  
- **Average Precision** — area under precision-recall curve  
- **MAP** — mean AP over multiple queries  

---

## cURL Examples

```bash
# Text search
curl -s -X POST http://localhost:5000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query":"red silk saree","top_k":5}' | python -m json.tool

# Image search
curl -s -X POST "http://localhost:5000/search/image?top_k=6" \
  -F "image=@/path/to/query.jpg" | python -m json.tool

# Hybrid search
curl -s -X POST http://localhost:5000/search/hybrid \
  -F "query=blue dress" -F "image=@/path/to/query.jpg" -F "alpha=0.6" | python -m json.tool

# Relevance feedback
curl -s -X POST http://localhost:5000/search/feedback \
  -H "Content-Type: application/json" \
  -d '{"query":"saree","positive_ids":["1","2"],"negative_ids":["3"]}' | python -m json.tool

# Evaluate
curl -s -X POST http://localhost:5000/search/evaluate \
  -H "Content-Type: application/json" \
  -d '{"query":"saree","relevant_ids":["1","2","3"],"k_values":[1,5,10]}' | python -m json.tool

# Health check
curl http://localhost:5000/health
```

---

## Architecture Notes

- **TF-IDF index** is built on first run and serialised as `.npz` + `.pkl` under `backend/data/embeddings/`.
- **CLIP embeddings** are stored as `.npy` arrays alongside an IDs array for FAISS lookup.
- Without local images, image search returns an empty list (CLIP loads but has nothing to index).
- The frontend falls back to the remote Flipkart CDN image URL when no local image exists.
