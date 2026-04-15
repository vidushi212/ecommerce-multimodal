# Backend Image Search Setup & Troubleshooting

## 1) Install dependencies

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --prefer-binary
```

## 2) Prepare dataset + embeddings

Run once before first image search:

```bash
cd backend
python setup_data.py
```

What this does:
- downloads Kaggle dataset via `kagglehub`
- copies CSV to `backend/data/products.csv`
- copies images to `backend/data/images/`
- validates required CSV columns and image count
- builds and caches CLIP embeddings

If Kaggle download is already done, use:

```bash
python setup_data.py --skip-download
```

## 3) Start backend

```bash
cd backend
python app.py
```

Startup logs now report:
- dataset CSV availability
- local image count
- embedding cache status
- image model readiness and any init error

## 4) Verify readiness

Health endpoint:

```bash
curl -s http://localhost:5000/health | python -m json.tool
```

Key fields:
- `status` should be `ok`
- `checks.local_images_count` should be > 0
- `checks.embeddings_cache_exists` should be true
- `image_retrieval.ready` should be true

## 5) Manual CLI image-search test

```bash
cd backend
python test_image_search.py --top-k 5
# or provide explicit file
python test_image_search.py --image /absolute/path/to/image.png --top-k 5
```

Expected:
- CLIP encoding succeeds
- non-empty search results

## Common reasons for "Image Search Results (0)"

1. **No local images**
   - `backend/data/images/` is empty
   - fix: run `python setup_data.py`

2. **Embeddings not built / stale cache**
   - missing `backend/data/embeddings/image_embeddings.npy`
   - fix: run `python setup_data.py`

3. **CSV missing or invalid**
   - missing required columns: `id, brand, title, sold_price, actual_price, url, img`
   - fix: re-run setup and verify CSV

4. **CLIP model load failure**
   - often due to network or model download issue
   - fix: retry and check startup logs / `/health` `image_retrieval.init_error`

5. **Bad image upload**
   - request must be multipart form-data with field name `image`
   - backend rejects empty file payloads

## API request format (image search)

```bash
curl -s -X POST "http://localhost:5000/search/image?top_k=6" \
  -F "image=@/absolute/path/to/query.jpg" | python -m json.tool
```

The form field **must** be named `image`.
