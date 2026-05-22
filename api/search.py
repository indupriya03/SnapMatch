"""
api/search.py
-------------
FastAPI backend for the deep-visual-retrieval project.

Endpoints:
  GET  /health      → system status, index info, DB stats
  GET  /categories  → all product categories with image counts
  GET  /image       → serve a product image by file path
  POST /search      → upload image → Top-K similar products

Run with:
  uvicorn api.search:app --reload --host 0.0.0.0 --port 8000
"""

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session

from src.searcher import Searcher
from db.database import get_db, test_connection, get_table_counts
from db import queries

BASE_DIR        = Path(__file__).resolve().parent.parent
EVAL_REPORT_RELAXED = BASE_DIR / "outputs" / "evaluation_report_relaxed.json"
DEFAULT_TOP_K   = 8
MAX_TOP_K       = 20
MAX_IMAGE_SIZE  = 10 * 1024 * 1024  # 10 MB

# ── Global searcher — loaded once at startup ───────────────────────────────────
_searcher: Searcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _searcher
    print("\n[STARTUP] Loading resources ...")
    test_connection()
    _searcher = Searcher()
    print("[STARTUP] Ready.\n")
    yield
    print("[SHUTDOWN] Shutting down.")


app = FastAPI(
    title="Deep Visual Retrieval API",
    description="Visual product similarity search — ResNet50 + FAISS + MySQL",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health(db: Session = Depends(get_db)):
    db_counts = get_table_counts()
    return JSONResponse({
        "status":        "ok",
        "model":         "ResNet50 (ImageNet, fc=Identity)",
        "index_type":    "IndexFlatIP (cosine similarity)",
        "total_vectors": _searcher.total_vectors,
        "total_images":  _searcher.total_images,
        "database":      db_counts,
    })


@app.get("/categories")
async def get_categories(db: Session = Depends(get_db)):
    categories   = queries.get_all_super_categories(db)
    image_counts = queries.get_category_image_counts(db)
    return JSONResponse({
        "categories":       categories,
        "image_counts":     image_counts,
        "total_categories": len(categories),
    })


@app.get("/categories/{category_name}/samples")
async def get_category_samples(
    category_name: str,
    limit: int = Query(default=8, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """
    Return sample products for a given super category name.
    Used by Browse by Category page in Streamlit.
    """
    products = queries.get_products_by_category(db, category_name, limit=limit)
    if not products:
        raise HTTPException(status_code=404, detail=f"Category '{category_name}' not found or empty.")
    return JSONResponse({
        "category":  category_name,
        "count":     len(products),
        "products":  products,
    })


@app.get("/metrics")
async def get_metrics():
    """Return relaxed evaluation metrics from outputs/evaluation_report_relaxed.json."""
    if not EVAL_REPORT_RELAXED.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation report not found. Run evaluate.py first."
        )
    with open(EVAL_REPORT_RELAXED) as f:
        return JSONResponse(json.load(f))


@app.get("/image")
async def serve_image(
    path: str = Query(..., description="Relative image file path")
):
    img_path = Path(path)
    try:
        resolved = img_path.resolve()
        data_dir = (BASE_DIR / "data").resolve()
        resolved.relative_to(data_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Path not allowed.")
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")
    suffix     = img_path.suffix.lower()
    media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    return FileResponse(str(img_path), media_type=media_type)


@app.post("/search")
async def search_similar(
    file: UploadFile = File(...),
    top_k: int = Query(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
    category_filter: str = Query(default=None),
    db: Session = Depends(get_db),
):
    # Validate
    allowed = {"image/jpeg", "image/jpg", "image/png"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Upload JPEG or PNG only.")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")

    # Search
    fetch_k = MAX_TOP_K if category_filter else top_k
    result  = _searcher.search(image_bytes, top_k=top_k, fetch_k=fetch_k)

    filenames      = result["filenames"]
    score_map      = result["score_map"]
    search_time_ms = result["search_time_ms"]

    # Enrich with MySQL metadata
    metadata = queries.get_products_by_filenames(db, filenames)

    results = []
    rank    = 1
    for fname in filenames:
        meta = metadata.get(fname, {})
        if category_filter and meta.get("category") != category_filter:
            continue
        results.append({
            "rank":             rank,
            "similarity_score": score_map[fname],
            "image_url":        f"/image?path={meta.get('image_path', '')}",
            "filename":         fname,
            "category":         meta.get("category",     "unknown"),
            "display_name":     meta.get("display_name", "Unknown"),
            "class_id":         meta.get("class_id"),
            "super_class_id":   meta.get("super_class_id"),
            "split":            meta.get("split", "unknown"),
        })
        rank += 1
        if len(results) >= top_k:
            break

    return JSONResponse({
        "query_filename":  file.filename,
        "top_k_requested": top_k,
        "total_returned":  len(results),
        "search_time_ms":  search_time_ms,
        "category_filter": category_filter,
        "results":         results,
    })