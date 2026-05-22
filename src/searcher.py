"""
src/searcher.py
---------------
Reusable FAISS search logic — shared by both api/search.py (FastAPI)
and app.py (Streamlit). Load once, call many times.

Usage:
    from src.searcher import Searcher
    searcher = Searcher()
    results = searcher.search(image_bytes, top_k=8)
"""

import io
import json
import time
import numpy as np
import faiss
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

BASE_DIR         = Path(__file__).resolve().parent.parent
INDEX_PATH       = BASE_DIR / "outputs" / "faiss_index.bin"
IMAGE_PATHS_PATH = BASE_DIR / "outputs" / "image_paths.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
EMBEDDING_DIM = 2048


class Searcher:
    """
    Loads ResNet50 + FAISS index once and exposes a search() method.
    Designed to be cached with @st.cache_resource in Streamlit.
    """

    def __init__(self):
        print("[Searcher] Loading ResNet50 ...")
        weights      = models.ResNet50_Weights.IMAGENET1K_V1
        model        = models.resnet50(weights=weights)
        model.fc     = nn.Identity()
        model.eval()
        self.model   = model

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        print("[Searcher] Loading FAISS index ...")
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
        self.index = faiss.read_index(str(INDEX_PATH))

        print("[Searcher] Loading image paths ...")
        if not IMAGE_PATHS_PATH.exists():
            raise FileNotFoundError(f"image_paths.json not found at {IMAGE_PATHS_PATH}")
        with open(IMAGE_PATHS_PATH) as f:
            self.image_paths: list[str] = json.load(f)

        print(f"[Searcher] Ready — {self.index.ntotal} vectors indexed.")

    def embed(self, image_bytes: bytes) -> np.ndarray:
        """Convert raw image bytes → normalised (1, 2048) float32 embedding."""
        img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            emb = self.model(tensor).numpy().astype("float32")
        faiss.normalize_L2(emb)
        return emb

    def search(
        self,
        image_bytes: bytes,
        top_k: int = 8,
        fetch_k: int | None = None,
    ) -> dict:
        """
        Search FAISS for top-K similar images.

        Args:
            image_bytes : raw bytes of the query image
            top_k       : number of results to return
            fetch_k     : how many FAISS hits to fetch before filtering
                          (defaults to top_k; set higher when filtering by category)

        Returns:
            {
              "filenames":    [...],           # ranked list of filenames
              "score_map":    {fname: score},  # cosine similarity scores
              "search_time_ms": float,
            }
        """
        if fetch_k is None:
            fetch_k = top_k

        t0 = time.time()
        query_emb = self.embed(image_bytes)
        distances, indices = self.index.search(query_emb, fetch_k)
        search_time_ms = (time.time() - t0) * 1000

        filenames = []
        score_map = {}

        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            fname = Path(self.image_paths[idx]).name
            filenames.append(fname)
            score_map[fname] = round(float(score), 4)

        return {
            "filenames":       filenames,
            "score_map":       score_map,
            "search_time_ms":  round(search_time_ms, 2),
        }

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    @property
    def total_images(self) -> int:
        return len(self.image_paths)