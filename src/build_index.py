"""
build_index.py
--------------
Step 4 of the deep-visual-retrieval pipeline.

What this script does:
  1. Loads embeddings.npy        (shape: N × 2048)
  2. Loads image_paths.json      (list of N file paths)
  3. Normalises all embeddings   (required for cosine similarity)
  4. Builds a FAISS IndexFlatIP  (exact cosine similarity search)
  5. Verifies the index with a sample search
  6. Saves faiss_index.bin       → outputs/faiss_index.bin
  7. Saves build_report.json     → outputs/build_report.json

Why IndexFlatIP?
  IP = Inner Product. After L2 normalisation, inner product between
  two vectors equals their cosine similarity. This is the standard
  approach for image retrieval — it ranks results by visual similarity
  regardless of embedding magnitude.

Why not a more complex index (IVF, HNSW)?
  IndexFlatIP does exact search — it checks every vector. For 9,600
  images this is completely fine and returns results in milliseconds.
  Approximate indexes (IVF, HNSW) trade accuracy for speed and are
  only needed for millions of vectors. For a portfolio project,
  exact search is the correct and honest choice.

Usage:
  python src/build_index.py

Requirements:
  - pip install faiss-cpu numpy
  - outputs/embeddings.npy and outputs/image_paths.json must exist
    (run extract_embeddings.py first)
"""

import json
import time
import numpy as np
import faiss
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDINGS_PATH  = Path("outputs/embeddings.npy")
IMAGE_PATHS_PATH = Path("outputs/image_paths.json")
INDEX_PATH       = Path("outputs/faiss_index.bin")
REPORT_PATH      = Path("outputs/build_report.json")

EMBEDDING_DIM    = 2048     # must match ResNet50 output
TOP_K_VERIFY     = 5        # number of results to fetch in verification search
# ──────────────────────────────────────────────────────────────────────────────


def load_embeddings() -> tuple[np.ndarray, list[str]]:
    """
    Load embeddings and image paths saved by extract_embeddings.py.

    Validates:
      - Both files exist
      - Row count matches between embeddings and image paths
      - Embedding dimension matches EMBEDDING_DIM
    """
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"\n[ERROR] embeddings.npy not found at {EMBEDDINGS_PATH}\n"
            "Run extract_embeddings.py first.\n"
        )
    if not IMAGE_PATHS_PATH.exists():
        raise FileNotFoundError(
            f"\n[ERROR] image_paths.json not found at {IMAGE_PATHS_PATH}\n"
            "Run extract_embeddings.py first.\n"
        )

    print("\n[LOAD] Loading embeddings ...")
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    print(f"  embeddings.npy loaded   : shape {embeddings.shape}")

    with open(IMAGE_PATHS_PATH, "r") as f:
        image_paths = json.load(f)
    print(f"  image_paths.json loaded : {len(image_paths)} entries")

    # Validate alignment
    if embeddings.shape[0] != len(image_paths):
        raise ValueError(
            f"[ERROR] Mismatch: embeddings has {embeddings.shape[0]} rows "
            f"but image_paths has {len(image_paths)} entries.\n"
            "Re-run extract_embeddings.py to regenerate both files together."
        )

    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"[ERROR] Expected embedding dim {EMBEDDING_DIM}, "
            f"got {embeddings.shape[1]}.\n"
            "Check that extract_embeddings.py used ResNet50."
        )

    print(f"  Validation              : ✅ shapes aligned")
    return embeddings, image_paths


def normalise_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalise all embedding vectors in place.

    Why this is needed:
      FAISS IndexFlatIP computes raw inner products (dot products).
      For two unit vectors (L2 norm = 1.0), the inner product equals
      cosine similarity. Without normalisation, vectors with larger
      magnitudes would dominate the search regardless of direction,
      giving wrong results.

    After normalisation:
      Every vector has L2 norm = 1.0
      Inner product between any two vectors = cosine similarity
      Range: -1.0 (opposite) to 1.0 (identical)
    """
    print("\n[NORMALISE] L2-normalising embeddings ...")

    # Check norms before
    norms_before = np.linalg.norm(embeddings, axis=1)
    print(f"  Norms before : min={norms_before.min():.4f}  "
          f"max={norms_before.max():.4f}  "
          f"mean={norms_before.mean():.4f}")

    # faiss.normalize_L2 modifies the array in place
    faiss.normalize_L2(embeddings)

    # Verify norms after — should all be 1.0
    norms_after = np.linalg.norm(embeddings, axis=1)
    print(f"  Norms after  : min={norms_after.min():.4f}  "
          f"max={norms_after.max():.4f}  "
          f"mean={norms_after.mean():.4f}")
    print(f"  ✅ All vectors normalised to unit length")

    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP (exact inner product search).

    Process:
      1. Create an empty index for EMBEDDING_DIM-dimensional vectors
      2. Add all normalised embeddings to the index
      3. The index is now ready for similarity search

    IndexFlatIP stores all vectors in memory as a flat array.
    Search scans all N vectors for every query — exact, no approximation.
    At 9,600 vectors this takes <10ms per query on CPU.
    """
    print(f"\n[INDEX] Building FAISS IndexFlatIP ...")
    print(f"  Index type    : IndexFlatIP (exact cosine similarity)")
    print(f"  Dimension     : {EMBEDDING_DIM}")
    print(f"  Vectors to add: {embeddings.shape[0]}")

    start = time.time()

    # Create the index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    # Add all embeddings — this is instantaneous for 9,600 vectors
    index.add(embeddings)

    elapsed = time.time() - start

    print(f"  Vectors stored: {index.ntotal}")
    print(f"  Build time    : {elapsed:.2f} seconds")
    print(f"  ✅ Index built successfully")

    return index


def verify_index(
    index: faiss.IndexFlatIP,
    embeddings: np.ndarray,
    image_paths: list[str],
) -> dict:
    """
    Run a quick sanity check search to verify the index works correctly.

    Uses the first image in the dataset as a query.
    The top result should always be the query image itself
    with a similarity score of 1.0 (identical vector).

    Returns a verification report dict.
    """
    print(f"\n[VERIFY] Running sanity check search ...")

    query = embeddings[0:1]             # shape (1, 2048) — first image
    query_path = image_paths[0]

    distances, indices = index.search(query, TOP_K_VERIFY)

    print(f"  Query image   : {Path(query_path).name}")
    print(f"  Top-{TOP_K_VERIFY} results:")

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
        result_path = Path(image_paths[idx]).name
        marker = " ← query (should be rank 1, score 1.0)" if rank == 0 else ""
        print(f"    Rank {rank+1}: {result_path}  (similarity: {score:.4f}){marker}")
        results.append({
            "rank": rank + 1,
            "image": result_path,
            "similarity_score": float(score),
        })

    # Sanity check: rank 1 should be the query itself with score ≈ 1.0
    top_score = float(distances[0][0])
    top_idx   = int(indices[0][0])
    self_match = (top_idx == 0 and abs(top_score - 1.0) < 1e-4)

    if self_match:
        print(f"\n  ✅ Sanity check passed: query returned itself at rank 1 "
              f"with score {top_score:.4f}")
    else:
        print(f"\n  ⚠ Sanity check warning: unexpected top result. "
              f"Index may have an issue.")

    return {
        "query_image": query_path,
        "top_k": TOP_K_VERIFY,
        "results": results,
        "sanity_check_passed": self_match,
    }


def save_index(index: faiss.IndexFlatIP) -> None:
    """Save the FAISS index to disk as a binary file."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    size_mb = INDEX_PATH.stat().st_size / 1_000_000
    print(f"\n[SAVE] faiss_index.bin → {INDEX_PATH}  ({size_mb:.1f} MB)")


def save_report(
    embeddings: np.ndarray,
    image_paths: list[str],
    verification: dict,
) -> None:
    """Save the build report to outputs/build_report.json."""
    report = {
        "index_type": "IndexFlatIP",
        "similarity_metric": "cosine (via L2 normalisation + inner product)",
        "embedding_dim": EMBEDDING_DIM,
        "total_vectors": embeddings.shape[0],
        "total_images": len(image_paths),
        "index_path": str(INDEX_PATH),
        "embeddings_path": str(EMBEDDINGS_PATH),
        "image_paths_path": str(IMAGE_PATHS_PATH),
        "verification": verification,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVE] build_report.json → {REPORT_PATH}")


def print_summary(embeddings: np.ndarray, verification: dict) -> None:
    """Print a clean final summary."""
    print("\n" + "=" * 50)
    print("          FAISS INDEX BUILD SUMMARY")
    print("=" * 50)
    print(f"  Index type      : IndexFlatIP")
    print(f"  Similarity      : Cosine (L2 norm + inner product)")
    print(f"  Vectors stored  : {embeddings.shape[0]}")
    print(f"  Embedding dim   : {EMBEDDING_DIM}")
    print(f"  Sanity check    : "
          f"{'✅ passed' if verification['sanity_check_passed'] else '⚠ warning'}")
    print("-" * 50)
    print(f"  Saved to:")
    print(f"    {INDEX_PATH}")
    print(f"    {REPORT_PATH}")
    print("=" * 50)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("       STEP 4 — BUILD FAISS INDEX")
    print("=" * 50)

    try:
        # 1. Load embeddings and image paths
        embeddings, image_paths = load_embeddings()

        # 2. L2-normalise for cosine similarity
        embeddings = normalise_embeddings(embeddings)

        # 3. Build the FAISS index
        index = build_faiss_index(embeddings)

        # 4. Verify with a sample search
        verification = verify_index(index, embeddings, image_paths)

        # 5. Save index to disk
        save_index(index)

        # 6. Save report
        save_report(embeddings, image_paths, verification)

        # 7. Summary
        print_summary(embeddings, verification)

        print("\n[DONE] Step 4 complete. Run app.py next.\n")

    except (FileNotFoundError, ValueError) as e:
        print(e)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Index build cancelled by user.")