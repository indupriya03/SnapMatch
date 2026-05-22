"""
evaluate.py
-----------
Step 6 of the deep-visual-retrieval pipeline.

Evaluates the visual similarity search system using:
  - Precision@K  : fraction of Top-K results sharing the same class_id
                   as the query image
  - Recall@K     : fraction of all relevant images retrieved in Top-K
  - Search time  : average FAISS query time in milliseconds

How evaluation works:
  1. Load FAISS index, embeddings, image_paths, and MySQL metadata
  2. Sample N query images from the TEST split (unseen during indexing)
  3. For each query:
       - Search FAISS for Top-K similar images
       - Count how many results share the same class_id (relevant)
       - Compute Precision@K and Recall@K
  4. Average metrics across all queries
  5. Save detailed report → outputs/evaluation_report.json

Why use class_id for relevance?
  Two images with the same class_id are the same fine-grained product
  model (e.g. same specific mug design). This gives an objective,
  ground-truth definition of "similar" without any manual annotation.

Usage:
  python src/evaluate.py

Requirements:
  - outputs/faiss_index.bin, embeddings.npy, image_paths.json must exist
  - MySQL must be running with products table populated
  - pip install faiss-cpu numpy tqdm sqlalchemy pymysql python-dotenv
"""

import json
import time
import random
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from db.database import SessionLocal, test_connection
from db import queries

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent.parent
INDEX_PATH       = BASE_DIR / "outputs" / "faiss_index.bin"
EMBEDDINGS_PATH  = BASE_DIR / "outputs" / "embeddings.npy"
IMAGE_PATHS_PATH = BASE_DIR / "outputs" / "image_paths.json"
REPORT_PATH      = BASE_DIR / "outputs" / "evaluation_report.json"

K_VALUES         = [1, 5, 10, 20]   # evaluate Precision and Recall at each K
N_QUERIES        = 200               # number of test images to evaluate
RANDOM_SEED      = 42
# ──────────────────────────────────────────────────────────────────────────────


def load_artifacts() -> tuple[faiss.IndexFlatIP, np.ndarray, list[str]]:
    """
    Load FAISS index, embeddings, and image paths from disk.
    These must already exist (run build_index.py first).
    """
    print("\n[LOAD] Loading FAISS artifacts ...")

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}.\n"
            "Run build_index.py first."
        )
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"embeddings.npy not found at {EMBEDDINGS_PATH}.\n"
            "Run extract_embeddings.py first."
        )
    if not IMAGE_PATHS_PATH.exists():
        raise FileNotFoundError(
            f"image_paths.json not found at {IMAGE_PATHS_PATH}.\n"
            "Run extract_embeddings.py first."
        )

    index       = faiss.read_index(str(INDEX_PATH))
    embeddings  = np.load(EMBEDDINGS_PATH).astype("float32")
    with open(IMAGE_PATHS_PATH) as f:
        image_paths = json.load(f)

    print(f"  FAISS index   : {index.ntotal} vectors")
    print(f"  Embeddings    : {embeddings.shape}")
    print(f"  Image paths   : {len(image_paths)} entries")

    return index, embeddings, image_paths


def build_filename_to_index(image_paths: list[str]) -> dict[str, int]:
    """
    Build a reverse mapping: filename → FAISS index position.
    Used to look up the embedding for a query image by filename.
    """
    return {
        Path(p).name: i for i, p in enumerate(image_paths)
    }


def select_query_images(
    db,
    image_paths: list[str],
    filename_to_idx: dict[str, int],
    n_queries: int,
) -> list[dict]:
    """
    Sample N_QUERIES images from the TEST split to use as query images.

    Why test split only?
      The train split was conceptually "seen" — test images are held out.
      Evaluating on test split gives a fairer measure of generalisation.

    Returns list of dicts:
      {filename, faiss_idx, class_id, category}
    """
    print(f"\n[QUERIES] Selecting {n_queries} test images as queries ...")

    # Get all test split products from MySQL
    all_filenames = list(filename_to_idx.keys())

    # Fetch metadata for all images in one batch
    all_meta = queries.get_products_by_filenames(db, all_filenames)

    # Filter to test split only
    test_images = [
        {
            "filename": fname,
            "faiss_idx": filename_to_idx[fname],
            "class_id": meta["class_id"],
            "category": meta["category"],
        }
        for fname, meta in all_meta.items()
        if meta.get("split") == "test"
        and meta.get("class_id") is not None
        and fname in filename_to_idx
    ]

    if not test_images:
        raise RuntimeError(
            "[ERROR] No test images found in MySQL.\n"
            "Make sure build_metadata.py completed successfully."
        )

    print(f"  Test images available: {len(test_images)}")

    # Sample N_QUERIES randomly
    random.seed(RANDOM_SEED)
    n_select = min(n_queries, len(test_images))
    selected = random.sample(test_images, n_select)

    print(f"  Query images selected: {len(selected)}")
    return selected


def get_relevant_filenames(
    db,
    class_id: int,
    relevant_cache: dict,
    use_super_class: bool = True,
) -> set[str]:
    """
    Return set of all filenames sharing the same class_id as the query.
    Uses a cache to avoid repeated DB queries for the same class_id.
    use_super_class=True  → same product type (bicycle, mug...)
    use_super_class=False → same eBay listing (very strict)
    """
    # if class_id not in relevant_cache:
    #     relevant_cache[class_id] = queries.get_similar_class_filenames(
    #         db, class_id
    #     )
    # return relevant_cache[class_id]
    cache_key = f"super_{class_id}" if use_super_class else class_id

    if cache_key not in relevant_cache:
        if use_super_class:
            relevant_cache[cache_key] = \
                queries.get_similar_super_class_filenames(db, class_id)
        else:
            relevant_cache[cache_key] = \
                queries.get_similar_class_filenames(db, class_id)

    return relevant_cache[cache_key]

def evaluate_single_query(
    query: dict,
    index: faiss.IndexFlatIP,
    embeddings: np.ndarray,
    image_paths: list[str],
    relevant_filenames: set[str],
    max_k: int,
) -> dict:
    """
    Run FAISS search for one query image and compute metrics.

    Returns dict with:
      search_time_ms   : how long FAISS search took
      retrieved        : list of retrieved filenames in rank order
      hits_at_k        : {k: number of relevant results in top-k}
      precision_at_k   : {k: precision@k value}
      recall_at_k      : {k: recall@k value}
    """
    # Get query embedding
    query_embedding = embeddings[query["faiss_idx"]:query["faiss_idx"]+1]

    # FAISS search — fetch max_k + 1 to exclude query itself
    t0 = time.time()
    distances, indices = index.search(query_embedding, max_k + 1)
    search_time_ms = (time.time() - t0) * 1000

    # Build retrieved filename list — exclude the query image itself
    retrieved = []
    for idx in indices[0]:
        if idx == -1:
            continue
        fname = Path(image_paths[idx]).name
        if fname == query["filename"]:
            continue        # skip self
        retrieved.append(fname)
        if len(retrieved) >= max_k:
            break

    # Total relevant images (excluding query itself)
    n_relevant = max(1, len(relevant_filenames) - 1)

    # Compute metrics at each K
    hits_at_k      = {}
    precision_at_k = {}
    recall_at_k    = {}

    for k in K_VALUES:
        top_k   = retrieved[:k]
        hits    = sum(1 for f in top_k if f in relevant_filenames)
        hits_at_k[k]      = hits
        precision_at_k[k] = hits / k
        recall_at_k[k]    = hits / n_relevant

    return {
        "search_time_ms":  round(search_time_ms, 3),
        "retrieved":       retrieved[:max(K_VALUES)],
        "n_relevant":      n_relevant,
        "hits_at_k":       hits_at_k,
        "precision_at_k":  precision_at_k,
        "recall_at_k":     recall_at_k,
    }


def run_evaluation(
    index: faiss.IndexFlatIP,
    embeddings: np.ndarray,
    image_paths: list[str],
    query_images: list[dict],
    db,use_super_class: bool = True,
) -> dict:
    """
    Run evaluation across all query images and aggregate metrics.
    Returns full results dict ready for saving and printing.
    """
    max_k          = max(K_VALUES)
    relevant_cache = {}   # class_id → set of relevant filenames

    # Accumulators
    all_precision  = defaultdict(list)
    all_recall     = defaultdict(list)
    all_times      = []

    per_query_results = []

    print(f"\n[EVALUATE] Running {len(query_images)} queries ...")

    for query in tqdm(query_images, desc="  Evaluating", unit="query"):
        # Get relevant filenames for this query's class
        relevant = get_relevant_filenames(db, query["class_id"], relevant_cache,use_super_class=use_super_class)

        # Run search and compute metrics
        result = evaluate_single_query(
            query, index, embeddings, image_paths, relevant, max_k
        )

        # Accumulate
        for k in K_VALUES:
            all_precision[k].append(result["precision_at_k"][k])
            all_recall[k].append(result["recall_at_k"][k])
        all_times.append(result["search_time_ms"])

        per_query_results.append({
            "filename":      query["filename"],
            "class_id":      query["class_id"],
            "category":      query["category"],
            "n_relevant":    result["n_relevant"],
            "search_time_ms": result["search_time_ms"],
            "precision_at_k": result["precision_at_k"],
            "recall_at_k":    result["recall_at_k"],
        })

    # Aggregate mean metrics
    mean_precision = {k: float(np.mean(all_precision[k])) for k in K_VALUES}
    mean_recall    = {k: float(np.mean(all_recall[k]))    for k in K_VALUES}
    mean_time      = float(np.mean(all_times))
    max_time       = float(np.max(all_times))

    # Per-category breakdown
    cat_precision = defaultdict(lambda: defaultdict(list))
    for r in per_query_results:
        for k in K_VALUES:
            cat_precision[r["category"]][k].append(r["precision_at_k"][k])

    category_metrics = {
        cat: {
            f"precision@{k}": round(float(np.mean(vals[k])), 4)
            for k in K_VALUES
        }
        for cat, vals in cat_precision.items()
    }

    return {
        "n_queries":           len(query_images),
        "k_values":            K_VALUES,
        "mean_precision_at_k": {str(k): round(v, 4) for k, v in mean_precision.items()},
        "mean_recall_at_k":    {str(k): round(v, 4) for k, v in mean_recall.items()},
        "mean_search_time_ms": round(mean_time, 3),
        "max_search_time_ms":  round(max_time, 3),
        "sub_second_search":   max_time < 1000,
        "category_metrics":    category_metrics,
        "per_query":           per_query_results,
    }


def save_report(results: dict, label: str = "strict") -> None:
    path = BASE_DIR / "outputs" / f"evaluation_report_{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[REPORT] Saved to {path}")


def print_summary(results: dict, label: str = "") -> None:
    """Print a clean evaluation summary to the console."""
    print("\n" + "=" * 55)
    print(f"           EVALUATION RESULTS SUMMARY — {label}")
    print("=" * 55)
    print(f"  Queries evaluated   : {results['n_queries']}")
    print(f"  Avg search time     : {results['mean_search_time_ms']} ms")
    print(f"  Max search time     : {results['max_search_time_ms']} ms")
    print(f"  Sub-second search   : {'✅ Yes' if results['sub_second_search'] else '❌ No'}")

    print("\n  Precision@K (fraction of Top-K that are relevant):")
    print(f"  {'K':<6} {'Precision':>10} {'Recall':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10}")
    for k in results["k_values"]:
        p = results["mean_precision_at_k"][str(k)]
        r = results["mean_recall_at_k"][str(k)]
        print(f"  {k:<6} {p:>10.4f} {r:>10.4f}")

    print("\n  Per-category Precision@5:")
    print(f"  {'Category':<28} {'P@5':>8}")
    print(f"  {'-'*28} {'-'*8}")
    for cat, metrics in sorted(results["category_metrics"].items()):
        p5 = metrics.get("precision@5", 0)
        print(f"  {cat:<28} {p5:>8.4f}")

    print("=" * 55)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("      STEP 6 — EVALUATION")
    print("      Precision@K · Recall@K · Search Time")
    print("=" * 55)

    try:
        # 1. Load FAISS artifacts
        index, embeddings, image_paths = load_artifacts()

        # 2. Build reverse lookup
        filename_to_idx = build_filename_to_index(image_paths)

        # 3. Connect to MySQL
        print("\n[DB] Connecting to MySQL ...")
        test_connection()
        db = SessionLocal()

        try:
            # 4. Select query images from test split
            query_images = select_query_images(
                db, image_paths, filename_to_idx, N_QUERIES
            )

            # ── Run STRICT evaluation (same class_id) ──────────────────
            print("\n" + "─" * 55)
            print("  [STRICT] Same class_id — same eBay product listing")
            print("─" * 55)
            results_strict = run_evaluation(
                index, embeddings, image_paths,
                query_images, db,
                use_super_class=False,        # ← strict
            )

            # ── Run RELAXED evaluation (same super_class_id) ───────────
            print("\n" + "─" * 55)
            print("  [RELAXED] Same super_class_id — same product type")
            print("─" * 55)
            results_relaxed = run_evaluation(
                index, embeddings, image_paths,
                query_images, db,
                use_super_class=True,         # ← relaxed
            )

        finally:
            db.close()

        # 6. Save and print report
        save_report(results_strict,  "strict")
        save_report(results_relaxed, "relaxed")
        print_summary(results_strict,  label="STRICT  (class_id)")
        print_summary(results_relaxed, label="RELAXED (super_class_id)")


        print("\n[DONE] Evaluation complete.\n")

    except (FileNotFoundError, RuntimeError) as e:
        print(e)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Evaluation cancelled.")