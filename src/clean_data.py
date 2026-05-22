"""
clean_data.py
-------------
Data cleaning for the deep-visual-retrieval pipeline.

Two passes — both in this single file:

  Pass 1 (structural) — no embeddings needed:
    1. Corrupt check     — PIL verify
    2. Format check      — JPEG or PNG only
    3. Size check        — minimum 50 × 50 pixels
    4. Channel check     — RGB (3 channels)
    5. Blank check       — pixel std below threshold
    6. Duplicate check   — MD5 perceptual hash

  Pass 2 (semantic noise) — requires embeddings.npy:
    7. Outlier check     — cosine similarity to category centroid
                           images below threshold are mislabeled
                           or visually unrelated to their category

Why two passes?
  Structural checks work on raw pixels — no embeddings needed.
  Semantic noise detection needs embeddings to measure how well
  an image fits its category. Embeddings are generated between
  Pass 1 and Pass 2.

Usage:
  python src/clean_data.py --pass 1    # structural checks
  python src/clean_data.py --pass 2    # noise detection
  python src/clean_data.py             # runs Pass 1 by default

Pipeline order:
  Step 2 → clean_data.py --pass 1
  Step 3 → extract_embeddings.py
  Step 4 → clean_data.py --pass 2
  Step 5 → extract_embeddings.py  (re-run on clean data)
  Step 6 → build_index.py
"""

import os
import json
import argparse
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parent.parent
DATA_DIR          = BASE_DIR / "data" / "my_project_data"
CLEANED_DIR       = BASE_DIR / "data" / "cleaned_data"
EMBEDDINGS_PATH   = BASE_DIR / "outputs" / "embeddings.npy"
IMAGE_PATHS_PATH  = BASE_DIR / "outputs" / "image_paths.json"
REPORT_PASS1_PATH = BASE_DIR / "outputs" / "cleaning_report.json"
REPORT_PASS2_PATH = BASE_DIR / "outputs" / "noise_report.json"

# Pass 1 settings
MIN_SIZE          = 50       # minimum width AND height in pixels
BLANK_STD_LIMIT   = 2.0      # pixel std below this = blank image

# Pass 2 settings
OUTLIER_THRESHOLD = 0.50     # cosine similarity below this = flagged
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — STRUCTURAL CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def get_image_hash(img: Image.Image) -> str:
    """Perceptual hash for duplicate detection."""
    img_small = img.resize((16, 16)).convert("RGB")
    return hashlib.md5(np.array(img_small).tobytes()).hexdigest()


def is_blank(img: Image.Image) -> bool:
    """True if image has almost no pixel variation — blank or solid colour."""
    arr = np.array(img.convert("RGB")).astype(np.float32)
    return float(arr.std()) < BLANK_STD_LIMIT


def check_image(
    img_path: Path,
    seen_hashes: dict,
) -> tuple[bool, str]:
    """
    Run all 6 structural checks on one image.
    Returns (passed, reason).
    """
    # Check 1 — Corrupt?
    try:
        img = Image.open(img_path)
        img.verify()
        img = Image.open(img_path)   # must reopen after verify()
    except (UnidentifiedImageError, Exception):
        return False, "corrupt"

    # Check 2 — Valid format?
    if img.format not in ("JPEG", "PNG"):
        return False, "wrong_format"

    # Check 3 — Minimum size?
    w, h = img.size
    if w < MIN_SIZE or h < MIN_SIZE:
        return False, "too_small"

    # Check 4 — RGB channels?
    img = img.convert("RGB")
    if img.mode != "RGB":
        return False, "non_rgb"

    # Check 5 — Not blank?
    if is_blank(img):
        return False, "blank"

    # Check 6 — Not duplicate?
    img_hash = get_image_hash(img)
    if img_hash in seen_hashes:
        return False, f"duplicate_of:{seen_hashes[img_hash]}"
    seen_hashes[img_hash] = str(img_path)

    return True, "passed"


def run_pass1() -> dict:
    """
    Pass 1 — structural cleaning.
    Iterates all images, runs 6 checks, deletes failures in place.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Data folder not found: {DATA_DIR}\n"
            "Run download_data.py first.\n"
        )

    report = {
        "pass":           1,
        "description":    "Structural checks",
        "data_dir":       str(DATA_DIR),
        "min_size_px":    MIN_SIZE,
        "total_found":    0,
        "total_removed":  0,
        "total_passed":   0,
        "removed_by_reason": {
            "corrupt":      [],
            "wrong_format": [],
            "too_small":    [],
            "non_rgb":      [],
            "blank":        [],
            "duplicate":    [],
        },
        "per_category":   {},
    }

    seen_hashes   = {}
    category_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

    print(f"\n[PASS 1] Structural cleaning — {len(category_dirs)} categories ...\n")

    for cat_dir in category_dirs:
        image_files = list({
            f for f in cat_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        })

        cat_report = {"found": len(image_files), "removed": 0, "passed": 0}
        report["total_found"] += len(image_files)

        for img_path in tqdm(image_files, desc=f"  {cat_dir.name}", unit="img"):
            passed, reason = check_image(img_path, seen_hashes)

            if passed:
                cat_report["passed"] += 1
                report["total_passed"] += 1
            else:
                try:
                    os.remove(img_path)
                except OSError as e:
                    print(f"\n  [WARN] Could not delete {img_path.name}: {e}")

                cat_report["removed"] += 1
                report["total_removed"] += 1

                if reason.startswith("duplicate_of:"):
                    report["removed_by_reason"]["duplicate"].append(str(img_path))
                elif reason in report["removed_by_reason"]:
                    report["removed_by_reason"][reason].append(str(img_path))

        report["per_category"][cat_dir.name] = cat_report

    # ── Copy surviving images to cleaned_data/ ────────────────────────────────
    import shutil
    print(f"\n[COPY] Copying clean images to {CLEANED_DIR} ...")
    if CLEANED_DIR.exists():
        shutil.rmtree(CLEANED_DIR)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    for cat_dir in sorted(DATA_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        dest_cat = CLEANED_DIR / cat_dir.name
        dest_cat.mkdir(exist_ok=True)
        for img_path in cat_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy2(img_path, dest_cat / img_path.name)
                copied += 1
    print(f"  ✅ {copied} images copied to {CLEANED_DIR}")
    report["cleaned_dir"] = str(CLEANED_DIR)
    report["total_copied"] = copied

    return report


def print_pass1_summary(report: dict) -> None:
    print("\n" + "=" * 50)
    print("     DATA CLEANING — PASS 1 REPORT")
    print("=" * 50)
    print(f"  Total images scanned  : {report['total_found']}")
    print(f"  Corrupt               : {len(report['removed_by_reason']['corrupt'])}")
    print(f"  Wrong format          : {len(report['removed_by_reason']['wrong_format'])}")
    print(f"  Too small (<50px)     : {len(report['removed_by_reason']['too_small'])}")
    print(f"  Non-RGB               : {len(report['removed_by_reason']['non_rgb'])}")
    print(f"  Blank                 : {len(report['removed_by_reason']['blank'])}")
    print(f"  Duplicates            : {len(report['removed_by_reason']['duplicate'])}")
    print("-" * 50)
    print(f"  Total removed         : {report['total_removed']}")
    print(f"  ✅ Clean images left  : {report['total_passed']}")
    print("=" * 50)


# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — SEMANTIC NOISE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def load_embeddings() -> tuple[np.ndarray, list[str]]:
    """Load Pass 1 embeddings from disk."""
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"\n[ERROR] embeddings.npy not found at {EMBEDDINGS_PATH}\n"
            "Run extract_embeddings.py (Pass 1) first.\n"
        )
    if not IMAGE_PATHS_PATH.exists():
        raise FileNotFoundError(
            f"\n[ERROR] image_paths.json not found at {IMAGE_PATHS_PATH}\n"
            "Run extract_embeddings.py (Pass 1) first.\n"
        )
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    with open(IMAGE_PATHS_PATH) as f:
        image_paths = json.load(f)

    print(f"  Embeddings loaded : {embeddings.shape}")
    print(f"  Image paths loaded: {len(image_paths)}")
    return embeddings, image_paths


def compute_centroids(
    embeddings: np.ndarray,
    image_paths: list[str],
) -> tuple[dict, dict]:
    """
    Compute L2-normalised centroid per category.
    The centroid = mean of all embeddings in that category,
    normalised to unit length.
    A good image should have high cosine similarity to its centroid.
    A mislabeled or noisy image will have low similarity.
    """
    category_indices = defaultdict(list)
    for i, path in enumerate(image_paths):
        cat = Path(path).parent.name
        category_indices[cat].append(i)

    centroids = {}
    for cat, indices in category_indices.items():
        cat_embs  = embeddings[indices]
        mean_emb  = cat_embs.mean(axis=0)
        mean_emb /= np.linalg.norm(mean_emb)   # normalise to unit vector
        centroids[cat] = mean_emb

    return centroids, category_indices


def run_pass2() -> dict:
    """
    Pass 2 — semantic noise detection.
    Loads embeddings, computes category centroids, flags and removes
    images whose cosine similarity to centroid is below OUTLIER_THRESHOLD.
    """
    print(f"\n[PASS 2] Semantic noise detection ...")
    print(f"  Outlier threshold : {OUTLIER_THRESHOLD} (cosine similarity)")

    embeddings, image_paths = load_embeddings()
    centroids, category_indices = compute_centroids(embeddings, image_paths)

    # Normalise all embeddings once for fast dot product
    norms  = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-10, None)

    report = {
        "pass":             2,
        "description":      "Semantic noise detection",
        "outlier_threshold": OUTLIER_THRESHOLD,
        "total_checked":    len(image_paths),
        "total_flagged":    0,
        "total_removed":    0,
        "per_category":     {},
    }

    print(f"\n  {'Category':<28} {'Images':>7} {'Flagged':>8} {'Removed':>8}")
    print(f"  {'-'*28} {'-'*7} {'-'*8} {'-'*8}")

    for cat, indices in sorted(category_indices.items()):
        centroid   = centroids[cat]             # shape (2048,)
        cat_normed = normed[indices]            # shape (N, 2048)

        # Cosine similarity of each image to its category centroid
        sims = (cat_normed @ centroid).flatten()

        flagged = []
        removed = 0

        for idx, sim in zip(indices, sims):
            if sim < OUTLIER_THRESHOLD:
                img_path = Path(image_paths[idx])
                flagged.append({
                    "filename":               img_path.name,
                    "image_path":             str(img_path),
                    "similarity_to_centroid": round(float(sim), 4),
                })

                # Remove from disk in place
                # Remove from cleaned_data copy, not the original
                clean_path = CLEANED_DIR / Path(image_paths[idx]).parent.name / img_path.name
                target = clean_path if clean_path.exists() else img_path
                if target.exists():
                    os.remove(target)
                    removed += 1

        report["per_category"][cat] = {
            "total_images": len(indices),
            "flagged":      len(flagged),
            "removed":      removed,
            "outliers":     sorted(
                flagged,
                key=lambda x: x["similarity_to_centroid"]
            ),
        }

        report["total_flagged"] += len(flagged)
        report["total_removed"] += removed

        print(
            f"  {cat:<28} {len(indices):>7} "
            f"{len(flagged):>8} {removed:>8}"
        )

    return report


def print_pass2_summary(report: dict) -> None:
    remaining = report["total_checked"] - report["total_removed"]
    print("\n" + "=" * 50)
    print("     DATA CLEANING — PASS 2 REPORT")
    print("=" * 50)
    print(f"  Outlier threshold  : {report['outlier_threshold']}")
    print(f"  Total checked      : {report['total_checked']}")
    print(f"  Total flagged      : {report['total_flagged']}")
    print(f"  Total removed      : {report['total_removed']}")
    print(f"  ✅ Clean images    : {remaining}")
    print("=" * 50)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED — Save report
# ══════════════════════════════════════════════════════════════════════════════

def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] Saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data cleaning — structural (Pass 1) or noise (Pass 2)"
    )
    parser.add_argument(
        "--pass",
        type=int,
        choices=[1, 2],
        default=1,
        dest="cleaning_pass",
        help="Which pass to run: 1=structural, 2=noise detection (default: 1)",
    )
    args = parser.parse_args()

    if args.cleaning_pass == 1:
        print("=" * 50)
        print("  STEP 2 — DATA CLEANING PASS 1 (Structural)")
        print("=" * 50)
        try:
            report = run_pass1()
            save_report(report, REPORT_PASS1_PATH)
            print_pass1_summary(report)
            print("\n[DONE] Pass 1 complete.")
            print("       Next: run extract_embeddings.py\n")
        except FileNotFoundError as e:
            print(e)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Cleaning cancelled.")

    elif args.cleaning_pass == 2:
        print("=" * 50)
        print("  STEP 4 — DATA CLEANING PASS 2 (Noise Detection)")
        print("=" * 50)
        try:
            report = run_pass2()
            save_report(report, REPORT_PASS2_PATH)
            print_pass2_summary(report)
            print("\n[DONE] Pass 2 complete.")
            print("       Next: re-run extract_embeddings.py\n")
        except FileNotFoundError as e:
            print(e)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Noise detection cancelled.")