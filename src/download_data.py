"""
download_data.py
----------------
Step 1 of the deep-visual-retrieval pipeline.

What this script does:
  1. Downloads the Stanford Online Products dataset from Kaggle
  2. Unzips it locally
  3. Samples a fixed number of images per category (subset selection)
  4. Saves the subset to  data/my_project_data/
  5. Saves filtered metadata to  data/metadata/
  6. Deletes the large zip and full dataset to free disk space
  7. Saves a download report to  outputs/download_report.json

Usage:
  python src/download_data.py

Requirements:
  - kaggle CLI configured (~/.kaggle/kaggle.json must exist)
  - pip install kaggle tqdm
"""

from multiprocessing.util import info
import os
import json
import random
import shutil
import zipfile
from pathlib import Path
from streamlit.config import cat
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
#KAGGLE_DATASET   = "liucong12601/stanford-online-products-dataset"
ZIP_NAME         = "archive.zip"
EXTRACTED_FOLDER = "Stanford_Online_Products"          # folder name inside zip
DATA_DIR         = Path("data/my_project_data")        # subset destination
DOWNLOAD_DIR     = Path("data/raw_download")           # temp download location
REPORT_PATH      = Path("outputs/download_report.json")
IMAGES_PER_CAT   = 800                                 # 800 × 12 = 9,600 images
RANDOM_SEED      = 42

# All 12 SOP categories
CATEGORIES = [
    "bicycle_final",
    "cabinet_final",
    "chair_final",
    "coffee_maker_final",
    "fan_final",
    "kettle_final",
    "lamp_final",
    "mug_final",
    "sofa_final",
    "stapler_final",
    "table_final",
    "toaster_final",
]
# ──────────────────────────────────────────────────────────────────────────────

def download_dataset() -> Path:
    """Check if zip exists locally — downloaded manually by user."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOAD_DIR / ZIP_NAME

    if zip_path.exists():
        print(f"[OK] Zip found at {zip_path}")
        return zip_path
    else:
        raise FileNotFoundError(
            f"\n[ERROR] Zip not found at {zip_path}\n"
            "Please download manually from:\n"
            "https://www.kaggle.com/datasets/liucong12601/stanford-online-products-dataset\n"
            f"Then move the zip file to:\n"
            f"{zip_path}\n"
        )


def unzip_dataset(zip_path: Path) -> Path:
    """Unzip the downloaded file into DOWNLOAD_DIR."""
    extract_path = DOWNLOAD_DIR / EXTRACTED_FOLDER

    if extract_path.exists():
        print(f"[SKIP] Already extracted at {extract_path} — skipping unzip.")
        return extract_path

    print(f"\n[UNZIP] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="  Extracting", unit="file"):
            zf.extract(member, DOWNLOAD_DIR)

    print(f"[OK] Extracted to {extract_path}")
    return extract_path


def select_subset(extract_path: Path) -> tuple[dict, set]:
    """
    Sample IMAGES_PER_CAT images from each category folder.
    Copies selected images into DATA_DIR/category_name/.
    Returns a report dictionary and set of selected relative paths.
    """
    random.seed(RANDOM_SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "images_per_category": IMAGES_PER_CAT,
        "random_seed": RANDOM_SEED,
        "categories": {},
        "total_selected": 0,
    }

    # ── FIX: track selected paths for metadata filtering ──
    selected_paths = set()

    print(f"\n[SUBSET] Selecting {IMAGES_PER_CAT} images per category ...\n")

    for category in CATEGORIES:
        source_cat = extract_path / category

        # Handle both direct and nested extraction paths
        if not source_cat.exists():
            # Try one level deeper (some zips nest differently)
            source_cat = extract_path / EXTRACTED_FOLDER / category
        if not source_cat.exists():
            print(f"  [WARN] Category folder not found: {category} — skipping.")
            continue

        # ── FIX: use set comprehension to avoid Windows case-insensitive
        #         glob double-counting (*.jpg and *.JPG return same files)
        images = list({
            f for f in source_cat.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg"}
        })

        if not images:
            print(f"  [WARN] No images found in {category} — skipping.")
            continue

        # Sample min(IMAGES_PER_CAT, available) images
        n_select = min(IMAGES_PER_CAT, len(images))
        selected = random.sample(images, n_select)

        # Copy to destination
        dest_cat = DATA_DIR / category
        dest_cat.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(selected, desc=f"  {category}", unit="img"):
            shutil.copy2(img_path, dest_cat / img_path.name)
            # ── FIX: record relative path matching Ebay_train/test.txt format
            selected_paths.add(f"{category}/{img_path.name}")

        report["categories"][category] = {
            "available": len(images),
            "selected": n_select,
            "destination": str(dest_cat),
        }
        report["total_selected"] += n_select

        print(
            f"  [OK] {category}: "
            f"{n_select} selected from {len(images)} available"
        )

    return report, selected_paths


def save_filtered_metadata(extract_path: Path, selected_paths: set) -> None:
    """Filter Ebay_train/test.txt to only rows matching our subset."""
    META_DIR = Path("data/metadata")
    META_DIR.mkdir(parents=True, exist_ok=True)

    for txt_file in ["Ebay_train.txt", "Ebay_test.txt"]:
        src = extract_path / txt_file
        if not src.exists():
            print(f"  [WARN] {txt_file} not found — skipping.")
            continue

        with open(src, "r") as f:
            lines = f.readlines()

        header = lines[0]  # first line is column names

        # Keep only rows whose path matches a selected image
        filtered = [
            line for line in lines[1:]
            if line.strip().split()[-1] in selected_paths
        ]

        out_path = META_DIR / txt_file
        with open(out_path, "w") as f:
            f.write(header)
            f.writelines(filtered)

        print(f"  Saved {len(filtered)} rows → {txt_file}")


def cleanup_raw(zip_path: Path, extract_path: Path) -> None:
    """
    Delete the zip file and full extracted dataset to free disk space.
    Only the subset in DATA_DIR is kept.
    """
    print("\n[CLEANUP] Removing large temporary files to free disk space ...")

    if zip_path.exists():
        zip_size_mb = zip_path.stat().st_size / 1_000_000
        zip_path.unlink()
        print(f"  Deleted zip: {zip_path.name}  ({zip_size_mb:.0f} MB freed)")

    if extract_path.exists():
        shutil.rmtree(extract_path)
        print(f"  Deleted extracted folder: {extract_path.name}")

    # Remove raw_download dir if empty
    try:
        DOWNLOAD_DIR.rmdir()
        print(f"  Removed empty folder: {DOWNLOAD_DIR}")
    except OSError:
        pass  # not empty, leave it


def save_report(report: dict) -> None:
    """Save the download + subset report to outputs/."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] Saved to {REPORT_PATH}")


def print_summary(report: dict) -> None:
    """Print a clean summary table to the console."""
    print("\n" + "=" * 50)
    print("         DOWNLOAD & SUBSET SUMMARY")
    print("=" * 50)
    print(f"  Images per category : {report['images_per_category']}")
    print(f"  Random seed         : {report['random_seed']}")
    print(f"  Categories found    : {len(report['categories'])}")
    print(f"  Total images saved  : {report['total_selected']}")
    print("-" * 50)
    for cat, info in report["categories"].items():
        print(f"  {cat:<28} {info['selected']} images (available: {info['available']})")    
    print("=" * 50)
    print(f"\n  Subset saved to: {DATA_DIR}/")
    print(f"  Report saved to: {REPORT_PATH}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   STEP 1 — DATA DOWNLOAD & SUBSET SELECTION")
    print("=" * 50)

    try:
     
        # 1. Download from Kaggle
        zip_path = download_dataset()

        # 2. Unzip
        extract_path = unzip_dataset(zip_path)

        # 3. Select subset
        report, selected_paths = select_subset(extract_path)

        # 4. Save filtered metadata before deleting raw
        print("\n[METADATA] Saving filtered metadata ...")
        save_filtered_metadata(extract_path, selected_paths)

        # 5. Delete zip + full dataset to save space
        cleanup_raw(zip_path, extract_path)

        # 6. Save and print report
        save_report(report)
        print_summary(report)

        print("[DONE] Step 1 complete. Run clean_data.py next.\n")

    except FileNotFoundError as e:
        print(e)
    except RuntimeError as e:
        print(e)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Download cancelled by user.")