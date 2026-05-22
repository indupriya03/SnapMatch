"""
build_metadata.py
-----------------
Step 5 of the deep-visual-retrieval pipeline.

What this script does:
  1. Connects to MySQL and creates all tables (if not exist)
  2. Parses Ebay_train.txt and Ebay_test.txt from my_project_data/
  3. Filters rows to only images that exist in our 9,569 subset
  4. Populates three tables in order:
       super_categories → categories → products
  5. Updates image counts per category
  6. Saves a report → outputs/metadata_report.json

Why filter to subset only?
  Ebay_train.txt contains metadata for ALL 120k original images.
  We only have 9,569 images in our subset. We insert only the rows
  that match our actual image files — no orphan records.

Usage:
  python src/build_metadata.py

Requirements:
  - MySQL running with database 'deep_visual_retrieval' created
  - .env file configured with DB credentials
  - data/my_project_data/ must exist (run download + clean first)
  - pip install sqlalchemy pymysql python-dotenv
"""

import json
import time
from pathlib import Path
from tqdm import tqdm
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


from db.database import create_tables, SessionLocal, test_connection
from db.models import SuperCategory, Category, Product
from db import queries

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent.parent
DATA_DIR         = BASE_DIR / "data" / "my_project_data"
REPORT_PATH      = BASE_DIR / "outputs" / "metadata_report.json"
EBAY_DIR         = BASE_DIR / "data" / "metadata"

# Ebay txt files saved by download_data.py into my_project_data/
TRAIN_TXT        = EBAY_DIR / "Ebay_train.txt"
TEST_TXT         = EBAY_DIR / "Ebay_test.txt"

# Clean display names for each category folder
DISPLAY_NAMES = {
    "bicycle_final":      "Bicycle",
    "cabinet_final":      "Cabinet",
    "chair_final":        "Chair",
    "coffee_maker_final": "Coffee Maker",
    "fan_final":          "Fan",
    "kettle_final":       "Kettle",
    "lamp_final":         "Lamp",
    "mug_final":          "Mug",
    "sofa_final":         "Sofa",
    "stapler_final":      "Stapler",
    "table_final":        "Table",
    "toaster_final":      "Toaster",
}
# ──────────────────────────────────────────────────────────────────────────────


def collect_existing_images() -> dict[str, Path]:
    """
    Walk DATA_DIR and collect all image files that actually exist
    in our subset. Returns dict: filename → full Path.

    This is used to filter Ebay txt rows — we only insert metadata
    for images we actually have on disk.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Data folder not found: {DATA_DIR}\n"
            "Run download_data.py and clean_data.py first.\n"
        )

    existing = {}
    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    for cat_dir in sorted(DATA_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for img_path in cat_dir.iterdir():
            if img_path.suffix in extensions:
                existing[img_path.name] = img_path

    print(f"  Found {len(existing)} images on disk across {len(list(DATA_DIR.iterdir()))} folders")
    return existing


def parse_ebay_txt(
    txt_path: Path,
    split: str,
    existing_images: dict[str, Path],
) -> list[dict]:
    """
    Parse one Ebay txt file (train or test).

    File format (tab or space separated):
      image_id  class_id  super_class_id  image_path
      1         1         1               bicycle_final/111085122871_0.JPG

    Filters to only rows where the image exists in our subset.

    Returns list of dicts ready for DB insertion.
    """
    if not txt_path.exists():
        print(f"  [WARN] {txt_path.name} not found — skipping {split} split")
        return []

    records = []
    skipped = 0

    with open(txt_path, "r") as f:
        next(f)  # skip header line: "image_id class_id super_class_id path"

        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            image_id       = int(parts[0])
            class_id       = int(parts[1])
            super_class_id = int(parts[2])
            img_path_str   = parts[3]   # e.g. bicycle_final/111085122871_0.JPG

            # Extract filename and category from path
            img_path  = Path(img_path_str)
            filename  = img_path.name           # 111085122871_0.JPG
            cat_name  = img_path.parent.name    # bicycle_final

            # Only include images that exist in our subset
            if filename not in existing_images:
                skipped += 1
                continue

            records.append({
                "image_id":       image_id,
                "class_id":       class_id,
                "super_class_id": super_class_id,
                "filename":       filename,
                "category":       cat_name,
                "image_path":     str(existing_images[filename]).replace("\\", "/"),
                "split":          split,
            })

    print(f"  {txt_path.name}: {len(records)} matched, {skipped} skipped")
    return records


def insert_super_categories(db, all_records: list[dict]) -> dict[str, int]:
    """
    Insert unique super categories into super_categories table.
    Returns dict: category_name → super_category.id
    """
    unique_cats = sorted(set(r["category"] for r in all_records))
    cat_id_map  = {}

    print(f"\n[INSERT] super_categories ({len(unique_cats)} rows) ...")

    for cat_name in unique_cats:
        # Check if already exists
        existing = queries.get_super_category_by_name(db, cat_name)
        if existing:
            cat_id_map[cat_name] = existing.id
            continue

        display = DISPLAY_NAMES.get(cat_name, cat_name.replace("_final", "").title())
        super_cat = SuperCategory(
            name=cat_name,
            display_name=display,
            total_images=0,     # updated later
        )
        db.add(super_cat)
        db.flush()              # flush to get the auto-generated id
        cat_id_map[cat_name] = super_cat.id
        print(f"  Inserted: {cat_name} (id={super_cat.id})")

    db.commit()
    print(f"  ✅ {len(cat_id_map)} super categories ready")
    return cat_id_map


def insert_categories(
    db,
    all_records: list[dict],
    cat_id_map: dict[str, int],
) -> dict[int, int]:
    """
    Insert unique class_ids into categories table.
    Returns dict: class_id → category.id
    """
    # Collect unique (class_id, super_class_id, category_name) combos
    seen = {}
    for r in all_records:
        if r["class_id"] not in seen:
            seen[r["class_id"]] = {
                "class_id":         r["class_id"],
                "super_class_id":   r["super_class_id"],
                "category_name":    r["category"],
            }

    class_id_map = {}
    print(f"\n[INSERT] categories ({len(seen)} unique classes) ...")

    for class_id, info in tqdm(seen.items(), desc="  Inserting classes", unit="class"):
        # Check if already exists
        existing = queries.get_category_by_class_id(db, class_id)
        if existing:
            class_id_map[class_id] = existing.id
            continue

        cat = Category(
            super_category_id=cat_id_map[info["category_name"]],
            class_id=class_id,
            total_images=0,   # updated later
        )
        db.add(cat)
        db.flush()
        class_id_map[class_id] = cat.id

    db.commit()
    print(f"  ✅ {len(class_id_map)} categories ready")
    return class_id_map


def insert_products(
    db,
    all_records: list[dict],
    class_id_map: dict[int, int],
) -> int:
    """
    Insert all product image records into products table.
    Uses batch inserts for performance.
    Returns total number of rows inserted.
    """
    print(f"\n[INSERT] products ({len(all_records)} rows) ...")

    # Check for existing records to avoid duplicates on re-run
    existing_filenames = set(
        row[0] for row in db.query(Product.filename).all()
    )

    new_records  = []
    skipped      = 0

    for r in all_records:
        if r["filename"] in existing_filenames:
            skipped += 1
            continue

        new_records.append(Product(
            category_id=class_id_map[r["class_id"]],
            image_id=r["image_id"],
            class_id=r["class_id"],
            super_class_id=r["super_class_id"],
            image_path=r["image_path"],
            filename=r["filename"],
            split=r["split"],
        ))

    if skipped:
        print(f"  Skipped {skipped} already-existing records")

    # Batch insert in chunks of 500 for performance
    BATCH_SIZE = 500
    inserted   = 0

    for i in tqdm(
        range(0, len(new_records), BATCH_SIZE),
        desc="  Inserting products",
        unit="batch",
    ):
        batch = new_records[i : i + BATCH_SIZE]
        db.add_all(batch)
        db.commit()
        inserted += len(batch)

    print(f"  ✅ {inserted} products inserted")
    return inserted


def update_image_counts(db) -> None:
    """
    Update total_images counts in super_categories and categories tables.
    Run after all products are inserted.
    """
    print("\n[UPDATE] Updating image counts ...")

    # Update categories.total_images
    categories = db.query(Category).all()
    for cat in categories:
        cat.total_images = (
            db.query(Product)
            .filter(Product.category_id == cat.id)
            .count()
        )
    db.commit()

    # Update super_categories.total_images
    super_cats = db.query(SuperCategory).all()
    for sc in super_cats:
        sc.total_images = (
            db.query(Product)
            .join(Category)
            .filter(Category.super_category_id == sc.id)
            .count()
        )
    db.commit()
    print("  ✅ Image counts updated")


def save_report(stats: dict) -> None:
    """Save metadata build report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[REPORT] Saved to {REPORT_PATH}")


def print_summary(stats: dict) -> None:
    """Print clean summary to console."""
    print("\n" + "=" * 50)
    print("         METADATA BUILD SUMMARY")
    print("=" * 50)
    print(f"  Super categories  : {stats['super_categories']}")
    print(f"  Fine-grained classes: {stats['categories']}")
    print(f"  Total products    : {stats['total_products']}")
    print(f"  Train images      : {stats['train_images']}")
    print(f"  Test images       : {stats['test_images']}")
    print(f"  Build time        : {stats['build_time_seconds']:.1f}s")
    print("-" * 50)
    print("  Per category:")
    print(f"  {'Category':<25} {'Images':>8}")
    print(f"  {'-'*25} {'-'*8}")
    for cat, count in stats["per_category"].items():
        print(f"  {cat:<25} {count:>8}")
    print("=" * 50)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("    STEP 5 — BUILD MYSQL METADATA")
    print("=" * 50)

    t0 = time.time()

    try:
        # 1. Test MySQL connection
        print("\n[DB] Testing MySQL connection ...")
        test_connection()

        # 2. Create tables if not exist
        create_tables()

        # 3. Collect images that exist on disk
        print("\n[SCAN] Scanning image files ...")
        existing_images = collect_existing_images()

        # 4. Parse both Ebay txt files
        print("\n[PARSE] Parsing Ebay metadata files ...")
        train_records = parse_ebay_txt(TRAIN_TXT, "train", existing_images)
        test_records  = parse_ebay_txt(TEST_TXT,  "test",  existing_images)
        all_records   = train_records + test_records
        print(f"  Total matched records: {len(all_records)}")

        if not all_records:
            raise RuntimeError(
                "[ERROR] No matching records found.\n"
                "Make sure Ebay_train.txt and Ebay_test.txt exist in:\n"
                f"{DATA_DIR}"
            )

        # 5. Insert into MySQL
        db = SessionLocal()
        try:
            cat_id_map   = insert_super_categories(db, all_records)
            class_id_map = insert_categories(db, all_records, cat_id_map)
            inserted     = insert_products(db, all_records, class_id_map)
            update_image_counts(db)

            # 6. Collect stats for report
            db_stats = queries.get_database_stats(db)
            cat_counts = queries.get_category_image_counts(db)

        finally:
            db.close()

        # 7. Build and save report
        elapsed = time.time() - t0
        stats = {
            "super_categories": db_stats["total_super_categories"],
            "categories":       db_stats["total_categories"],
            "total_products":   db_stats["total_products"],
            "train_images":     db_stats["total_train_images"],
            "test_images":      db_stats["total_test_images"],
            "build_time_seconds": round(elapsed, 2),
            "per_category": {
                c["category"]: c["image_count"] for c in cat_counts
            },
        }

        save_report(stats)
        print_summary(stats)

        print("\n[DONE] Step 5 complete. Run app.py next.\n")

    except FileNotFoundError as e:
        print(e)
    except RuntimeError as e:
        print(e)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Metadata build cancelled.")