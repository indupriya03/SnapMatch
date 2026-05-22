"""
db/queries.py
-------------
All database query functions for the deep-visual-retrieval project.

This is the ONLY file that writes SQL / ORM queries.
app.py calls these functions — it never writes queries directly.

Functions:
  Product queries:
    get_products_by_filenames()     ← core search query (used by /search)
    get_product_by_filename()       ← single product lookup
    get_products_by_category()      ← filter by category
    get_products_by_class_id()      ← filter by fine-grained class

  Category queries:
    get_all_super_categories()      ← used by /categories endpoint
    get_super_category_by_name()    ← lookup by folder name
    get_categories_by_super()       ← all classes within a super category

  Stats queries:
    get_database_stats()            ← used by /health endpoint
    get_category_image_counts()     ← image count per category

Usage:
  from db import queries
  results = queries.get_products_by_filenames(db, filenames)
"""

from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func
from db.models import SuperCategory, Category, Product


# ── Product Queries ────────────────────────────────────────────────────────────

def get_products_by_filenames(
    db: Session,
    filenames: list[str],
) -> dict[str, dict]:
    """
    Core search query — called by POST /search after FAISS returns results.

    Given a list of filenames (from image_paths.json via FAISS indices),
    returns a dict mapping filename → product metadata.

    Why a dict return?
      FAISS returns results in ranked order. We need to preserve that rank
      while enriching each result with metadata. A dict keyed by filename
      allows O(1) lookup when building the final response.

    Args:
      db        : SQLAlchemy session
      filenames : list of image filenames e.g. ['img001.JPG', 'img045.JPG']

    Returns:
      {
        'img001.JPG': {
          'class_id': 47,
          'super_class_id': 3,
          'category': 'mug_final',
          'display_name': 'Mug',
          'split': 'train',
          'image_path': 'data/my_project_data/mug_final/img001.JPG',
        },
        ...
      }
    """
    if not filenames:
        return {}

    # Join products → categories → super_categories in one query
    rows = (
        db.query(
            Product.filename,
            Product.class_id,
            Product.super_class_id,
            Product.image_path,
            Product.split,
            SuperCategory.name.label("category"),
            SuperCategory.display_name,
        )
        .join(Category, Product.category_id == Category.id)
        .join(SuperCategory, Category.super_category_id == SuperCategory.id)
        .filter(Product.filename.in_(filenames))
        .all()
    )

    # Build filename → metadata dict
    metadata = {}
    for row in rows:
        metadata[row.filename] = {
            "class_id":     row.class_id,
            "super_class_id": row.super_class_id,
            "category":     row.category,
            "display_name": row.display_name,
            "split":        row.split,
            "image_path":   row.image_path,
        }

    return metadata


def get_product_by_filename(
    db: Session,
    filename: str,
) -> dict | None:
    """
    Lookup a single product by filename.
    Returns None if not found.
    """
    row = (
        db.query(
            Product.filename,
            Product.class_id,
            Product.super_class_id,
            Product.image_path,
            Product.split,
            SuperCategory.name.label("category"),
            SuperCategory.display_name,
        )
        .join(Category, Product.category_id == Category.id)
        .join(SuperCategory, Category.super_category_id == SuperCategory.id)
        .filter(Product.filename == filename)
        .first()
    )

    if not row:
        return None

    return {
        "filename":       row.filename,
        "class_id":       row.class_id,
        "super_class_id": row.super_class_id,
        "category":       row.category,
        "display_name":   row.display_name,
        "split":          row.split,
        "image_path":     row.image_path,
    }


def get_products_by_category(
    db: Session,
    category_name: str,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """
    Get all products in a given super category.
    Supports pagination via limit/offset.
    Used for category browsing and optional frontend filtering.
    """
    rows = (
        db.query(
            Product.filename,
            Product.class_id,
            Product.image_path,
            Product.split,
            SuperCategory.name.label("category"),
            SuperCategory.display_name,
        )
        .join(Category, Product.category_id == Category.id)
        .join(SuperCategory, Category.super_category_id == SuperCategory.id)
        .filter(SuperCategory.name == category_name)
        .limit(limit)
        .offset(offset)
        .all()
    )

    return [
        {
            "filename":     row.filename,
            "class_id":     row.class_id,
            "image_path":   row.image_path,
            "split":        row.split,
            "category":     row.category,
            "display_name": row.display_name,
        }
        for row in rows
    ]


def get_products_by_class_id(
    db: Session,
    class_id: int,
) -> list[dict]:
    """
    Get all products belonging to a specific fine-grained class.
    Used for Precision@K evaluation — checking if results share
    the same class_id as the query image.
    """
    rows = (
        db.query(
            Product.filename,
            Product.class_id,
            Product.image_path,
            Product.split,
        )
        .filter(Product.class_id == class_id)
        .all()
    )

    return [
        {
            "filename":   row.filename,
            "class_id":   row.class_id,
            "image_path": row.image_path,
            "split":      row.split,
        }
        for row in rows
    ]


# ── Category Queries ───────────────────────────────────────────────────────────

def get_all_super_categories(db: Session) -> list[dict]:
    """
    Return all 12 super categories with image counts.
    Used by GET /categories endpoint and frontend filter dropdown.
    """
    rows = (
        db.query(
            SuperCategory.id,
            SuperCategory.name,
            SuperCategory.display_name,
            SuperCategory.total_images,
        )
        .order_by(SuperCategory.display_name)
        .all()
    )

    return [
        {
            "id":           row.id,
            "name":         row.name,
            "display_name": row.display_name,
            "total_images": row.total_images,
        }
        for row in rows
    ]


def get_super_category_by_name(
    db: Session,
    name: str,
) -> SuperCategory | None:
    """
    Lookup a super category by its folder name.
    e.g. get_super_category_by_name(db, 'mug_final')
    Used by build_metadata.py during population.
    """
    return (
        db.query(SuperCategory)
        .filter(SuperCategory.name == name)
        .first()
    )


def get_categories_by_super(
    db: Session,
    super_category_id: int,
) -> list[dict]:
    """
    Get all fine-grained classes within a super category.
    e.g. all specific mug models within 'mug_final'.
    """
    rows = (
        db.query(
            Category.id,
            Category.class_id,
            Category.total_images,
        )
        .filter(Category.super_category_id == super_category_id)
        .order_by(Category.class_id)
        .all()
    )

    return [
        {
            "id":           row.id,
            "class_id":     row.class_id,
            "total_images": row.total_images,
        }
        for row in rows
    ]


def get_category_by_class_id(
    db: Session,
    class_id: int,
) -> Category | None:
    """
    Lookup a Category row by its class_id.
    Used by build_metadata.py during population.
    """
    return (
        db.query(Category)
        .filter(Category.class_id == class_id)
        .first()
    )


# ── Stats Queries ──────────────────────────────────────────────────────────────

def get_database_stats(db: Session) -> dict:
    """
    Return row counts for all three tables.
    Used by GET /health endpoint.
    """
    return {
        "total_super_categories": db.query(
            func.count(SuperCategory.id)
        ).scalar(),

        "total_categories": db.query(
            func.count(Category.id)
        ).scalar(),

        "total_products": db.query(
            func.count(Product.id)
        ).scalar(),

        "total_train_images": db.query(
            func.count(Product.id)
        ).filter(Product.split == "train").scalar(),

        "total_test_images": db.query(
            func.count(Product.id)
        ).filter(Product.split == "test").scalar(),
    }


def get_category_image_counts(db: Session) -> list[dict]:
    """
    Return image count per super category.
    Useful for dashboard and README stats.
    """
    rows = (
        db.query(
            SuperCategory.display_name,
            SuperCategory.name,
            func.count(Product.id).label("image_count"),
        )
        .join(Category, Category.super_category_id == SuperCategory.id)
        .join(Product, Product.category_id == Category.id)
        .group_by(SuperCategory.id, SuperCategory.display_name, SuperCategory.name)
        .order_by(SuperCategory.display_name)
        .all()
    )

    return [
        {
            "category":     row.name,
            "display_name": row.display_name,
            "image_count":  row.image_count,
        }
        for row in rows
    ]


def get_similar_class_filenames(
    db: Session,
    class_id: int,
) -> set[str]:
    """
    Return a set of all filenames that share the same class_id.
    Used for Precision@K and Recall@K evaluation.

    Precision@K = how many of Top-K results have the same class_id
                  as the query image.
    """
    rows = (
        db.query(Product.filename)
        .filter(Product.class_id == class_id)
        .all()
    )
    return {row.filename for row in rows}

def get_similar_super_class_filenames(
    db: Session,
    class_id: int,
) -> set[str]:
    """
    Return all filenames sharing the same super_class_id as this class_id.
    Used for category-level Precision@K evaluation.
    """
    # First get the super_class_id for this class_id
    row = (
        db.query(Product.super_class_id)
        .filter(Product.class_id == class_id)
        .first()
    )
    if not row:
        return set()

    super_class_id = row.super_class_id

    # Return all filenames with same super_class_id
    rows = (
        db.query(Product.filename)
        .filter(Product.super_class_id == super_class_id)
        .all()
    )
    return {r.filename for r in rows}