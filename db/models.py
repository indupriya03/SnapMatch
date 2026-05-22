"""
db/models.py
------------
SQLAlchemy ORM table definitions for the deep-visual-retrieval project.

Tables:
  1. super_categories  — 12 top-level product types (bicycle, mug, chair...)
  2. categories        — fine-grained product classes within each super category
  3. products          — one row per image, linked to its category

Relationships:
  SuperCategory  →  Category  (one to many)
  Category       →  Product   (one to many)

Usage:
  from db.models import Base, SuperCategory, Category, Product
  Base.metadata.create_all(engine)   ← creates all tables in MySQL
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    Enum,
    Index,
    DateTime,
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

# Base class for all models
Base = declarative_base()


class SuperCategory(Base):
    """
    Top-level product category.

    Examples:
      id=1  name='bicycle_final'
      id=2  name='cabinet_final'
      ...
      id=12 name='toaster_final'

    One SuperCategory has many Categories (fine-grained classes).
    """
    __tablename__ = "super_categories"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    name            = Column(String(100), nullable=False, unique=True)
    display_name    = Column(String(100), nullable=False)   # e.g. "Bicycle"
    total_images    = Column(Integer, default=0)            # updated after insert
    created_at      = Column(DateTime, default=datetime.utcnow)

    # Relationship — one super_category has many categories
    categories = relationship(
        "Category",
        back_populates="super_category",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<SuperCategory id={self.id} name={self.name}>"


class Category(Base):
    """
    Fine-grained product class within a super category.
    Maps directly to class_id in Ebay_train.txt / Ebay_test.txt.

    Example:
      id=1  super_category_id=1  class_id=1   (specific bicycle model)
      id=2  super_category_id=1  class_id=2   (another bicycle model)

    One Category has many Products (images of that specific product model).
    """
    __tablename__ = "categories"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    super_category_id   = Column(
                            Integer,
                            ForeignKey("super_categories.id", ondelete="CASCADE"),
                            nullable=False,
                          )
    class_id            = Column(Integer, nullable=False, unique=True)  # from Ebay txt
    total_images        = Column(Integer, default=0)
    created_at          = Column(DateTime, default=datetime.utcnow)

    # Relationships
    super_category  = relationship("SuperCategory", back_populates="categories")
    products        = relationship(
                        "Product",
                        back_populates="category",
                        cascade="all, delete-orphan",
                      )

    # Index on class_id for fast lookups
    __table_args__ = (
        Index("idx_class_id", "class_id"),
        Index("idx_super_category_id", "super_category_id"),
    )

    def __repr__(self):
        return f"<Category id={self.id} class_id={self.class_id}>"


class Product(Base):
    """
    One row per product image.

    Stores file path, filename, split (train/test), and links to
    its fine-grained category (class_id) and super category.

    image_id    — original ID from Ebay_train.txt / Ebay_test.txt
    class_id    — fine-grained class (links to categories.class_id)
    image_path  — relative path e.g. data/my_project_data/mug_final/img001.JPG
    filename    — just the filename e.g. img001.JPG
    split       — 'train' or 'test' (from which Ebay txt file it came)
    """
    __tablename__ = "products"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    category_id     = Column(
                        Integer,
                        ForeignKey("categories.id", ondelete="CASCADE"),
                        nullable=False,
                      )
    image_id        = Column(Integer, nullable=False)
    class_id        = Column(Integer, nullable=False)
    super_class_id  = Column(Integer, nullable=False)
    image_path      = Column(String(500), nullable=False)
    filename        = Column(String(200), nullable=False)
    split           = Column(
                        Enum("train", "test", name="split_enum"),
                        nullable=False,
                      )
    created_at      = Column(DateTime, default=datetime.utcnow)

    # Relationship
    category = relationship("Category", back_populates="products")

    # Indexes for fast queries
    __table_args__ = (
        Index("idx_filename",       "filename"),
        Index("idx_class_id_prod",  "class_id"),
        Index("idx_split",          "split"),
        Index("idx_category_id",    "category_id"),
    )

    def __repr__(self):
        return (
            f"<Product id={self.id} "
            f"filename={self.filename} "
            f"class_id={self.class_id} "
            f"split={self.split}>"
        )