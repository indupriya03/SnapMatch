"""
db/database.py
--------------
MySQL database connection and session management.

What this file does:
  1. Reads database credentials from .env file
  2. Creates a SQLAlchemy engine (connection pool to MySQL)
  3. Creates a SessionLocal factory for database sessions
  4. Provides get_db() dependency for FastAPI route injection
  5. Provides create_tables() to initialise schema on first run
  6. Provides test_connection() to verify MySQL is reachable

Usage in FastAPI routes:
  from db.database import get_db
  from sqlalchemy.orm import Session

  @app.get("/categories")
  def get_categories(db: Session = Depends(get_db)):
      return queries.get_all_categories(db)

Usage for table creation (run once):
  from db.database import create_tables
  create_tables()

Requirements:
  pip install sqlalchemy pymysql python-dotenv

.env file (project root):
  DB_HOST=localhost
  DB_PORT=3306
  DB_NAME=deep_visual_retrieval
  DB_USER=root
  DB_PASSWORD=your_password
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError

# ── Load environment variables from .env file ─────────────────────────────────
# Look for .env in project root (one level up from db/)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
# ──────────────────────────────────────────────────────────────────────────────


def get_db_url() -> str:
    """
    Build the MySQL connection URL from environment variables.

    Format:
      mysql+pymysql://user:password@host:port/database

    pymysql is a pure Python MySQL driver — no C dependencies,
    works on all platforms including Windows without extra setup.
    """
    host     = os.getenv("DB_HOST",     "localhost")
    port     = os.getenv("DB_PORT",     "3306")
    name     = os.getenv("DB_NAME",     "deep_visual_retrieval")
    user     = os.getenv("DB_USER",     "root")
    password = os.getenv("DB_PASSWORD", "")

    # Validate required fields
    if not password:
        raise ValueError(
            "\n[ERROR] DB_PASSWORD not set in .env file.\n"
            f"  Expected .env at: {ENV_PATH}\n"
            "  Add: DB_PASSWORD=your_mysql_password\n"
        )

    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}"


# ── Engine ────────────────────────────────────────────────────────────────────
# Created once at module import time
# pool_pre_ping=True — tests connection before use (handles stale connections)
# pool_size=5        — maintain 5 connections in the pool
# max_overflow=10    — allow up to 10 extra connections under load
engine = create_engine(
    get_db_url(),
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,         # set True to log all SQL queries (useful for debugging)
)
# ──────────────────────────────────────────────────────────────────────────────


# ── Session factory ───────────────────────────────────────────────────────────
# autocommit=False — changes must be explicitly committed
# autoflush=False  — changes not auto-flushed before queries
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)
# ──────────────────────────────────────────────────────────────────────────────


def get_db():
    """
    FastAPI dependency that provides a database session per request.

    Usage:
      @app.get("/something")
      def route(db: Session = Depends(get_db)):
          results = queries.some_query(db)
          return results

    How it works:
      - Opens a new session at the start of each request
      - Yields it to the route handler
      - Closes it automatically after the response is sent
        (even if an exception occurs — the finally block guarantees cleanup)

    This pattern ensures:
      - No connection leaks
      - Each request gets its own isolated session
      - Transactions are properly closed
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """
    Create all tables in MySQL if they don't exist.

    Reads table definitions from db/models.py (via Base.metadata).
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS internally.

    Call this once when setting up the project:
      python -c "from db.database import create_tables; create_tables()"
    Or it is called automatically by build_metadata.py
    """
    from db.models import Base

    print("\n[DB] Creating tables in MySQL ...")
    try:
        Base.metadata.create_all(bind=engine)
        print("  ✅ Tables created (or already exist):")
        print("     super_categories")
        print("     categories")
        print("     products")
    except OperationalError as e:
        raise RuntimeError(
            f"\n[ERROR] Could not create tables: {e}\n"
            "Check your MySQL credentials in .env and ensure MySQL is running.\n"
        )


def test_connection() -> bool:
    """
    Test that MySQL is reachable and credentials are correct.

    Returns True if connection succeeds, raises RuntimeError if not.
    Called at FastAPI startup to catch misconfigurations early.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("  ✅ MySQL connection successful")
        return True
    except OperationalError as e:
        raise RuntimeError(
            f"\n[ERROR] Cannot connect to MySQL: {e}\n"
            "Check:\n"
            "  1. MySQL is running on your laptop\n"
            "  2. DB credentials in .env are correct\n"
            "  3. Database 'deep_visual_retrieval' exists\n"
            "     → Run in MySQL: CREATE DATABASE deep_visual_retrieval;\n"
        )


def get_table_counts() -> dict:
    """
    Return row counts for all three tables.
    Used by /health endpoint to show database status.
    """
    db = SessionLocal()
    try:
        counts = {}
        for table in ["super_categories", "categories", "products"]:
            result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
            counts[table] = result.scalar()
        return counts
    except Exception:
        return {"super_categories": 0, "categories": 0, "products": 0}
    finally:
        db.close()