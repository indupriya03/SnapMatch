"""
pipeline.py
-----------
Single entry point for the deep-visual-retrieval pipeline.

Runs all steps in order:
  Step 1 — download_data.py      Download + subset selection
  Step 2 — clean_data.py         Data cleaning (in place)
  Step 3 — embeddings.py ResNet50 embedding extraction
  Step 4 — build_index.py        FAISS index construction
  Step 5 — build_metadata.py     Populate MySQL
  Step 6 — evaluate.py           Precision@K + Recall@K

Each step:
  - Checks if its outputs already exist (skip if so)
  - Logs start time and duration
  - Stops the pipeline if it fails

Usage:
  # Run full pipeline from scratch
  python pipeline.py

  # Run from a specific step (skips earlier steps)
  python pipeline.py --from-step 3

  # Run only a specific step
  python pipeline.py --step 4

  # Skip evaluation (faster)
  python pipeline.py --skip-eval

  # Force re-run all steps even if outputs exist
  python pipeline.py --force

Requirements:
  - .env file configured with MySQL credentials
  - data/raw_download/archive.zip must exist before running
  - pip install all packages from requirements.txt
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

STEPS = [
    {
        "number":       1,
        "name":         "Data Download & Subset Selection",
        "script":       BASE_DIR / "src" / "download_data.py",
        "check_output": BASE_DIR / "data" / "my_project_data",
        "check_type":   "dir",
        "description":  "Unzip + sample 800 images per category",
    },
    {
        "number":       2,
        "name":         "Data Cleaning — Pass 1 (Structural)",
        "script":       BASE_DIR / "src" / "clean_data.py",
        "script_args":  ["--pass", "1"],          # ← add args support
        "check_output": BASE_DIR / "outputs" / "cleaning_report.json",
        "check_type":   "file",
        "description":  "Corrupt · size · RGB · blank · duplicate checks",
    },
    {
        "number":       3,
        "name":         "Embedding Extraction — Pass 1",
        "script":       BASE_DIR / "src" / "embeddings.py",
        "script_args":  ["--pass", "1"],
        "check_output": BASE_DIR / "outputs" / "embeddings.npy",
        "check_type":   "file",
        "description":  "ResNet50 → embeddings.npy for noise detection",
    },
    {
        "number":       4,
        "name":         "Data Cleaning — Pass 2 (Noise Detection)",
        "script":       BASE_DIR / "src" / "clean_data.py",
        "script_args":  ["--pass", "2"],          # ← same script, different pass
        "check_output": BASE_DIR / "outputs" / "noise_report.json",
        "check_type":   "file",
        "description":  "Outlier detection using category centroids",
    },
    {
        "number":       5,
        "name":         "Embedding Extraction — Pass 2 (Final)",
        "script":       BASE_DIR / "src" / "embeddings.py",
        "script_args":  ["--pass", "2"],
        "check_output": BASE_DIR / "outputs" / "embeddings_final.npy",
        "check_type":   "file",
        "description":  "Re-extract embeddings on noise-free images",
    },
    {
        "number":       6,
        "name":         "FAISS Index Build",
        "script":       BASE_DIR / "src" / "build_index.py",
        "script_args":  ["--embeddings", "outputs/embeddings_final.npy"],
        "check_output": BASE_DIR / "outputs" / "faiss_index.bin",
        "check_type":   "file",
        "description":  "L2 normalise + IndexFlatIP cosine similarity",
    },
    {
        "number":       7,
        "name":         "MySQL Metadata Population",
        "script":       BASE_DIR / "src" / "build_metadata.py",
        "check_output": BASE_DIR / "outputs" / "metadata_report.json",
        "check_type":   "file",
        "description":  "Parse Ebay txt → populate MySQL",
    },
    {
        "number":       8,
        "name":         "Evaluation",
        "script":       BASE_DIR / "src" / "evaluate.py",
        "check_output": BASE_DIR / "outputs" / "evaluation_report_relaxed.json",
        "check_type":   "file",
        "description":  "Precision@K · Recall@K · search time",
    },
]
# ──────────────────────────────────────────────────────────────────────────────


def separator(char="─", width=58):
    print(char * width)


def output_exists(step: dict) -> bool:
    """Check if a step's output already exists on disk."""
    path = step["check_output"]
    if step["check_type"] == "dir":
        return Path(path).exists() and any(Path(path).iterdir())
    return Path(path).exists()


def run_step(step: dict, force: bool = False) -> bool:
    """
    Run a single pipeline step.

    Returns True if step succeeded or was skipped.
    Returns False if step failed.
    """
    num    = step["number"]
    name   = step["name"]
    script = step["script"]

    print(f"\n{'='*58}")
    print(f"  STEP {num} — {name}")
    print(f"  {step['description']}")
    print(f"{'='*58}")

    # Check if output already exists
    if not force and output_exists(step):
        print(f"  [SKIP] Output already exists — skipping step {num}")
        print(f"         Use --force to re-run this step")
        return True

    # Check script exists
    if not script.exists():
        print(f"  [ERROR] Script not found: {script}")
        return False

    # Run the script
    print(f"  [RUN] python {script.name}")
    t0 = time.time()

    # Build command — append script_args if defined for this step
    script_args = step.get("script_args", [])
    cmd = [sys.executable, str(script)] + script_args

    result = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  [FAILED] Step {num} failed after {elapsed:.1f}s")
        print(f"  Check the error output above and fix before re-running.")
        return False

    print(f"\n  [DONE] Step {num} completed in {elapsed:.1f}s")
    return True


def print_pipeline_header(steps_to_run: list[dict]) -> None:
    """Print the pipeline plan before running."""
    separator("═")
    print("   DEEP VISUAL RETRIEVAL — PIPELINE RUNNER")
    separator("═")
    print()
    print("  Steps to run:")
    for step in steps_to_run:
        exists = output_exists(step)
        status = "✅ done" if exists else "⏳ pending"
        print(f"    {step['number']}. {step['name']:<35} {status}")
    print()
    separator()


def print_pipeline_summary(
    results: list[tuple[int, str, bool]],
    total_time: float,
) -> None:
    """Print final summary after all steps complete."""
    print(f"\n{'='*58}")
    print("   PIPELINE SUMMARY")
    print(f"{'='*58}")

    all_ok = True
    for num, name, ok in results:
        icon = "✅" if ok else "❌"
        print(f"  {icon} Step {num}: {name}")
        if not ok:
            all_ok = False

    separator()
    print(f"  Total time : {total_time/60:.1f} minutes")

    if all_ok:
        print(f"\n  ✅ All steps complete!")
        print(f"\n  Start the API server:")
        print(f"    uvicorn app:app --reload --host 0.0.0.0 --port 8000")
        print(f"\n  Then open: http://localhost:8000")
    else:
        print(f"\n  ❌ Pipeline failed. Fix errors above and re-run.")
        print(f"     Tip: use --from-step N to resume from the failed step")

    print(f"{'='*58}\n")


def check_prerequisites() -> bool:
    """
    Check that required files exist before starting the pipeline.
    Returns True if all prerequisites are met.
    """
    print("\n[CHECK] Verifying prerequisites ...")
    ok = True

    # Check .env exists
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        print("  [ERROR] .env file not found.")
        print("          Copy .env.example to .env and fill in your MySQL password.")
        ok = False
    else:
        print("  ✅ .env found")

    # Check archive.zip exists (needed for step 1)
    zip_path = BASE_DIR / "data" / "raw_download" / "archive.zip"
    data_dir = BASE_DIR / "data" / "my_project_data"
    if not zip_path.exists() and not data_dir.exists():
        print("  [ERROR] archive.zip not found at data/raw_download/archive.zip")
        print("          Download from Kaggle and place it there before running.")
        ok = False
    elif zip_path.exists():
        print(f"  ✅ archive.zip found ({zip_path.stat().st_size / 1e9:.2f} GB)")
    else:
        print("  ✅ my_project_data already exists — step 1 will be skipped")

    # Check db/__init__.py exists
    init_path = BASE_DIR / "db" / "__init__.py"
    if not init_path.exists():
        print("  [WARN] db/__init__.py not found — creating it now ...")
        init_path.touch()
        print("  ✅ db/__init__.py created")
    else:
        print("  ✅ db/__init__.py found")

    # Check outputs/ exists
    outputs_dir = BASE_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    print("  ✅ outputs/ directory ready")

    return ok


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep Visual Retrieval — Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                  Run full pipeline
  python pipeline.py --from-step 3   Resume from step 3
  python pipeline.py --step 4        Run only step 4
  python pipeline.py --skip-eval     Skip evaluation (step 6)
  python pipeline.py --force         Force re-run all steps
        """
    )
    parser.add_argument(
        "--from-step", type=int, default=1,
        metavar="N",
        help="Start pipeline from step N (default: 1)"
    )
    parser.add_argument(
        "--step", type=int, default=None,
        metavar="N",
        help="Run only step N"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip step 6 (evaluation)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run steps even if outputs already exist"
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # Determine which steps to run
    if args.step is not None:
        # Run only a specific step
        steps_to_run = [s for s in STEPS if s["number"] == args.step]
        if not steps_to_run:
            print(f"[ERROR] Invalid step number: {args.step}. Must be 1-6.")
            sys.exit(1)
    else:
        # Run from --from-step onwards
        steps_to_run = [s for s in STEPS if s["number"] >= args.from_step]

    # Optionally skip evaluation
    if args.skip_eval:
        steps_to_run = [s for s in steps_to_run if s["number"] != 6]

    if not steps_to_run:
        print("[ERROR] No steps to run.")
        sys.exit(1)

    # Print plan
    print_pipeline_header(steps_to_run)

    # Check prerequisites (only if step 1 is included)
    if any(s["number"] <= 2 for s in steps_to_run):
        if not check_prerequisites():
            print("\n[ABORT] Fix prerequisites before running the pipeline.\n")
            sys.exit(1)

    # Run steps
    pipeline_start = time.time()
    results        = []

    for step in steps_to_run:
        ok = run_step(step, force=args.force)
        results.append((step["number"], step["name"], ok))
        if not ok:
            # Stop pipeline on failure
            remaining = [
                s for s in steps_to_run
                if s["number"] > step["number"]
            ]
            for s in remaining:
                results.append((s["number"], s["name"], False))
            break

    total_time = time.time() - pipeline_start
    print_pipeline_summary(results, total_time)

    # Exit with error code if any step failed
    if not all(ok for _, _, ok in results):
        sys.exit(1)