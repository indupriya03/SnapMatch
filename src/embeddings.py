"""
extract_embeddings.py
---------------------
Step 3 of the deep-visual-retrieval pipeline.

What this script does:
  1. Loads all cleaned images from data/cleaned_data/
  2. Loads pretrained ResNet50 with the final classification layer removed
  3. Passes images through ResNet50 in batches to extract 2048-dim embeddings
  4. Saves embeddings  → outputs/embeddings.npy   shape: (N, 2048)
  5. Saves image paths → outputs/image_paths.json  list of N file paths
  6. Saves a report   → outputs/embedding_report.json

Why we remove the final layer:
  ResNet50's last layer maps 2048 features → 1000 ImageNet class labels.
  We don't want class predictions — we want the rich 2048-dim visual
  feature vector that sits just before that layer. That vector is the
  embedding that captures colour, texture, shape, and structure.

Usage:
  python src/embeddings.py

Requirements:
  - pip install torch torchvision pillow tqdm numpy
  - GPU is used automatically if available, otherwise falls back to CPU
  - CPU estimated time: 50–70 mins for 9,600 images
  - GPU estimated time:  2–5  mins for 9,600 images
"""

import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ── Device Configuration ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR          = Path("data/cleaned_data")
EMBEDDINGS_PATH   = Path("outputs/embeddings.npy")
IMAGE_PATHS_PATH  = Path("outputs/image_paths.json")
REPORT_PATH       = Path("outputs/embedding_report.json")

BATCH_SIZE        = 64 if torch.cuda.is_available() else 32   # GPU can handle larger batches
IMAGE_SIZE        = 224      # ResNet50 expects 224 × 224
EMBEDDING_DIM     = 2048     # ResNet50 output after removing final layer

# ImageNet mean and std — required for pretrained ResNet50
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]
# ──────────────────────────────────────────────────────────────────────────────


def load_model() -> nn.Module:
    """
    Load pretrained ResNet50 and remove the final classification layer.

    Original ResNet50 architecture (last 2 layers):
        AdaptiveAvgPool2d  →  output: (batch, 2048, 1, 1)
        Linear(2048, 1000) →  output: (batch, 1000)  ← we remove this

    After modification:
        AdaptiveAvgPool2d  →  output: (batch, 2048, 1, 1)
        Flatten            →  output: (batch, 2048)   ← this is our embedding
    """
    print("\n[MODEL] Loading pretrained ResNet50 ...")

    # Load ResNet50 with ImageNet pretrained weights
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model   = models.resnet50(weights=weights)

    # Remove the final fully connected (classification) layer
    # Replace with Identity so the 2048-dim vector flows through unchanged
    model.fc = nn.Identity()

    # Set to evaluation mode — disables dropout and batchnorm training behaviour
    model.eval()

    # ── Move model to GPU if available ────────────────
    model = model.to(device)

    print(f"  Architecture : ResNet50 (pretrained on ImageNet)")
    print(f"  Final layer  : removed (fc replaced with Identity)")
    print(f"  Embedding dim: {EMBEDDING_DIM}")
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  Note         : No GPU found — running on CPU (slower but fully supported)")
    print(f"  Mode         : eval (no gradient computation)")

    return model


def build_transform() -> transforms.Compose:
    """
    Build the image preprocessing pipeline required by ResNet50.

    Steps:
      1. Resize shortest edge to 256px  (preserves aspect ratio)
      2. CenterCrop to 224 × 224        (standard ResNet50 input size)
      3. Convert to tensor              (pixel values 0.0 → 1.0)
      4. Normalize with ImageNet stats  (required for pretrained weights)
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def collect_image_paths() -> list[Path]:
    """
    Walk DATA_DIR and collect all image file paths across all categories.
    Returns a sorted list for reproducibility.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Data folder not found: {DATA_DIR}\n"
            "Run download_data.py and clean_data.py first.\n"
        )

    # ── FIX: use suffix.lower() to avoid Windows case-insensitive
    #         double-counting (.jpg and .JPG are the same file)
    image_paths = []

    for cat_dir in sorted(DATA_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for img_path in sorted(cat_dir.rglob("*")):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"} and img_path.is_file():
                image_paths.append(img_path)

    if not image_paths:
        raise RuntimeError(
            f"[ERROR] No images found in {DATA_DIR}.\n"
            "Make sure clean_data.py completed successfully."
        )

    return image_paths


def load_image(img_path: Path, transform: transforms.Compose) -> torch.Tensor | None:
    """
    Load a single image, apply preprocessing transform.
    Returns None if the image cannot be loaded (safety net).
    """
    try:
        img = Image.open(img_path).convert("RGB")
        return transform(img)          # shape: (3, 224, 224)
    except Exception:
        return None


def extract_embeddings(
    model: nn.Module,
    image_paths: list[Path],
    transform: transforms.Compose,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract embeddings for all images using batch processing.

    Process:
      - Images are loaded and preprocessed in batches of BATCH_SIZE
      - Each batch is passed through ResNet50 in a single forward pass
      - torch.no_grad() prevents gradient computation (saves memory + speed)
      - Embeddings are collected and stacked into a NumPy array

    Returns:
      embeddings   : np.ndarray  shape (N, 2048)  float32
      valid_paths  : list[str]   N file paths matching each embedding row
    """
    all_embeddings = []
    valid_paths    = []
    skipped        = []

    total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time    = time.time()

    estimated = "2–5 minutes" if device.type == "cuda" else "50–70 minutes"
    print(f"\n[EXTRACT] Processing {len(image_paths)} images "
          f"in batches of {BATCH_SIZE} ...")
    print(f"  Estimated time: {estimated} on {device}\n")

    with torch.no_grad():   # no gradients needed — inference only
        for batch_idx in tqdm(
            range(0, len(image_paths), BATCH_SIZE),
            total=total_batches,
            desc="  Extracting",
            unit="batch",
        ):
            batch_paths  = image_paths[batch_idx : batch_idx + BATCH_SIZE]
            batch_tensors = []
            batch_valid_paths = []

            # Load each image in the batch
            for img_path in batch_paths:
                tensor = load_image(img_path, transform)
                if tensor is not None:
                    batch_tensors.append(tensor)
                    batch_valid_paths.append(str(img_path))
                else:
                    skipped.append(str(img_path))

            if not batch_tensors:
                continue

            # Stack into a single batch tensor: (batch_size, 3, 224, 224)
            # ── Move batch to same device as model ────
            batch_input = torch.stack(batch_tensors).to(device)

            # Forward pass through ResNet50
            # Output shape: (batch_size, 2048) after fc=Identity
            batch_embeddings = model(batch_input)

            # Move back to CPU for NumPy conversion
            batch_np = batch_embeddings.cpu().numpy().reshape(batch_embeddings.shape[0], -1)
            # Handle edge case: single image batch gives shape (2048,) not (1, 2048)
            if batch_np.ndim == 1:
                batch_np = batch_np[np.newaxis, :]

            all_embeddings.append(batch_np)
            valid_paths.extend(batch_valid_paths)

    # Pre-allocate output array — avoids RAM spike from np.vstack
    n_images   = sum(e.shape[0] for e in all_embeddings)
    emb_dim    = all_embeddings[0].shape[1]
    embeddings = np.zeros((n_images, emb_dim), dtype=np.float32)
    row = 0
    for batch_emb in all_embeddings:
        embeddings[row: row + batch_emb.shape[0]] = batch_emb
        row += batch_emb.shape[0]
    del all_embeddings   # free batch list immediately

    elapsed = time.time() - start_time
    print(f"\n  Extraction complete in {elapsed/60:.1f} minutes")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Skipped (load errors): {len(skipped)}")

    return embeddings, valid_paths, skipped


def save_outputs(
    embeddings: np.ndarray,
    valid_paths: list[str],
    skipped: list[str],
) -> None:
    """Save embeddings, image paths, and report to outputs/."""
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save embeddings as NumPy binary — fast load, compact size
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"\n[SAVE] embeddings.npy → {EMBEDDINGS_PATH}")
    print(f"       Shape : {embeddings.shape}")
    size_mb = EMBEDDINGS_PATH.stat().st_size / 1_000_000
    print(f"       Size  : {size_mb:.1f} MB")

    # Save image paths as JSON — maps row index → file path
    with open(IMAGE_PATHS_PATH, "w") as f:
        json.dump(valid_paths, f, indent=2)
    print(f"\n[SAVE] image_paths.json → {IMAGE_PATHS_PATH}")
    print(f"       Entries: {len(valid_paths)}")

    # Save report
    report = {
        "total_images_processed": len(valid_paths),
        "skipped_images": len(skipped),
        "skipped_paths": skipped,
        "embedding_shape": list(embeddings.shape),
        "embedding_dim": EMBEDDINGS_PATH.stat().st_size,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "model": "ResNet50 (ImageNet pretrained, fc=Identity)",
        "dtype": str(embeddings.dtype),
        "device": str(device),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVE] embedding_report.json → {REPORT_PATH}")


def print_summary(embeddings: np.ndarray, valid_paths: list[str], skipped: list[str]) -> None:
    """Print a clean final summary."""
    print("\n" + "=" * 50)
    print("        EMBEDDING EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"  Device            : {device}")
    print(f"  Images processed  : {len(valid_paths)}")
    print(f"  Images skipped    : {len(skipped)}")
    print(f"  Embedding shape   : {embeddings.shape}")
    print(f"  Embedding dim     : {embeddings.shape[1]}")
    print(f"  Data type         : {embeddings.dtype}")
    print(f"  Min value         : {embeddings.min():.4f}")
    print(f"  Max value         : {embeddings.max():.4f}")
    print(f"  Mean value        : {embeddings.mean():.4f}")
    print("-" * 50)
    print(f"  Saved to:")
    print(f"    {EMBEDDINGS_PATH}")
    print(f"    {IMAGE_PATHS_PATH}")
    print(f"    {REPORT_PATH}")
    print("=" * 50)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass", type=int, choices=[1, 2], default=1,
                        dest="emb_pass", help="1=Pass1 embeddings.npy, 2=Pass2 embeddings_final.npy")
    args = parser.parse_args()

    is_pass2 = args.emb_pass == 2
    step_num = "5" if is_pass2 else "3"

    # Override output paths for Pass 2
    if is_pass2:
        EMBEDDINGS_PATH = Path("outputs/embeddings_final.npy")
        REPORT_PATH     = Path("outputs/embedding_report_final.json")
    
    print("=" * 50)
    print(f"    STEP {step_num} — EMBEDDING EXTRACTION (Pass {args.emb_pass})")
    print("=" * 50)

    try:
        # 1. Load ResNet50 without classification head
        model     = load_model()

        # 2. Build image preprocessing transform
        transform = build_transform()

        # 3. Collect all image paths from cleaned data folder
        image_paths = collect_image_paths()
        print(f"\n[DATA] Found {len(image_paths)} images across "
              f"{len(set(p.parent for p in image_paths))} categories")
        
        # 4. Extract embeddings in batches
        embeddings, valid_paths, skipped = extract_embeddings(
            model, image_paths, transform
        )
        # Use correct output path for this pass
        out_emb  = Path("outputs/embeddings_final.npy") if is_pass2 else EMBEDDINGS_PATH
        out_rep  = Path("outputs/embedding_report_final.json") if is_pass2 else REPORT_PATH

        # Temporarily override module-level paths for save_outputs
        #global EMBEDDINGS_PATH, REPORT_PATH
        EMBEDDINGS_PATH = out_emb
        REPORT_PATH     = out_rep

        # 5. Save outputs
        save_outputs(embeddings, valid_paths, skipped)
   
        # 6. Print summary
        print_summary(embeddings, valid_paths, skipped)
        print(f"\n[DONE] Step {step_num} complete.\n")

    except FileNotFoundError as e:
        print(e)
    except RuntimeError as e:
        print(e)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Extraction cancelled by user.")