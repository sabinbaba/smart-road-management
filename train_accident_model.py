"""
YOLO11 Custom Training Script
Dataset: Car Accident Detection
Classes: Moderate, Severe
"""

from ultralytics import YOLO
import os
import yaml
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH  = r"C:\Users\user\Music\smart-traffic\car_accident_dataset"
DATA_YAML     = os.path.join(DATASET_PATH, "data.yaml")
MODEL         = "yolo11n.pt"       # Pretrained base model (auto-downloads)
PROJECT_NAME  = "accident_training"
RUN_NAME      = "accident_v1"

# Training settings
EPOCHS        = 50       # Number of training cycles (increase for better accuracy)
IMAGE_SIZE    = 640      # Input image size (keep 640 for best results)
BATCH_SIZE    = 16       # Reduce to 8 if you get memory errors
PATIENCE      = 10       # Stop early if no improvement after N epochs
WORKERS       = 4        # Data loading workers

# ─────────────────────────────────────────────
#  FIX data.yaml PATHS (make them absolute)
# ─────────────────────────────────────────────
def fix_yaml_paths(yaml_path, dataset_path):
    """Fix relative paths in data.yaml to absolute paths."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Update paths to absolute
    data["train"] = str(Path(dataset_path) / "train" / "images")
    data["val"]   = str(Path(dataset_path) / "valid" / "images")

    # Add test path if it exists
    test_path = Path(dataset_path) / "test" / "images"
    if test_path.exists():
        data["test"] = str(test_path)

    # Save fixed yaml
    fixed_yaml = str(Path(dataset_path) / "data_fixed.yaml")
    with open(fixed_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"[INFO] Fixed data.yaml saved to: {fixed_yaml}")
    print(f"[INFO] Train path : {data['train']}")
    print(f"[INFO] Val path   : {data['val']}")
    print(f"[INFO] Classes    : {data['names']}")
    return fixed_yaml

# ─────────────────────────────────────────────
#  MAIN TRAINING
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  YOLO11 — Car Accident Detection Training")
    print("  Classes: Moderate, Severe")
    print("=" * 55)

    # Fix yaml paths
    print("\n[STEP 1] Fixing dataset paths...")
    fixed_yaml = fix_yaml_paths(DATA_YAML, DATASET_PATH)

    # Load YOLO11 base model
    print("\n[STEP 2] Loading YOLO11 base model...")
    model = YOLO(MODEL)

    # Start training
    print("\n[STEP 3] Starting training...")
    print(f"         Epochs     : {EPOCHS}")
    print(f"         Image size : {IMAGE_SIZE}")
    print(f"         Batch size : {BATCH_SIZE}")
    print(f"         Patience   : {PATIENCE}")
    print()

    results = model.train(
        data      = fixed_yaml,
        epochs    = EPOCHS,
        imgsz     = IMAGE_SIZE,
        batch     = BATCH_SIZE,
        patience  = PATIENCE,
        workers   = WORKERS,
        project   = PROJECT_NAME,
        name      = RUN_NAME,
        exist_ok  = True,
        pretrained= True,
        optimizer = "Adam",
        lr0       = 0.001,
        verbose   = True,
    )

    # ── Results ─────────────────────────────
    print("\n" + "=" * 55)
    print("  ✅ TRAINING COMPLETE!")
    print("=" * 55)

    best_model = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
    last_model = Path(PROJECT_NAME) / RUN_NAME / "weights" / "last.pt"

    print(f"\n📦 Best model saved at:")
    print(f"   {best_model.absolute()}")
    print(f"\n📦 Last model saved at:")
    print(f"   {last_model.absolute()}")
    print(f"\n📊 Training results saved at:")
    print(f"   {Path(PROJECT_NAME) / RUN_NAME}")
    print("\n💡 Next step: Run validate_model.py to test your model!")

# ─────────────────────────────────────────────
#  VALIDATE (quick test after training)
# ─────────────────────────────────────────────
def validate():
    best_model = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
    if not best_model.exists():
        print("[ERROR] No trained model found. Run training first!")
        return

    print("\n[INFO] Validating model...")
    model = YOLO(str(best_model))
    fixed_yaml = str(Path(DATASET_PATH) / "data_fixed.yaml")
    metrics = model.val(data=fixed_yaml)

    print("\n📊 Validation Results:")
    print(f"   mAP50      : {metrics.box.map50:.3f}")
    print(f"   mAP50-95   : {metrics.box.map:.3f}")
    print(f"   Precision  : {metrics.box.mp:.3f}")
    print(f"   Recall     : {metrics.box.mr:.3f}")

if __name__ == "__main__":
    main()
    validate()
