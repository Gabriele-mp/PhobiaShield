"""
Process, validate e split datasets
Branch: TheArchitect
"""
from src.data.validate_my_datasets import DatasetValidator
from src.data.convert_to_yolo import AnnotationConverter
from pathlib import Path

print("=" * 60)
print("🏗️ THE ARCHITECT - Dataset Processing")
print("=" * 60)

# Class mapping (coordinati con team!)
CLASS_MAPPING = {
    "clown": 0,
    "shark": 1,
    # Altri membri aggiungeranno:
    # "spider": 2,
    # "snake": 3,
    # etc.
}

# ========================================
# STEP 1: VALIDATE CLOWN
# ========================================
print("\n🤡 STEP 1: Validating CLOWN dataset...")

clown_val = DatasetValidator(
    "data/raw/clown/images",
    "data/raw/clown/labels"
)
clown_stats = clown_val.validate()
clown_val.print_report(clown_stats)

# ========================================
# STEP 2: VALIDATE SHARK
# ========================================
print("\n🦈 STEP 2: Validating SHARK dataset...")

shark_val = DatasetValidator(
    "data/raw/shark/images",
    "data/raw/shark/labels"
)
shark_stats = shark_val.validate()
shark_val.print_report(shark_stats)

# ========================================
# STEP 3: CLEAN IF NEEDED
# ========================================
print("\n🧹 STEP 3: Cleaning datasets...")

clown_issues = sum(len(v) for v in clown_val.issues.values())
shark_issues = sum(len(v) for v in shark_val.issues.values())

if clown_issues > 0:
    print(f"   Cleaning clown ({clown_issues} issues)...")
    clown_val.clean_dataset("data/clean/clown")
    clown_source = "data/clean/clown"
else:
    print("   Clown dataset is clean!")
    clown_source = "data/raw/clown"

if shark_issues > 0:
    print(f"   Cleaning shark ({shark_issues} issues)...")
    shark_val.clean_dataset("data/clean/shark")
    shark_source = "data/clean/shark"
else:
    print("   Shark dataset is clean!")
    shark_source = "data/raw/shark"

# ========================================
# STEP 4: CREATE TRAIN/VAL SPLIT
# ========================================
print("\n📊 STEP 4: Creating train/val split (80/20)...")

converter = AnnotationConverter(CLASS_MAPPING)

# Split clown
converter.create_train_val_split(
    images_dir=f"{clown_source}/images",
    labels_dir=f"{clown_source}/labels",
    output_dir="data/final/clown",
    val_split=0.2
)

# Split shark
converter.create_train_val_split(
    images_dir=f"{shark_source}/images",
    labels_dir=f"{shark_source}/labels",
    output_dir="data/final/shark",
    val_split=0.2
)

# ========================================
# STEP 5: SUMMARY
# ========================================
print("\n" + "=" * 60)
print("✅ PROCESSING COMPLETE!")
print("=" * 60)

# Count files
clown_train = len(list(Path("data/final/clown/train/images").glob("*.jpg")))
clown_val = len(list(Path("data/final/clown/val/images").glob("*.jpg")))
shark_train = len(list(Path("data/final/shark/train/images").glob("*.jpg")))
shark_val = len(list(Path("data/final/shark/val/images").glob("*.jpg")))

print(f"\n📊 Final Statistics:")
print(f"   🤡 Clown: {clown_train} train, {clown_val} val")
print(f"   🦈 Shark: {shark_train} train, {shark_val} val")
print(f"   📦 Total: {clown_train + shark_train} train, {clown_val + shark_val} val")

print("\n📁 Dataset ready at:")
print("   data/final/clown/")
print("   data/final/shark/")

print("\n🎯 Next steps:")
print("   1. git add download_my_datasets.py process_my_datasets.py")
print("   2. git commit -m 'feat(architect): add clown and shark datasets'")
print("   3. git push origin TheArchitect")
print("=" * 60)