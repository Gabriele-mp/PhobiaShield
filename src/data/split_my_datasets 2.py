"""
Split dataset in train/val
"""

from src.data.convert_to_yolo import AnnotationConverter

print("=" * 60)
print("📊 Creating Train/Val Split")
print("=" * 60)

# Class mapping (importante per merge futuro!)
class_mapping = {
    "clown": 0,
    "shark": 1,
    # Altri membri aggiungeranno le loro classi
}

converter = AnnotationConverter(class_mapping)

# ========================================
# SPLIT CLOWN
# ========================================
print("\n🤡 Splitting CLOWN dataset (80/20)...")

converter.create_train_val_split(
    images_dir="data/raw/clown/images",
    labels_dir="data/raw/clown/labels",
    output_dir="data/final/clown",
    val_split=0.2
)

# ========================================
# SPLIT SHARK
# ========================================
print("\n🦈 Splitting SHARK dataset (80/20)...")

converter.create_train_val_split(
    images_dir="data/raw/shark/images",
    labels_dir="data/raw/shark/labels",
    output_dir="data/final/shark",
    val_split=0.2
)

print("\n" + "=" * 60)
print("✅ DATASET READY FOR TRAINING!")
print("=" * 60)
print("\n📁 Structure:")
print("   data/final/clown/train/  (~320 images)")
print("   data/final/clown/val/    (~80 images)")
print("   data/final/shark/train/  (~320 images)")
print("   data/final/shark/val/    (~80 images)")
print("\n🎯 Total: ~800 images ready!")
print("=" * 60)