import os
import shutil
from pathlib import Path

def flatten_dataset(dataset_name):
    base_dir = Path(f"data/raw/{dataset_name}")
    print(f"📦 Flattening {dataset_name} in {base_dir}...")

    # Define target directories
    target_images = base_dir / "images"
    target_labels = base_dir / "labels"
    
    # Create them if they don't exist
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    # 1. Move Images
    # Look for images in all subdirectories recursively
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for img_path in base_dir.rglob(ext):
            # Skip if already in the target folder
            if img_path.parent == target_images:
                continue
                
            # Move file
            new_path = target_images / img_path.name
            if not new_path.exists():
                shutil.move(str(img_path), str(new_path))

    # 2. Move Labels
    # Look for txt files
    for lbl_path in base_dir.rglob("*.txt"):
        # Skip if already in target folder or is classes.txt/data.yaml
        if lbl_path.parent == target_labels or lbl_path.name in ["classes.txt", "data.yaml"]:
            continue
            
        # Move file
        new_path = target_labels / lbl_path.name
        if not new_path.exists():
            shutil.move(str(lbl_path), str(new_path))

    # 3. Clean up empty subfolders (optional but tidy)
    for sub in ["train", "valid", "test", "val"]:
        sub_path = base_dir / sub
        if sub_path.exists():
            shutil.rmtree(sub_path)
            
    # Count results
    img_count = len(list(target_images.glob("*")))
    lbl_count = len(list(target_labels.glob("*")))
    print(f"✅ {dataset_name} flattened: {img_count} images, {lbl_count} labels")

if __name__ == "__main__":
    flatten_dataset("clown")
    flatten_dataset("shark")
