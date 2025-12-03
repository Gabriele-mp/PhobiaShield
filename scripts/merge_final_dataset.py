"""
Final Dataset Merge Script
Unisce tutti i 5 dataset del team con split automatico

Team datasets:
- Clown (ID 0)
- Shark (ID 1) 
- Spider (ID 2)
- Blood (ID 3) - from Marco
- Needle (ID 4) - from Marco

Usage:
    python scripts/merge_final_dataset.py
"""

import shutil
from pathlib import Path
from tqdm import tqdm
import random
from collections import defaultdict

# Global mapping
GLOBAL_MAPPING = {
    'clown': 0,
    'shark': 1,
    'spider': 2,
    'blood': 3,
    'needle': 4
}

CLASS_NAMES = ['Clown', 'Shark', 'Spider', 'Blood', 'Needle']

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def remap_label_file(label_path, old_id, new_id, filter_only_target=False, keep_only_class_id=None):
    """
    Remap class IDs in label file.
    
    Args:
        label_path: Path to label file
        old_id: Old class ID (can be None if using keep_only_class_id)
        new_id: New class ID (from GLOBAL_MAPPING)
        filter_only_target: Legacy parameter
        keep_only_class_id: If specified, keep ONLY lines with this class_id, discard all others
    
    Returns:
        bool: True if file has objects after filtering, False otherwise
    """
    lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                
                # Mode 1: Keep only specific class
                if keep_only_class_id is not None:
                    if class_id == keep_only_class_id:
                        # This is the class we want, remap to new_id
                        parts[0] = str(new_id)
                        lines.append(' '.join(parts))
                    # else: skip this line completely
                
                # Mode 2: Normal remap
                elif old_id is not None and class_id == old_id:
                    parts[0] = str(new_id)
                    lines.append(' '.join(parts))
                elif old_id is None:
                    lines.append(' '.join(parts))
    
    # Only write if we have lines
    if lines:
        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))
        return True
    else:
        return False

def process_dataset(source_images, source_labels, class_name, class_id, output_dir):
    """
    Process single dataset and copy to staging area.
    
    Args:
        source_images: Path to source images
        source_labels: Path to source labels
        class_name: Name of class
        class_id: Global class ID
        output_dir: Staging output directory
    
    Returns:
        count: Number of images processed
    """
    source_images = Path(source_images)
    source_labels = Path(source_labels)
    output_dir = Path(output_dir)
    
    staging_images = output_dir / 'images'
    staging_labels = output_dir / 'labels'
    staging_images.mkdir(parents=True, exist_ok=True)
    staging_labels.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(source_images.glob('*.jpg')) + list(source_images.glob('*.png'))
    
    count = 0
    for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
        # Check for corresponding label
        label_path = source_labels / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            print(f"  ⚠️  Skipping {img_path.name} - no label found")
            continue
        
        # Copy image with unique name
        new_img_name = f"{class_name}_{img_path.name}"
        shutil.copy2(img_path, staging_images / new_img_name)
        
        # Copy and remap label
        new_label_name = f"{class_name}_{img_path.stem}.txt"
        new_label_path = staging_labels / new_label_name
        shutil.copy2(label_path, new_label_path)
        
        # Remap class IDs
        # For Blood: filter to keep ONLY class_id=3 (blood objects)
        # For others: remap from 0 to target class_id
        if class_name == 'blood':
            # Filter: keep only blood objects (original class 3)
            has_objects = remap_label_file(new_label_path, old_id=None, new_id=class_id, keep_only_class_id=3)
            if not has_objects:
                # No blood objects in this image, remove it
                new_label_path.unlink()
                (staging_images / new_img_name).unlink()
                continue
        elif class_name == 'needle':
            # Filter: keep only needle objects (original class 4)
            has_objects = remap_label_file(new_label_path, old_id=None, new_id=class_id, keep_only_class_id=4)
            if not has_objects:
                new_label_path.unlink()
                (staging_images / new_img_name).unlink()
                continue
        else:
            # Normal remap from 0 to class_id (for clown, shark, spider)
            remap_label_file(new_label_path, old_id=0, new_id=class_id, keep_only_class_id=None)
        
        count += 1
    
    return count

def split_dataset(staging_dir, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split staging dataset into train/val/test.
    
    Args:
        staging_dir: Directory with all images/labels
        output_dir: Final output directory
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
    """
    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)
    
    # Get all images
    images = list((staging_dir / 'images').glob('*'))
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    print(f"\n📊 Split statistics:")
    print(f"  Total: {total} images")
    print(f"  Train: {len(splits['train'])} ({len(splits['train'])/total*100:.1f}%)")
    print(f"  Val:   {len(splits['val'])} ({len(splits['val'])/total*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} ({len(splits['test'])/total*100:.1f}%)")
    
    # Copy to final structure
    for split_name, img_list in splits.items():
        split_img_dir = output_dir / split_name / 'images'
        split_label_dir = output_dir / split_name / 'labels'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(img_list, desc=f"Creating {split_name} split"):
            # Copy image
            shutil.copy2(img_path, split_img_dir / img_path.name)
            
            # Copy label
            label_path = staging_dir / 'labels' / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, split_label_dir / f"{img_path.stem}.txt")

def analyze_dataset(dataset_dir):
    """Print dataset statistics."""
    dataset_dir = Path(dataset_dir)
    
    print(f"\n{'='*60}")
    print(f"FINAL DATASET STATISTICS")
    print(f"{'='*60}")
    
    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split / 'labels'
        if not split_dir.exists():
            continue
        
        # Count by class
        class_counts = defaultdict(int)
        total = 0
        
        for label_file in split_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total += 1
        
        print(f"\n{split.upper()}:")
        print(f"  Images: {len(list((dataset_dir / split / 'images').glob('*')))}")
        print(f"  Objects: {total}")
        for class_id in sorted(class_counts.keys()):
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
            count = class_counts[class_id]
            print(f"    {class_name}: {count} ({count/total*100:.1f}%)")

def main():
    print("="*60)
    print("PhobiaShield - Final Dataset Merge")
    print("The Architect Module")
    print("="*60)
    
    # Paths configuration
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / 'data'
    
    # Source datasets
    DATASETS = {
        'clown': {
            'images': DATA_DIR / 'raw' / 'clown' / 'images',
            'labels': DATA_DIR / 'raw' / 'clown' / 'labels',
            'class_id': 0
        },
        'shark': {
            'images': DATA_DIR / 'raw' / 'shark' / 'images',
            'labels': DATA_DIR / 'raw' / 'shark' / 'labels',
            'class_id': 1
        },
        'spider': {
            'images': Path.home() / 'Desktop' / 'Phobia' / 'images',
            'labels': Path.home() / 'Desktop' / 'Phobia' / 'labels',
            'class_id': 2
        },
        'blood': {
            'images': Path.home() / 'Desktop' / 'Marco_Data' / 'Blood_ID3' / 'images',
            'labels': Path.home() / 'Desktop' / 'Marco_Data' / 'Blood_ID3' / 'labels',
            'class_id': 3
        },
        'needle': {
            'images': Path.home() / 'Desktop' / 'Marco_Data' / 'Needles_ID4' / 'images',
            'labels': Path.home() / 'Desktop' / 'Marco_Data' / 'Needles_ID4' / 'labels',
            'class_id': 4
        }
    }
    
    # Output directories
    staging_dir = DATA_DIR / 'staging'
    final_dir = DATA_DIR / 'phobiashield_final'
    
    # Clean staging if exists
    if staging_dir.exists():
        print(f"\n🗑️  Cleaning staging directory...")
        shutil.rmtree(staging_dir)
    
    # Process each dataset
    print(f"\n📦 Processing datasets...")
    total_images = 0
    
    for class_name, config in DATASETS.items():
        if not config['images'].exists():
            print(f"⚠️  {class_name.upper()} not found at {config['images']}, skipping...")
            continue
        
        count = process_dataset(
            config['images'],
            config['labels'],
            class_name,
            config['class_id'],
            staging_dir
        )
        total_images += count
        print(f"  ✓ {class_name.upper()}: {count} images")
    
    print(f"\n✓ Total processed: {total_images} images")
    
    # Split dataset
    print(f"\n📊 Splitting into train/val/test...")
    split_dataset(staging_dir, final_dir, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    # Analyze final dataset
    analyze_dataset(final_dir)
    
    # Clean staging
    print(f"\n🗑️  Cleaning staging directory...")
    shutil.rmtree(staging_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ FINAL DATASET READY!")
    print(f"{'='*60}")
    print(f"Location: {final_dir}")
    print(f"\nClasses: {', '.join(CLASS_NAMES)}")
    print(f"Global mapping: {GLOBAL_MAPPING}")
    print(f"\nNext steps:")
    print(f"  1. Update cfg/data/phobia.yaml with:")
    print(f"     root: data/phobiashield_final")
    print(f"     num_classes: 5")
    print(f"  2. Train: python scripts/train_complete.py")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    random.seed(42)  # For reproducible splits
    main()