"""
Dataset Balancing Script for PhobiaShield
Oversamples rare classes to create balanced training set
"""

import os
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random

def balance_dataset(data_root, target_ratio=1.0):
    """
    Oversample rare classes to balance dataset
    
    Args:
        data_root: Path to dataset root (contains train/val folders)
        target_ratio: Target instances per class as ratio of max class (1.0 = full balance)
    """
    
    train_dir = Path(data_root) / 'train'
    images_dir = train_dir / 'images'
    labels_dir = train_dir / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ Error: {train_dir} must contain 'images' and 'labels' folders")
        return
    
    # Map class IDs to names
    class_names = {0: 'Clown', 1: 'Shark', 2: 'Spider', 3: 'Blood', 4: 'Needle'}
    
    # Count instances per class and track which files contain each class
    class_counts = Counter()
    files_per_class = defaultdict(list)
    
    print("🔍 Analyzing dataset...")
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file) as f:
            lines = f.readlines()
            if not lines:
                continue
                
            classes_in_file = set()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                class_counts[class_id] += 1
                classes_in_file.add(class_id)
            
            # Track files for each class
            for cls in classes_in_file:
                files_per_class[cls].append(label_file)
    
    print("\n📊 Original distribution:")
    total_instances = sum(class_counts.values())
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        pct = 100 * count / total_instances
        print(f"  Class {cls} ({class_names[cls]:6s}): {count:4d} instances ({pct:5.1f}%)")
    
    # Calculate target count
    max_count = max(class_counts.values())
    target_count = int(max_count * target_ratio)
    
    print(f"\n🎯 Target: {target_count} instances per class (ratio={target_ratio})")
    
    # Oversample rare classes
    total_duplicated = 0
    for cls in sorted(class_counts.keys()):
        current = class_counts[cls]
        needed = target_count - current
        
        if needed <= 0:
            print(f"  Class {cls} ({class_names[cls]:6s}): Already sufficient ({current} >= {target_count})")
            continue
        
        print(f"  Class {cls} ({class_names[cls]:6s}): Duplicating {needed} images...")
        
        # Get files containing this class
        available_files = files_per_class[cls]
        
        # Randomly select files to duplicate
        to_duplicate = random.choices(available_files, k=needed)
        
        for i, src_label in enumerate(to_duplicate):
            base_name = src_label.stem
            
            # Find unique name
            dup_idx = 0
            while True:
                new_name = f"{base_name}_dup{dup_idx}"
                new_label_path = labels_dir / f"{new_name}.txt"
                if not new_label_path.exists():
                    break
                dup_idx += 1
            
            # Copy image
            src_img = images_dir / f"{base_name}.jpg"
            if not src_img.exists():
                src_img = images_dir / f"{base_name}.png"
            
            if src_img.exists():
                dst_img = images_dir / f"{new_name}{src_img.suffix}"
                shutil.copy(src_img, dst_img)
                
                # Copy label
                dst_label = labels_dir / f"{new_name}.txt"
                shutil.copy(src_label, dst_label)
                
                total_duplicated += 1
            else:
                print(f"    ⚠️  Warning: Image not found for {base_name}")
    
    print(f"\n✅ Balanced dataset created!")
    print(f"   Total images duplicated: {total_duplicated}")
    
    # Verify final distribution
    print("\n📊 Final distribution:")
    class_counts_final = Counter()
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_counts_final[int(parts[0])] += 1
    
    total_final = sum(class_counts_final.values())
    for cls in sorted(class_counts_final.keys()):
        count = class_counts_final[cls]
        pct = 100 * count / total_final
        print(f"  Class {cls} ({class_names[cls]:6s}): {count:4d} instances ({pct:5.1f}%)")
    
    print(f"\n📈 Total images: {len(list(images_dir.glob('*.*')))}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Balance PhobiaShield dataset by oversampling rare classes')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--ratio', type=float, default=1.0, help='Target ratio (1.0 = full balance, 0.5 = half balance)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PhobiaShield Dataset Balancer")
    print("=" * 60)
    
    balance_dataset(args.data, args.ratio)
