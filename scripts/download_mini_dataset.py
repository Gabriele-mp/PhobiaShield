"""
Quick Mini Dataset Downloader
Scarica 50 immagini per classe per test veloce

Usage:
    python scripts/download_mini_dataset.py
"""

import os
import requests
from pathlib import Path
import zipfile
from tqdm import tqdm

def download_file(url, dest_path):
    """Download file con progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_coco_subset():
    """Download COCO validation subset (piccolo)."""
    print("📥 Downloading COCO validation images...")
    
    # COCO val2017 images (solo subset)
    url = "http://images.cocodataset.org/zips/val2017.zip"
    dest = Path("data/raw/coco_val2017.zip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if not dest.exists():
        download_file(url, dest)
        print("✓ Downloaded COCO images")
    
    # Extract
    print("📂 Extracting...")
    with zipfile.ZipFile(dest, 'r') as zip_ref:
        zip_ref.extractall("data/raw/")
    print("✓ Extracted")
    
    # Annotations
    print("📥 Downloading annotations...")
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_dest = Path("data/raw/annotations.zip")
    
    if not ann_dest.exists():
        download_file(ann_url, ann_dest)
    
    with zipfile.ZipFile(ann_dest, 'r') as zip_ref:
        zip_ref.extractall("data/raw/")
    print("✓ Annotations ready")

def create_mini_dataset():
    """Crea mini dataset da COCO con 3 classi."""
    import json
    from shutil import copy2
    
    print("\n🔨 Creating mini dataset...")
    
    # Load COCO annotations
    with open('data/raw/annotations/instances_val2017.json', 'r') as f:
        coco = json.load(f)
    
    # Map COCO categories to our classes
    # COCO: person=1, cat=17, dog=18
    # Usiamo queste come proxy per test
    target_categories = {
        1: 0,   # person → clown (proxy)
        17: 1,  # cat → shark (proxy) 
        18: 2   # dog → spider (proxy)
    }
    
    # Create output structure
    output_dir = Path("data/mini_dataset")
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process annotations
    images_by_category = {cat_id: [] for cat_id in target_categories.keys()}
    
    for ann in tqdm(coco['annotations'], desc="Processing annotations"):
        if ann['category_id'] in target_categories:
            img_id = ann['image_id']
            images_by_category[ann['category_id']].append({
                'image_id': img_id,
                'bbox': ann['bbox'],
                'category': target_categories[ann['category_id']]
            })
    
    # Select 50 images per category
    selected_images = {}
    for cat_id, anns in images_by_category.items():
        selected = anns[:50]  # First 50
        for ann in selected:
            img_id = ann['image_id']
            if img_id not in selected_images:
                selected_images[img_id] = []
            selected_images[img_id].append(ann)
    
    # Copy images and create labels
    img_info = {img['id']: img for img in coco['images']}
    
    for i, (img_id, anns) in enumerate(tqdm(selected_images.items(), desc="Creating dataset")):
        # Determine split (70/15/15)
        if i < len(selected_images) * 0.7:
            split = 'train'
        elif i < len(selected_images) * 0.85:
            split = 'val'
        else:
            split = 'test'
        
        # Get image info
        img = img_info[img_id]
        src_path = Path(f"data/raw/val2017/{img['file_name']}")
        
        if not src_path.exists():
            continue
        
        # Copy image
        dst_path = output_dir / split / 'images' / img['file_name']
        copy2(src_path, dst_path)
        
        # Create label (YOLO format)
        label_path = output_dir / split / 'labels' / f"{img['file_name'].split('.')[0]}.txt"
        
        img_w, img_h = img['width'], img['height']
        
        with open(label_path, 'w') as f:
            for ann in anns:
                x, y, w, h = ann['bbox']
                # Convert to YOLO format (center_x, center_y, width, height) normalized
                cx = (x + w/2) / img_w
                cy = (y + h/2) / img_h
                nw = w / img_w
                nh = h / img_h
                
                f.write(f"{ann['category']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    
    print(f"\n✓ Mini dataset created in {output_dir}")
    print(f"  Train: {len(list((output_dir/'train'/'images').glob('*')))} images")
    print(f"  Val:   {len(list((output_dir/'val'/'images').glob('*')))} images")
    print(f"  Test:  {len(list((output_dir/'test'/'images').glob('*')))} images")

if __name__ == "__main__":
    print("="*50)
    print("Mini Dataset Downloader - PhobiaShield")
    print("="*50)
    
    # Download COCO subset
    download_coco_subset()
    
    # Create mini dataset
    create_mini_dataset()
    
    print("\n✅ Done! Dataset ready in data/mini_dataset/")