"""
PhobiaShield - Dataset Setup Script

Downloads and extracts DATASET_ULTIMATE_COMPLETE from Google Drive.

Usage:
    python scripts/setup_dataset.py
"""

import os
import zipfile
from pathlib import Path
import sys

# Configuration
DRIVE_LINK = "YOUR_GOOGLE_DRIVE_SHARE_LINK"  # Team: Update this!
LOCAL_ZIP = "data/DATASET_ULTIMATE_COMPLETE.zip"
EXTRACT_TO = "data/phobiashield_ultimate"

def check_dataset_exists():
    """Check if dataset is already extracted"""
    paths_to_check = [
        f"{EXTRACT_TO}/train/images",
        f"{EXTRACT_TO}/val/images",
        f"{EXTRACT_TO}/test/images"
    ]
    
    return all(os.path.exists(p) for p in paths_to_check)

def count_images(directory):
    """Count images in directory"""
    path = Path(directory)
    return len(list(path.glob('*.jpg'))) + len(list(path.glob('*.png')))

def extract_dataset():
    """Extract dataset zip file"""
    if not os.path.exists(LOCAL_ZIP):
        print(f"❌ Dataset zip not found: {LOCAL_ZIP}")
        print(f"\n📥 Please download from Google Drive:")
        print(f"   {DRIVE_LINK}")
        print(f"\n💾 Save to: {LOCAL_ZIP}")
        return False
    
    print(f"📦 Extracting dataset...")
    print(f"   From: {LOCAL_ZIP}")
    print(f"   To: {EXTRACT_TO}")
    
    os.makedirs(EXTRACT_TO, exist_ok=True)
    
    try:
        with zipfile.ZipFile(LOCAL_ZIP, 'r') as z:
            z.extractall(EXTRACT_TO)
        
        print(f"✅ Extraction complete!")
        return True
    
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def verify_dataset():
    """Verify dataset integrity"""
    print(f"\n🔍 Verifying dataset...")
    
    splits = ['train', 'val', 'test']
    total_images = 0
    
    for split in splits:
        img_dir = f"{EXTRACT_TO}/{split}/images"
        lbl_dir = f"{EXTRACT_TO}/{split}/labels"
        
        if not os.path.exists(img_dir):
            print(f"❌ Missing: {img_dir}")
            return False
        
        if not os.path.exists(lbl_dir):
            print(f"❌ Missing: {lbl_dir}")
            return False
        
        img_count = count_images(img_dir)
        lbl_count = len(list(Path(lbl_dir).glob('*.txt')))
        
        print(f"   {split.capitalize()}: {img_count} images, {lbl_count} labels")
        total_images += img_count
    
    print(f"\n✅ Total: {total_images} images")
    
    # Expected counts
    expected = {
        'train': 7593,
        'val': 1624,
        'test': 1634
    }
    
    actual_train = count_images(f"{EXTRACT_TO}/train/images")
    
    if abs(actual_train - expected['train']) > 10:
        print(f"⚠️  Warning: Expected ~{expected['train']} train images, got {actual_train}")
    
    return True

def main():
    """Main setup function"""
    print("="*70)
    print("PhobiaShield - Dataset Setup")
    print("="*70)
    
    # Check if already exists
    if check_dataset_exists():
        print("\n✅ Dataset already exists!")
        print(f"   Location: {EXTRACT_TO}")
        
        response = input("\n🔄 Re-extract? (y/n): ").strip().lower()
        if response != 'y':
            print("✅ Using existing dataset")
            verify_dataset()
            return 0
    
    # Extract
    if not extract_dataset():
        return 1
    
    # Verify
    if not verify_dataset():
        print("\n❌ Dataset verification failed!")
        print("Please check the zip file and try again.")
        return 1
    
    print("\n" + "="*70)
    print("✅ DATASET SETUP COMPLETE!")
    print("="*70)
    print(f"\nDataset location: {os.path.abspath(EXTRACT_TO)}")
    print("\nNext steps:")
    print("  1. Verify data with: python scripts/visualize_dataset.py")
    print("  2. Start training: python scripts/train.py")
    print("  3. Or use notebooks: notebooks/01_FPN_Training.ipynb")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
