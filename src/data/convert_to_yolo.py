import os
import shutil
import random
from pathlib import Path
import yaml

class AnnotationConverter:
    def __init__(self, class_mapping):
        self.class_mapping = class_mapping

    def create_train_val_split(self, images_dir, labels_dir, output_dir, val_split=0.2):
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        output_dir = Path(output_dir)
        
        # Create final directories
        (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Get all image files
        # Supports typical image extensions
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [f for f in images_dir.iterdir() if f.suffix.lower() in valid_extensions]
        
        # Shuffle
        random.shuffle(images)
        
        # Split index
        split_idx = int(len(images) * (1 - val_split))
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        print(f"   Processing split: {len(train_imgs)} train, {len(val_imgs)} val")
        
        self._move_files(train_imgs, labels_dir, output_dir / "train")
        self._move_files(val_imgs, labels_dir, output_dir / "val")
        
        self._create_yaml(output_dir)

    def _move_files(self, image_list, labels_source, dest_root):
        for img_path in image_list:
            # Copy Image
            shutil.copy2(img_path, dest_root / "images" / img_path.name)
            
            # Find and Copy Label
            # Try .txt first
            label_path = labels_source / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy2(label_path, dest_root / "labels" / label_path.name)

    def _create_yaml(self, output_dir):
        # Invert map for YAML: {0: 'clown', 1: 'shark'}
        names_map = {v: k for k, v in self.class_mapping.items()}
        sorted_names = [names_map[i] for i in range(len(names_map))]
        
        data = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': sorted_names
        }
        
        with open(output_dir / "data.yaml", 'w') as f:
            yaml.dump(data, f, sort_keys=False)