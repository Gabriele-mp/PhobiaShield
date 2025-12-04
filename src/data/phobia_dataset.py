"""
PhobiaDataset - Custom Dataset for PhobiaShield
The Architect Module

Loads images and YOLO-format labels for object detection.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PhobiaDataset(Dataset):
    """
    Dataset for PhobiaShield object detection.
    
    Expects YOLO format labels:
        <class_id> <center_x> <center_y> <width> <height>
    
    All coordinates normalized to [0, 1]
    """
    
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        img_size: int = 416,
        grid_size: int = 13,
        num_boxes: int = 2,
        num_classes: int = 3,
        augment: bool = False
    ):
        """
        Args:
            img_dir: Directory with images
            label_dir: Directory with YOLO format labels
            img_size: Input image size (default 416)
            grid_size: Grid size S (default 13)
            num_boxes: Boxes per cell B (default 2)
            num_classes: Number of classes C (default 3)
            augment: Apply data augmentation
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.augment = augment
        
        # Get image files
        self.image_files = sorted(list(self.img_dir.glob('*.jpg')) + 
                                  list(self.img_dir.glob('*.png')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {img_dir}")
        
        # Setup transforms
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup image transforms."""
        if self.augment:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor (3, H, W)
            target: Tensor (S, S, B*5 + C) - YOLO format
        """
        # Load image
        img_path = self.image_files[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load labels
        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        boxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        boxes.append([cx, cy, w, h])
                        class_labels.append(class_id)
        
        # Apply transforms
        if len(boxes) > 0:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            # No objects, just transform image
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            boxes = []
            class_labels = []
        
        # Convert to target tensor (S, S, B*5 + C)
        target = self._encode_target(boxes, class_labels)
        
        return image, target
    
    def _encode_target(self, boxes, class_labels):
        """
        Encode boxes into YOLO target format.
        
        Args:
            boxes: List of [cx, cy, w, h] normalized
            class_labels: List of class IDs
        
        Returns:
            target: Tensor (S, S, B*5 + C)
        """
        S = self.grid_size
        B = self.num_boxes
        C = self.num_classes
        
        # Initialize target (S, S, B*5 + C)
        target = torch.zeros(S, S, B * 5 + C)
        
        for box, class_id in zip(boxes, class_labels):
            cx, cy, w, h = box
            
            # Determine grid cell
            i = int(cy * S)  # Row
            j = int(cx * S)  # Col
            
            # Clip to grid
            i = min(i, S - 1)
            j = min(j, S - 1)
            
            # Check if cell already has object
            if target[j, i, 4] == 1:
                # Cell occupied, use second box if available
                if B > 1 and target[j, i, 9] == 0:
                    box_idx = 1
                else:
                    continue  # Skip if both boxes occupied
            else:
                box_idx = 0
            
            # Relative position within cell
            x_cell = cx * S - j
            y_cell = cy * S - i
            
            # Box parameters (first box: 0-4, second box: 5-9)
            offset = box_idx * 5
            target[j, i, offset + 0] = x_cell
            target[j, i, offset + 1] = y_cell
            target[j, i, offset + 2] = w
            target[j, i, offset + 3] = h
            target[j, i, offset + 4] = 1.0  # Confidence
            
            # Class (shared for all boxes in cell)
            class_offset = B * 5
            target[j, i, class_offset + class_id] = 1.0
        
        return target

    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        images = torch.stack([b[0] for b in batch], dim=0)
        targets = torch.stack([b[1] for b in batch], dim=0)
        return images, targets

# Test code
if __name__ == "__main__":
    # Test dataset
    dataset = PhobiaDataset(
        img_dir='data/mini_dataset/train/images',
        label_dir='data/mini_dataset/train/labels',
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load one sample
    img, target = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Objects in image: {(target[:, :, 4] > 0).sum().item()}")
    
    print("✓ PhobiaDataset test passed!")