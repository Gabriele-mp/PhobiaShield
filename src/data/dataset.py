"""
PhobiaDataset - Custom PyTorch Dataset for Phobia Object Detection

Responsabilità Membro A: Data & Science Specialist
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PhobiaDataset(Dataset):
    """
    Custom Dataset for Phobia Object Detection.
    
    Supports YOLO format annotations: [class, center_x, center_y, width, height]
    All coordinates are normalized between 0-1.
    
    Args:
        root_dir: Root directory containing images
        annotations_file: Path to annotations file (JSON or TXT)
        image_size: Target image size (H, W)
        transform: Albumentation transforms
        class_mapping: Dictionary mapping class names to IDs
    """
    
    def __init__(
        self,
        root_dir: str,
        annotations_file: str,
        image_size: Tuple[int, int] = (416, 416),
        transform: Optional[A.Compose] = None,
        class_mapping: Optional[Dict[str, int]] = None,
        grid_size: int = 13,
        num_boxes: int = 2,
        num_classes: int = 3,
    ):
        self.root_dir = Path(root_dir)
        self.annotations_file = Path(annotations_file)
        self.image_size = image_size
        self.transform = transform
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        # Default class mapping
        self.class_mapping = class_mapping or {
            "spider": 0,
            "snake": 1,
            "blood": 2,
        }
        
        # Load annotations
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from file."""
        if self.annotations_file.suffix == ".json":
            return self._load_json_annotations()
        elif self.annotations_file.suffix == ".txt":
            return self._load_yolo_annotations()
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotations_file.suffix}")
    
    def _load_json_annotations(self) -> List[Dict]:
        """Load COCO-style JSON annotations."""
        with open(self.annotations_file, "r") as f:
            data = json.load(f)
        
        annotations = []
        for item in data.get("images", []):
            image_id = item["id"]
            file_name = item["file_name"]
            
            # Get all annotations for this image
            image_annots = [
                ann for ann in data.get("annotations", [])
                if ann["image_id"] == image_id
            ]
            
            boxes = []
            for ann in image_annots:
                # Convert COCO bbox [x, y, width, height] to YOLO format
                x, y, w, h = ann["bbox"]
                img_w, img_h = item["width"], item["height"]
                
                # Normalize coordinates
                center_x = (x + w / 2) / img_w
                center_y = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                class_id = ann.get("category_id", 0)
                
                boxes.append([class_id, center_x, center_y, norm_w, norm_h])
            
            if boxes:  # Only add images with annotations
                annotations.append({
                    "image_path": str(self.root_dir / file_name),
                    "boxes": boxes
                })
        
        return annotations
    
    def _load_yolo_annotations(self) -> List[Dict]:
        """Load YOLO-style TXT annotations (one file per image)."""
        annotations = []
        
        # Assume txt file contains list of image paths
        with open(self.annotations_file, "r") as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        for img_path in image_paths:
            # Get corresponding label file
            label_path = Path(img_path).with_suffix(".txt")
            label_path = label_path.parent.parent / "labels" / label_path.name
            
            if not label_path.exists():
                continue
            
            # Read boxes
            boxes = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        boxes.append([int(class_id), cx, cy, w, h])
            
            if boxes:
                annotations.append({
                    "image_path": img_path,
                    "boxes": boxes
                })
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Returns:
            image: Tensor of shape (C, H, W)
            target: Tensor of shape (S, S, B*5 + C)
                   where S=grid_size, B=num_boxes, C=num_classes
        """
        # Load image
        annot = self.annotations[idx]
        image = self._load_image(annot["image_path"])
        
        # Get boxes
        boxes = np.array(annot["boxes"])  # Shape: (N, 5)
        
        # Apply transforms
        if self.transform:
            # Prepare bboxes for albumentation (x_min, y_min, x_max, y_max format)
            bboxes_for_aug = self._yolo_to_pascal(boxes)
            
            transformed = self.transform(
                image=image,
                bboxes=bboxes_for_aug,
                class_labels=boxes[:, 0].astype(int).tolist()
            )
            
            image = transformed["image"]
            
            # Convert back to YOLO format
            if len(transformed["bboxes"]) > 0:
                boxes = self._pascal_to_yolo(
                    np.array(transformed["bboxes"]),
                    np.array(transformed["class_labels"])
                )
            else:
                boxes = np.zeros((0, 5))
        else:
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Create target tensor
        target = self._create_target_tensor(boxes)
        
        return image, target
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and resize image."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        return image
    
    def _yolo_to_pascal(self, boxes: np.ndarray) -> List:
        """Convert YOLO format to Pascal VOC format for albumentation."""
        bboxes = []
        for box in boxes:
            _, cx, cy, w, h = box
            x_min = cx - w / 2
            y_min = cy - h / 2
            x_max = cx + w / 2
            y_max = cy + h / 2
            
            # Clip to [0, 1]
            x_min = max(0, min(1, x_min))
            y_min = max(0, min(1, y_min))
            x_max = max(0, min(1, x_max))
            y_max = max(0, min(1, y_max))
            
            bboxes.append([x_min, y_min, x_max, y_max])
        
        return bboxes
    
    def _pascal_to_yolo(self, bboxes: np.ndarray, class_labels: np.ndarray) -> np.ndarray:
        """Convert Pascal VOC format back to YOLO format."""
        boxes = []
        for bbox, cls in zip(bboxes, class_labels):
            x_min, y_min, x_max, y_max = bbox
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min
            boxes.append([cls, cx, cy, w, h])
        
        return np.array(boxes)
    
    def _create_target_tensor(self, boxes: np.ndarray) -> torch.Tensor:
        """
        Create target tensor in YOLO format.
        
        Target shape: (S, S, B*5 + C)
        - S: Grid size
        - B: Number of boxes per cell
        - 5: [x, y, w, h, confidence]
        - C: Number of classes
        """
        S = self.grid_size
        B = self.num_boxes
        C = self.num_classes
        
        # Initialize target tensor
        target = torch.zeros((S, S, B * 5 + C))
        
        for box in boxes:
            class_id, cx, cy, w, h = box
            class_id = int(class_id)
            
            # Determine which grid cell this box belongs to
            i = int(cx * S)  # Column
            j = int(cy * S)  # Row
            
            # Clip to valid range
            i = min(i, S - 1)
            j = min(j, S - 1)
            
            # Compute cell-relative coordinates
            x_cell = cx * S - i
            y_cell = cy * S - j
            
            # Width and height relative to image
            w_cell = w
            h_cell = h
            
            # Set the first box in this cell (simplified: just use first box)
            if target[j, i, 4] == 0:  # If cell is empty
                # Box coordinates
                target[j, i, 0] = x_cell
                target[j, i, 1] = y_cell
                target[j, i, 2] = w_cell
                target[j, i, 3] = h_cell
                target[j, i, 4] = 1.0  # Confidence
                
                # Class one-hot encoding
                target[j, i, B * 5 + class_id] = 1.0
        
        return target


def get_transforms(config: Dict, mode: str = "train") -> A.Compose:
    """
    Create albumentation transforms based on config.
    
    Args:
        config: Configuration dictionary
        mode: "train", "val", or "test"
    """
    if mode == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


if __name__ == "__main__":
    # Test dataset
    dataset = PhobiaDataset(
        root_dir="data/raw/images",
        annotations_file="data/annotations/train.json",
        image_size=(416, 416),
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        image, target = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Target shape: {target.shape}")
