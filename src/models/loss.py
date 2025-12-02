"""
Custom YOLO-style Loss Function

Responsabilità Membro B: Model Architect

Loss = λ_coord * Localization Loss 
     + λ_obj * Objectness Loss (when object present)
     + λ_noobj * Objectness Loss (when no object)
     + λ_class * Classification Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PhobiaLoss(nn.Module):
    """
    Custom YOLO-style loss for PhobiaNet.
    
    Components:
    1. Localization Loss (MSE): Penalizes incorrect bounding box coordinates
    2. Confidence Loss (BCE): Penalizes incorrect objectness predictions
    3. Classification Loss (CE): Penalizes incorrect class predictions
    
    Args:
        lambda_coord: Weight for localization loss
        lambda_obj: Weight for objectness when object is present
        lambda_noobj: Weight for objectness when no object is present
        lambda_class: Weight for classification loss
        grid_size: Grid size (S)
        num_boxes: Number of boxes per cell (B)
        num_classes: Number of classes (C)
    """
    
    def __init__(
        self,
        lambda_coord: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 0.5,
        lambda_class: float = 1.0,
        grid_size: int = 13,
        num_boxes: int = 2,
        num_classes: int = 3,
    ):
        super().__init__()
        
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.bce_loss = nn.BCELoss(reduction="sum")
        self.ce_loss = nn.CrossEntropyLoss(reduction="sum")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss.
        
        Args:
            predictions: Model output (B, S, S, B*(5+C))
            targets: Ground truth (B, S, S, B*5+C)
        
        Returns:
            total_loss: Total loss value
            loss_dict: Dictionary with individual loss components
        """
        batch_size = predictions.size(0)
        S = self.grid_size
        B = self.num_boxes
        C = self.num_classes
        
        # Initialize losses
        coord_loss = torch.tensor(0.0, device=predictions.device)
        conf_loss_obj = torch.tensor(0.0, device=predictions.device)
        conf_loss_noobj = torch.tensor(0.0, device=predictions.device)
        class_loss = torch.tensor(0.0, device=predictions.device)
        
        # Counter for object cells
        n_obj_cells = 0
        n_noobj_cells = 0
        
        # Iterate through grid cells
        for i in range(S):
            for j in range(S):
                # Check if target has object in this cell
                # We check the first box's confidence (simplified)
                target_conf = targets[:, j, i, 4]  # (B,)
                has_object = target_conf > 0  # Boolean mask (B,)
                
                # Process cells with objects
                if has_object.any():
                    # Select cells with objects
                    obj_idx = has_object
                    n_obj_cells += obj_idx.sum().item()
                    
                    # For simplicity, use only the first box (B=0)
                    # In full YOLO, you'd select the box with highest IoU
                    box_idx = 0
                    
                    # --- Coordinate Loss ---
                    # Predicted coordinates
                    pred_x = predictions[obj_idx, j, i, box_idx * 5 + 0]
                    pred_y = predictions[obj_idx, j, i, box_idx * 5 + 1]
                    pred_w = predictions[obj_idx, j, i, box_idx * 5 + 2]
                    pred_h = predictions[obj_idx, j, i, box_idx * 5 + 3]
                    
                    # Target coordinates
                    target_x = targets[obj_idx, j, i, box_idx * 5 + 0]
                    target_y = targets[obj_idx, j, i, box_idx * 5 + 1]
                    target_w = targets[obj_idx, j, i, box_idx * 5 + 2]
                    target_h = targets[obj_idx, j, i, box_idx * 5 + 3]
                    
                    # Apply sigmoid to x, y (they should be in [0, 1])
                    pred_x = torch.sigmoid(pred_x)
                    pred_y = torch.sigmoid(pred_y)
                    
                    # Compute coordinate loss
                    coord_loss += self.mse_loss(pred_x, target_x)
                    coord_loss += self.mse_loss(pred_y, target_y)
                    
                    # Use square root for width and height (as in original YOLO)
                    # This gives more weight to small box errors
                    coord_loss += self.mse_loss(
                        torch.sqrt(torch.abs(pred_w) + 1e-6),
                        torch.sqrt(torch.abs(target_w) + 1e-6)
                    )
                    coord_loss += self.mse_loss(
                        torch.sqrt(torch.abs(pred_h) + 1e-6),
                        torch.sqrt(torch.abs(target_h) + 1e-6)
                    )
                    
                    # --- Objectness Loss (object present) ---
                    pred_conf = predictions[obj_idx, j, i, box_idx * 5 + 4]
                    pred_conf = torch.sigmoid(pred_conf)
                    target_conf = targets[obj_idx, j, i, box_idx * 5 + 4]
                    
                    conf_loss_obj += self.bce_loss(pred_conf, target_conf)
                    
                    # --- Classification Loss ---
                    # Get class predictions
                    class_start = B * 5
                    pred_classes = predictions[obj_idx, j, i, class_start:class_start + C]
                    target_classes = targets[obj_idx, j, i, class_start:class_start + C]
                    
                    # Convert target to class indices
                    target_class_idx = torch.argmax(target_classes, dim=-1)
                    
                    # Compute cross-entropy loss
                    class_loss += F.cross_entropy(
                        pred_classes,
                        target_class_idx,
                        reduction="sum"
                    )
                
                # Process cells without objects
                noobj_idx = ~has_object
                if noobj_idx.any():
                    n_noobj_cells += noobj_idx.sum().item()
                    
                    # Penalize confidence for all boxes in cells without objects
                    for box_idx in range(B):
                        pred_conf = predictions[noobj_idx, j, i, box_idx * 5 + 4]
                        pred_conf = torch.sigmoid(pred_conf)
                        target_conf = torch.zeros_like(pred_conf)
                        
                        conf_loss_noobj += self.bce_loss(pred_conf, target_conf)
        
        # Normalize losses by batch size
        if n_obj_cells > 0:
            coord_loss = coord_loss / batch_size
            conf_loss_obj = conf_loss_obj / batch_size
            class_loss = class_loss / batch_size
        
        if n_noobj_cells > 0:
            conf_loss_noobj = conf_loss_noobj / batch_size
        
        # Weighted sum
        total_loss = (
            self.lambda_coord * coord_loss +
            self.lambda_obj * conf_loss_obj +
            self.lambda_noobj * conf_loss_noobj +
            self.lambda_class * class_loss
        )
        
        # Loss dictionary for logging
        loss_dict = {
            "total_loss": total_loss.item(),
            "coord_loss": coord_loss.item(),
            "conf_loss_obj": conf_loss_obj.item(),
            "conf_loss_noobj": conf_loss_noobj.item(),
            "class_loss": class_loss.item(),
            "n_obj_cells": n_obj_cells,
            "n_noobj_cells": n_noobj_cells,
        }
        
        return total_loss, loss_dict


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Boxes in format: [x, y, w, h] (center coordinates)
    
    Args:
        box1: Tensor of shape (..., 4)
        box2: Tensor of shape (..., 4)
    
    Returns:
        iou: Tensor of shape (...)
    """
    # Convert to corner coordinates
    box1_x1 = box1[..., 0] - box1[..., 2] / 2
    box1_y1 = box1[..., 1] - box1[..., 3] / 2
    box1_x2 = box1[..., 0] + box1[..., 2] / 2
    box1_y2 = box1[..., 1] + box1[..., 3] / 2
    
    box2_x1 = box2[..., 0] - box2[..., 2] / 2
    box2_y1 = box2[..., 1] - box2[..., 3] / 2
    box2_x2 = box2[..., 0] + box2[..., 2] / 2
    box2_y2 = box2[..., 1] + box2[..., 3] / 2
    
    # Intersection area
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou


if __name__ == "__main__":
    # Test loss function
    batch_size = 4
    grid_size = 13
    num_boxes = 2
    num_classes = 3
    
    # Create dummy predictions and targets
    predictions = torch.randn(batch_size, grid_size, grid_size, num_boxes * 5 + num_classes)
    targets = torch.zeros(batch_size, grid_size, grid_size, num_boxes * 5 + num_classes)
    
    # Add some dummy targets
    targets[0, 5, 5, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 1.0])  # Box
    targets[0, 5, 5, num_boxes * 5 + 0] = 1.0  # Class 0 (spider)
    
    # Create loss
    loss_fn = PhobiaLoss(
        lambda_coord=5.0,
        lambda_obj=1.0,
        lambda_noobj=0.5,
        lambda_class=1.0,
        grid_size=grid_size,
        num_boxes=num_boxes,
        num_classes=num_classes,
    )
    
    # Compute loss
    total_loss, loss_dict = loss_fn(predictions, targets)
    
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value}")
    
    # Test IoU
    box1 = torch.tensor([0.5, 0.5, 0.4, 0.4])
    box2 = torch.tensor([0.6, 0.6, 0.4, 0.4])
    iou = compute_iou(box1, box2)
    print(f"\nIoU test: {iou.item():.4f}")
