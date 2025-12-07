"""
Custom YOLO-style Loss Function

Responsabilità Membro B: Model Architect

CORRECTED VERSION:
- Lambda weights balanced (obj=5.0, noobj=0.05)
- Class weights balanced (Blood not over-penalized)
- NO Focal Loss by default (causes low confidence)
- NO GIoU by default (unstable)

Loss = λ_coord * Localization Loss 
     + λ_obj * Objectness Loss (when object present)
     + λ_noobj * Objectness Loss (when no object)
     + λ_class * Classification Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce_loss
        
        if self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Boxes in format: [x, y, w, h] (center coordinates)
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


def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU (GIoU) loss.
    
    Reference: "Generalized Intersection over Union" (Rezatofighi et al., 2019)
    """
    # Convert to corner coordinates
    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2
    
    target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
    target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
    target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
    target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2
    
    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    
    # Smallest enclosing box
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    
    # GIoU loss
    loss = 1 - giou
    
    return loss


class PhobiaLoss(nn.Module):
    """
    Custom YOLO-style loss for PhobiaNet.
    
    CORRECTED VERSION with proper lambda weights:
    - lambda_obj = 5.0 (increased from 1.0)
    - lambda_noobj = 0.05 (decreased from 0.5)
    - Balanced class weights (Blood not over-penalized)
    
    Args:
        lambda_coord: Weight for localization loss (default: 5.0)
        lambda_obj: Weight for objectness when object is present (default: 5.0, CORRECTED!)
        lambda_noobj: Weight for objectness when no object is present (default: 0.05, CORRECTED!)
        lambda_class: Weight for classification loss (default: 1.0)
        grid_size: Grid size (S)
        num_boxes: Number of boxes per cell (B)
        num_classes: Number of classes (C)
        class_weights: Optional tensor of class weights for imbalanced dataset
        use_focal: Use Focal Loss instead of BCE/CE (default: False)
        use_giou: Use GIoU Loss instead of MSE (default: False)
        focal_alpha: Alpha parameter for Focal Loss (default: 0.25)
        focal_gamma: Gamma parameter for Focal Loss (default: 2.0)
    """
    
    def __init__(
        self,
        lambda_coord: float = 5.0,
        lambda_obj: float = 5.0,      # CORRECTED: was 1.0
        lambda_noobj: float = 0.05,   # CORRECTED: was 0.5
        lambda_class: float = 1.0,
        grid_size: int = 13,
        num_boxes: int = 2,
        num_classes: int = 3,
        class_weights=None,
        use_focal: bool = False,
        use_giou: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        self.use_focal = use_focal
        self.use_giou = use_giou
        
        # CORRECTED: More balanced class weights
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.bce_loss = nn.BCELoss(reduction="sum")
        self.ce_loss = nn.CrossEntropyLoss(reduction="sum")
        
        # Focal loss (if enabled)
        if self.use_focal:
            self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='sum')
    
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
        
        # Move class weights to device if provided
        if self.class_weights is not None:
            class_weights = self.class_weights.to(predictions.device)
        else:
            class_weights = None
        
        # Iterate through grid cells
        for i in range(S):
            for j in range(S):
                # Check if target has object in this cell
                target_conf = targets[:, j, i, 4]  # (B,)
                has_object = target_conf > 0  # Boolean mask (B,)
                
                # Process cells with objects
                if has_object.any():
                    # Select cells with objects
                    obj_idx = has_object
                    n_obj_cells += obj_idx.sum().item()
                    
                    # For simplicity, use only the first box (B=0)
                    box_idx = 0
                    
                    # --- Coordinate Loss (GIoU or MSE) ---
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
                    
                    if self.use_giou:
                        # GIoU Loss for bbox regression
                        pred_boxes = torch.stack([pred_x, pred_y, pred_w, pred_h], dim=-1)
                        target_boxes = torch.stack([target_x, target_y, target_w, target_h], dim=-1)
                        
                        coord_loss += giou_loss(pred_boxes, target_boxes).sum()
                    else:
                        # MSE Loss (standard)
                        coord_loss += self.mse_loss(pred_x, target_x)
                        coord_loss += self.mse_loss(pred_y, target_y)
                        
                        # Use square root for width and height (as in original YOLO)
                        coord_loss += self.mse_loss(
                            torch.sqrt(torch.abs(pred_w) + 1e-6),
                            torch.sqrt(torch.abs(target_w) + 1e-6)
                        )
                        coord_loss += self.mse_loss(
                            torch.sqrt(torch.abs(pred_h) + 1e-6),
                            torch.sqrt(torch.abs(target_h) + 1e-6)
                        )
                    
                    # --- Objectness Loss (Focal or BCE) ---
                    pred_conf = predictions[obj_idx, j, i, box_idx * 5 + 4]
                    target_conf = targets[obj_idx, j, i, box_idx * 5 + 4]
                    
                    if self.use_focal:
                        # Focal Loss for objectness
                        conf_loss_obj += self.focal_loss_fn(pred_conf, target_conf)
                    else:
                        # BCE Loss (standard)
                        pred_conf = torch.sigmoid(pred_conf)
                        conf_loss_obj += self.bce_loss(pred_conf, target_conf)
                    
                    # --- Classification Loss (Focal or CE with weights) ---
                    # Get class predictions
                    class_start = B * 5
                    pred_classes = predictions[obj_idx, j, i, class_start:class_start + C]
                    target_classes = targets[obj_idx, j, i, class_start:class_start + C]
                    
                    # Convert target to class indices
                    target_class_idx = torch.argmax(target_classes, dim=-1)
                    
                    if self.use_focal:
                        # Focal Loss for classification
                        pred_probs = F.softmax(pred_classes, dim=-1)
                        target_one_hot = F.one_hot(target_class_idx, num_classes=C).float()
                        
                        for c in range(C):
                            weight = class_weights[c] if class_weights is not None else 1.0
                            focal = self.focal_loss_fn(
                                pred_classes[:, c],
                                target_one_hot[:, c]
                            )
                            class_loss += weight * focal
                    else:
                        # Cross-Entropy with class weights (standard)
                        if class_weights is not None:
                            class_loss += F.cross_entropy(
                                pred_classes,
                                target_class_idx,
                                weight=class_weights,
                                reduction="sum"
                            )
                        else:
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
                        target_conf = torch.zeros_like(pred_conf)
                        
                        if self.use_focal:
                            # Focal Loss for no-object confidence
                            conf_loss_noobj += self.focal_loss_fn(pred_conf, target_conf)
                        else:
                            # BCE Loss (standard)
                            pred_conf = torch.sigmoid(pred_conf)
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


if __name__ == "__main__":
    # Test loss function
    batch_size = 4
    grid_size = 13
    num_boxes = 2
    num_classes = 5
    
    # Create dummy predictions and targets
    predictions = torch.randn(batch_size, grid_size, grid_size, num_boxes * 5 + num_classes)
    targets = torch.zeros(batch_size, grid_size, grid_size, num_boxes * 5 + num_classes)
    
    # Add some dummy targets
    targets[0, 5, 5, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 1.0])  # Box
    targets[0, 5, 5, num_boxes * 5 + 0] = 1.0  # Class 0 (clown)
    
    print("="*60)
    print("Testing PhobiaLoss with CORRECTED Lambda Weights")
    print("="*60)
    
    # Test with CORRECTED lambda weights
    print("\nCORRECTED (lambda_obj=5.0, lambda_noobj=0.05):")
    class_weights = [2.0, 5.0, 3.0, 1.0, 10.0]  # More balanced
    loss_fn_corrected = PhobiaLoss(
        lambda_coord=5.0,
        lambda_obj=5.0,      # CORRECTED: was 1.0
        lambda_noobj=0.05,   # CORRECTED: was 0.5
        lambda_class=1.0,
        grid_size=grid_size,
        num_boxes=num_boxes,
        num_classes=num_classes,
        class_weights=class_weights,
        use_focal=False,
        use_giou=False
    )
    
    total_loss, loss_dict = loss_fn_corrected(predictions, targets)
    
    print(f"   Total Loss: {loss_dict['total_loss']:.2f}")
    print(f"   Coord: {loss_dict['coord_loss']:.2f}")
    print(f"   Obj: {loss_dict['conf_loss_obj']:.2f}")
    print(f"   NoObj: {loss_dict['conf_loss_noobj']:.2f}")
    print(f"   Class: {loss_dict['class_loss']:.2f}")
    
    # Compare with OLD (wrong) weights
    print("\nOLD (lambda_obj=1.0, lambda_noobj=0.5) for comparison:")
    loss_fn_old = PhobiaLoss(
        lambda_coord=5.0,
        lambda_obj=1.0,      # OLD
        lambda_noobj=0.5,    # OLD
        lambda_class=1.0,
        grid_size=grid_size,
        num_boxes=num_boxes,
        num_classes=num_classes,
        class_weights=class_weights,
        use_focal=False,
        use_giou=False
    )
    
    total_loss_old, loss_dict_old = loss_fn_old(predictions, targets)
    
    print(f"   Total Loss: {loss_dict_old['total_loss']:.2f}")
    print(f"   Obj: {loss_dict_old['conf_loss_obj']:.2f}")
    print(f"   NoObj: {loss_dict_old['conf_loss_noobj']:.2f}")
    
    print(f"\n📊 Effect of lambda correction:")
    print(f"   Obj weight increased: {5.0/1.0:.0f}x")
    print(f"   NoObj weight decreased: {0.5/0.05:.0f}x")
    print(f"   Ratio obj/noobj: {5.0/0.05:.0f}x (was {1.0/0.5:.0f}x)")
    
    print("\n✓ Loss function with CORRECTED lambda weights!")