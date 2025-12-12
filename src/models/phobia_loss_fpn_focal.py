# ============================================================
# loss_fpn.py - OPTIMIZED VERSION WITH FOCAL LOSS
# ============================================================
"""
PhobiaShield FPN Loss Function - Optimized

Features:
1. Focal Loss for confidence (auto-balances positive/negative)
2. Weighted classification loss (handles class imbalance if needed)
3. Optimized lambda weights
4. Multi-scale loss computation (P3, P4, P5)

Author: Member B (Model Architect)
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Automatically down-weights easy examples and focuses on hard examples.
    This is crucial for object detection where negative examples (empty cells)
    vastly outnumber positive examples (cells with objects).
    
    Args:
        alpha (float): Balancing factor for positive/negative examples
                      Default: 0.25 (25% weight on positives)
        gamma (float): Focusing parameter. Higher = more focus on hard examples
                      Default: 2.0 (standard value from paper)
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits (before sigmoid), shape [N]
            targets: Ground truth (0 or 1), shape [N]
        
        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate base cross-entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Calculate p_t: probability of correct class
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Calculate focal weight: (1 - p_t)^gamma
        # This down-weights easy examples (high p_t)
        # and up-weights hard examples (low p_t)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha balancing
        # alpha for positive class, (1-alpha) for negative class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combine all factors
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PhobiaLoss(nn.Module):
    """
    YOLO-style loss for single-scale detection.
    
    Loss Components:
    1. Localization Loss: MSE on bounding box coordinates (x, y, w, h)
    2. Confidence Loss: Focal Loss on objectness score
    3. Classification Loss: Cross-entropy on class predictions
    
    Args:
        grid_size (int): Grid size (e.g., 13 for 13×13 grid)
        num_boxes (int): Number of bounding boxes per grid cell (default: 2)
        num_classes (int): Number of object classes (default: 5)
        lambda_coord (float): Weight for coordinate loss (default: 5.0)
        lambda_conf (float): Weight for confidence loss (default: 1.0)
        lambda_class (float): Weight for classification loss (default: 1.0)
        use_focal_loss (bool): Use Focal Loss for confidence (default: True)
        focal_alpha (float): Focal Loss alpha parameter (default: 0.25)
        focal_gamma (float): Focal Loss gamma parameter (default: 2.0)
    """
    
    def __init__(self, 
                 grid_size, 
                 num_boxes=2, 
                 num_classes=5,
                 lambda_coord=5.0,
                 lambda_conf=1.0,
                 lambda_class=1.0,
                 use_focal_loss=True,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 class_weights=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        # Loss weights
        self.lambda_coord = lambda_coord
        self.lambda_conf = lambda_conf
        self.lambda_class = lambda_class
        
        # Focal Loss for confidence
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # Class weights for classification loss (optional)
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
        else:
            self.class_weights = None
    
    def forward(self, predictions, targets):
        """
        Compute loss for one scale.
        
        Args:
            predictions: Model predictions, shape [B, grid_size, grid_size, num_boxes*(5+num_classes)]
            targets: Ground truth, shape [B, grid_size, grid_size, num_boxes*(5+num_classes)]
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        device = predictions.device
        B = predictions.size(0)
        
        # Initialize loss accumulators
        coord_loss = 0.0
        conf_loss = 0.0
        class_loss = 0.0
        
        # Count for normalization
        num_objects = 0
        
        for b in range(B):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    for box_idx in range(self.num_boxes):
                        box_start = box_idx * 5
                        
                        # Extract predictions
                        pred_x = predictions[b, j, i, box_start + 0]
                        pred_y = predictions[b, j, i, box_start + 1]
                        pred_w = predictions[b, j, i, box_start + 2]
                        pred_h = predictions[b, j, i, box_start + 3]
                        pred_conf = predictions[b, j, i, box_start + 4]  # Logits
                        
                        # Extract targets
                        target_x = targets[b, j, i, box_start + 0]
                        target_y = targets[b, j, i, box_start + 1]
                        target_w = targets[b, j, i, box_start + 2]
                        target_h = targets[b, j, i, box_start + 3]
                        target_conf = targets[b, j, i, box_start + 4]  # 0 or 1
                        
                        # Check if this cell has an object
                        has_object = (target_conf > 0.5)
                        
                        # Confidence loss (ALL cells, positive and negative)
                        if self.use_focal_loss:
                            # Focal Loss automatically balances positive/negative
                            conf_loss += self.focal_loss(pred_conf.unsqueeze(0), 
                                                        target_conf.unsqueeze(0))
                        else:
                            # Standard BCE
                            conf_loss += F.binary_cross_entropy_with_logits(
                                pred_conf.unsqueeze(0),
                                target_conf.unsqueeze(0)
                            )
                        
                        # Localization and classification losses (only for cells with objects)
                        if has_object:
                            num_objects += 1
                            
                            # Localization loss (MSE on coordinates)
                            coord_loss += (
                                F.mse_loss(pred_x, target_x, reduction='sum') +
                                F.mse_loss(pred_y, target_y, reduction='sum') +
                                F.mse_loss(pred_w, target_w, reduction='sum') +
                                F.mse_loss(pred_h, target_h, reduction='sum')
                            )
                            
                            # Classification loss
                            class_start = self.num_boxes * 5
                            pred_classes = predictions[b, j, i, class_start:class_start + self.num_classes]
                            target_class_idx = targets[b, j, i, class_start:class_start + self.num_classes].argmax()
                            
                            if self.class_weights is not None:
                                class_loss += F.cross_entropy(
                                    pred_classes.unsqueeze(0),
                                    target_class_idx.unsqueeze(0),
                                    weight=self.class_weights
                                )
                            else:
                                class_loss += F.cross_entropy(
                                    pred_classes.unsqueeze(0),
                                    target_class_idx.unsqueeze(0)
                                )
        
        # Normalize by batch size
        num_objects = max(num_objects, 1)  # Avoid division by zero
        
        coord_loss = coord_loss / B
        conf_loss = conf_loss / B
        class_loss = class_loss / num_objects
        
        # Combine losses with weights
        total_loss = (
            self.lambda_coord * coord_loss +
            self.lambda_conf * conf_loss +
            self.lambda_class * class_loss
        )
        
        # Return loss and components for logging
        loss_dict = {
            'coord': coord_loss.item(),
            'conf': conf_loss.item(),
            'class': class_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


class FPNLoss(nn.Module):
    """
    Multi-scale FPN Loss.
    
    Combines losses from three detection scales:
    - P3 (52×52): Small objects
    - P4 (26×26): Medium objects
    - P5 (13×13): Large objects
    
    Args:
        Same as PhobiaLoss, applied to all scales
    """
    
    def __init__(self,
                 num_boxes=2,
                 num_classes=5,
                 lambda_coord=5.0,
                 lambda_conf=1.0,
                 lambda_class=1.0,
                 use_focal_loss=True,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 class_weights=None):
        super().__init__()
        
        # Create loss for each scale
        self.loss_p3 = PhobiaLoss(
            grid_size=52,
            num_boxes=num_boxes,
            num_classes=num_classes,
            lambda_coord=lambda_coord,
            lambda_conf=lambda_conf,
            lambda_class=lambda_class,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            class_weights=class_weights
        )
        
        self.loss_p4 = PhobiaLoss(
            grid_size=26,
            num_boxes=num_boxes,
            num_classes=num_classes,
            lambda_coord=lambda_coord,
            lambda_conf=lambda_conf,
            lambda_class=lambda_class,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            class_weights=class_weights
        )
        
        self.loss_p5 = PhobiaLoss(
            grid_size=13,
            num_boxes=num_boxes,
            num_classes=num_classes,
            lambda_coord=lambda_coord,
            lambda_conf=lambda_conf,
            lambda_class=lambda_class,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            class_weights=class_weights
        )
    
    def forward(self, predictions, targets):
        """
        Compute total FPN loss.
        
        Args:
            predictions: Tuple of (pred_p3, pred_p4, pred_p5)
            targets: Tuple of (target_p3, target_p4, target_p5)
        
        Returns:
            total_loss: Combined loss across all scales
            loss_dict: Dictionary with per-scale losses
        """
        pred_p3, pred_p4, pred_p5 = predictions
        target_p3, target_p4, target_p5 = targets
        
        # Compute loss for each scale
        loss_p3, dict_p3 = self.loss_p3(pred_p3, target_p3)
        loss_p4, dict_p4 = self.loss_p4(pred_p4, target_p4)
        loss_p5, dict_p5 = self.loss_p5(pred_p5, target_p5)
        
        # Combine (equal weight for all scales)
        total_loss = loss_p3 + loss_p4 + loss_p5
        
        # Aggregate loss dict for logging
        loss_dict = {
            'loss_p3': loss_p3.item(),
            'loss_p4': loss_p4.item(),
            'loss_p5': loss_p5.item(),
            'coord_p3': dict_p3['coord'],
            'coord_p4': dict_p4['coord'],
            'coord_p5': dict_p5['coord'],
            'conf_p3': dict_p3['conf'],
            'conf_p4': dict_p4['conf'],
            'conf_p5': dict_p5['conf'],
            'class_p3': dict_p3['class'],
            'class_p4': dict_p4['class'],
            'class_p5': dict_p5['class'],
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


# ============================================================
# FACTORY FUNCTION FOR EASY INSTANTIATION
# ============================================================

def create_fpn_loss(config='default'):
    """
    Factory function to create FPN loss with recommended settings.
    
    Args:
        config (str): Configuration preset
            - 'default': Standard settings
            - 'aggressive': More aggressive focal loss (handles severe imbalance)
            - 'conservative': Less aggressive, safer for initial training
    
    Returns:
        FPNLoss instance
    """
    
    if config == 'aggressive':
        # For severe positive/negative imbalance
        return FPNLoss(
            num_boxes=2,
            num_classes=5,
            lambda_coord=5.0,
            lambda_conf=1.0,
            lambda_class=1.0,
            use_focal_loss=True,
            focal_alpha=0.25,   # Standard
            focal_gamma=3.0,    # ← Higher gamma = more aggressive
            class_weights=None
        )
    
    elif config == 'conservative':
        # Safer for initial experiments
        return FPNLoss(
            num_boxes=2,
            num_classes=5,
            lambda_coord=5.0,
            lambda_conf=1.0,
            lambda_class=1.0,
            use_focal_loss=True,
            focal_alpha=0.25,
            focal_gamma=1.5,    # ← Lower gamma = less aggressive
            class_weights=None
        )
    
    else:  # 'default'
        # Recommended: Standard Focal Loss settings from RetinaNet paper
        return FPNLoss(
            num_boxes=2,
            num_classes=5,
            lambda_coord=5.0,
            lambda_conf=1.0,
            lambda_class=1.0,
            use_focal_loss=True,
            focal_alpha=0.25,   # Standard from paper
            focal_gamma=2.0,    # Standard from paper
            class_weights=None  # Auto-balanced by Focal Loss
        )

