"""
FPN Loss - Multi-Scale Loss for PhobiaNetFPN

Computes loss across 3 pyramid levels with appropriate weighting.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

from src.models.loss import PhobiaLoss


class FPNLoss(nn.Module):
    """
    Multi-scale loss for FPN.
    
    Computes PhobiaLoss at each pyramid level and combines them.
    
    Args:
        num_classes: Number of classes
        num_boxes: Boxes per cell
        scale_weights: Weights for each scale [P3, P4, P5]
        **loss_kwargs: Arguments for PhobiaLoss
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        num_boxes: int = 2,
        scale_weights: list = [1.0, 1.0, 1.0],  # Equal weight for all scales
        **loss_kwargs
    ):
        super().__init__()
        
        self.scale_weights = scale_weights
        
        # Loss for each pyramid level
        self.loss_p3 = PhobiaLoss(
            grid_size=52,
            num_boxes=num_boxes,
            num_classes=num_classes,
            **loss_kwargs
        )
        
        self.loss_p4 = PhobiaLoss(
            grid_size=26,
            num_boxes=num_boxes,
            num_classes=num_classes,
            **loss_kwargs
        )
        
        self.loss_p5 = PhobiaLoss(
            grid_size=13,
            num_boxes=num_boxes,
            num_classes=num_classes,
            **loss_kwargs
        )
    
    def forward(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute FPN loss.
        
        Args:
            predictions: (pred_p3, pred_p4, pred_p5)
            targets: (target_p3, target_p4, target_p5)
        
        Returns:
            total_loss: Weighted sum of losses
            loss_dict: Individual loss components
        """
        pred_p3, pred_p4, pred_p5 = predictions
        target_p3, target_p4, target_p5 = targets
        
        # Compute loss at each scale
        loss_p3, comp_p3 = self.loss_p3(pred_p3, target_p3)
        loss_p4, comp_p4 = self.loss_p4(pred_p4, target_p4)
        loss_p5, comp_p5 = self.loss_p5(pred_p5, target_p5)
        
        # Weighted combination
        total_loss = (
            self.scale_weights[0] * loss_p3 +
            self.scale_weights[1] * loss_p4 +
            self.scale_weights[2] * loss_p5
        )
        
        # Loss dict for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'loss_p3': loss_p3.item(),
            'loss_p4': loss_p4.item(),
            'loss_p5': loss_p5.item(),
            # Per-scale components
            'p3_coord': comp_p3['coord_loss'],
            'p3_obj': comp_p3['conf_loss_obj'],
            'p3_noobj': comp_p3['conf_loss_noobj'],
            'p3_cls': comp_p3['class_loss'],
            'p4_coord': comp_p4['coord_loss'],
            'p4_obj': comp_p4['conf_loss_obj'],
            'p4_noobj': comp_p4['conf_loss_noobj'],
            'p4_cls': comp_p4['class_loss'],
            'p5_coord': comp_p5['coord_loss'],
            'p5_obj': comp_p5['conf_loss_obj'],
            'p5_noobj': comp_p5['conf_loss_noobj'],
            'p5_cls': comp_p5['class_loss'],
        }
        
        return total_loss, loss_dict