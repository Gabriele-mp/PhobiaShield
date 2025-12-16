"""
PhobiaNetFPN - Multi-Scale Feature Pyramid Network

Extension of PhobiaNet with FPN for multi-scale object detection.
Detects objects at 3 scales: 52×52 (small), 26×26 (medium), 13×13 (large)

Author: The Architect
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

# Import your existing components
from src.models.phobia_net import ResidualBlock, CBAM, ConvBlock


class FPNNeck(nn.Module):
    """
    Feature Pyramid Network neck for multi-scale feature fusion.
    
    Takes 3 backbone features and produces 3 pyramid levels via:
    1. Lateral connections (1×1 conv to reduce channels)
    2. Top-down pathway (upsample + merge)
    3. Smooth convolutions (reduce aliasing)
    """
    
    def __init__(
        self,
        backbone_channels: List[int] = [128, 256, 512],
        fpn_channels: int = 256
    ):
        super().__init__()
        
        # Lateral connections (reduce channels to fpn_channels)
        self.lateral_c3 = nn.Conv2d(backbone_channels[0], fpn_channels, 1)
        self.lateral_c4 = nn.Conv2d(backbone_channels[1], fpn_channels, 1)
        self.lateral_c5 = nn.Conv2d(backbone_channels[2], fpn_channels, 1)
        
        # Smooth convolutions (3×3 to reduce aliasing after upsampling)
        self.smooth_p3 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
    
    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple:
        """
        Args:
            features: (C3, C4, C5) from backbone
                C3: (B, 128, 52, 52) - early features
                C4: (B, 256, 26, 26) - middle features
                C5: (B, 512, 13, 13) - deep features
        
        Returns:
            (P3, P4, P5): FPN feature pyramids
                P3: (B, 256, 52, 52) - for small objects
                P4: (B, 256, 26, 26) - for medium objects
                P5: (B, 256, 13, 13) - for large objects
        """
        c3, c4, c5 = features
        
        # Top-down pathway
        p5 = self.lateral_c5(c5)  # (B, 256, 13, 13)
        
        p4 = self.lateral_c4(c4)  # (B, 256, 26, 26)
        p4 = p4 + F.interpolate(p5, scale_factor=2, mode='nearest')
        
        p3 = self.lateral_c3(c3)  # (B, 256, 52, 52)
        p3 = p3 + F.interpolate(p4, scale_factor=2, mode='nearest')
        
        # Smooth (reduce aliasing)
        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        
        return p3, p4, p5


class DetectionHead(nn.Module):
    """
    Detection head for one pyramid level.
    Outputs: B × (5 + C) predictions per grid cell
    """
    
    def __init__(
        self,
        in_channels: int,
        num_boxes: int,
        num_classes: int
    ):
        super().__init__()
        
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        output_channels = num_boxes * (5 + num_classes)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, output_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PhobiaNetFPN(nn.Module):
    """
    PhobiaNet with Feature Pyramid Network for multi-scale detection.
    
    Architecture:
        Input (3, 416, 416)
        ↓
        Backbone (reuses ResidualBlock from phobia_net.py)
        ├─ C3: (128, 52, 52)  - early features
        ├─ C4: (256, 26, 26)  - middle features
        └─ C5: (512, 13, 13)  - deep features
        ↓
        FPN Neck
        ├─ P3: (256, 52, 52)  → DetectionHead → small objects
        ├─ P4: (256, 26, 26)  → DetectionHead → medium objects
        └─ P5: (256, 13, 13)  → DetectionHead → large objects
    
    Args:
        config: Model configuration
        use_attention: Use CBAM on each pyramid level (default: True)
    """
    
    def __init__(self, config: Dict, use_attention: bool = True):
        super().__init__()
        
        self.config = config
        self.num_classes = config["output"]["num_classes"]
        self.num_boxes = config["architecture"]["num_boxes_per_cell"]
        self.use_attention = use_attention
        
        # Grid sizes for each pyramid level
        self.grid_sizes = [52, 26, 13]  # P3, P4, P5
        
        # Build multi-scale backbone
        self.backbone = self._build_backbone()
        
        # FPN neck
        self.fpn = FPNNeck(
            backbone_channels=[128, 256, 512],
            fpn_channels=256
        )
        
        # Optional CBAM attention on each pyramid level
        if self.use_attention:
            self.attention_p3 = CBAM(256, reduction=16)
            self.attention_p4 = CBAM(256, reduction=16)
            self.attention_p5 = CBAM(256, reduction=16)
        
        # Detection heads (one per pyramid level)
        self.head_p3 = DetectionHead(256, self.num_boxes, self.num_classes)
        self.head_p4 = DetectionHead(256, self.num_boxes, self.num_classes)
        self.head_p5 = DetectionHead(256, self.num_boxes, self.num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_backbone(self) -> nn.ModuleDict:
        """
        Build backbone that outputs features at 3 scales.
        
        Returns C3, C4, C5 features at strides 8, 16, 32
        """
        leaky_slope = self.config["architecture"].get("leaky_relu_slope", 0.1)
        
        # Initial conv: 416 → 208
        stem = nn.Sequential(
            ResidualBlock(3, 16, stride=2, leaky_slope=leaky_slope),
        )
        
        # Stage 1: 208 → 104 → 52 (stride 8)
        stage1 = nn.Sequential(
            ResidualBlock(16, 32, stride=2, leaky_slope=leaky_slope),
            ResidualBlock(32, 64, stride=2, leaky_slope=leaky_slope),
        )
        
        # Stage 2: 52 → 26 (stride 16)
        stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, leaky_slope=leaky_slope),
        )
        
        # Stage 3: 26 → 13 (stride 32)
        stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, leaky_slope=leaky_slope),
            ResidualBlock(256, 512, stride=1, leaky_slope=leaky_slope),
        )
        
        return nn.ModuleDict({
            'stem': stem,
            'stage1': stage1,
            'stage2': stage2,
            'stage3': stage3,
        })
    
    def _initialize_weights(self):
        """Initialize weights (Kaiming for Conv, constant for BN)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, 3, 416, 416)
        
        Returns:
            (pred_p3, pred_p4, pred_p5): Predictions at 3 scales
                pred_p3: (B, 52, 52, B*(5+C)) - small objects
                pred_p4: (B, 26, 26, B*(5+C)) - medium objects
                pred_p5: (B, 13, 13, B*(5+C)) - large objects
        """
        # Backbone: extract multi-scale features
        x = self.backbone['stem'](x)     # (B, 16, 208, 208)
        
        c3 = self.backbone['stage1'](x)  # (B, 64, 52, 52) then (B, 128, 52, 52)
        # Actually stage1 ends at 64 channels, need adjustment
        # Let me fix the architecture
        
        x = self.backbone['stem'](x)         # (B, 16, 208, 208)
        x = self.backbone['stage1'](x)       # (B, 64, 52, 52)
        
        # Need to get C3 (128 channels) - add extra block
        c3 = ResidualBlock(16, 128, stride=1, leaky_slope=0.1)(x)  # (B, 128, 52, 52)
        
        c4 = self.backbone['stage2'](c3)     # (B, 128, 26, 26) -> need 256
        c4 = ResidualBlock(128, 256, stride=1, leaky_slope=0.1)(c4)  # (B, 256, 26, 26)
        
        c5 = self.backbone['stage3'](c4)     # (B, 512, 13, 13)
        
        # FPN: fuse features
        p3, p4, p5 = self.fpn((c3, c4, c5))
        
        # Optional CBAM attention
        if self.use_attention:
            p3 = self.attention_p3(p3)
            p4 = self.attention_p4(p4)
            p5 = self.attention_p5(p5)
        
        # Detection heads
        pred_p3 = self.head_p3(p3)  # (B, B*(5+C), 52, 52)
        pred_p4 = self.head_p4(p4)  # (B, B*(5+C), 26, 26)
        pred_p5 = self.head_p5(p5)  # (B, B*(5+C), 13, 13)
        
        # Reshape to (B, H, W, B*(5+C))
        pred_p3 = pred_p3.permute(0, 2, 3, 1)
        pred_p4 = pred_p4.permute(0, 2, 3, 1)
        pred_p5 = pred_p5.permute(0, 2, 3, 1)
        
        return pred_p3, pred_p4, pred_p5
    
    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ) -> List[List[Dict]]:
        """
        Make predictions with NMS across all scales.
        
        Args:
            x: (B, 3, H, W)
            conf_threshold: Confidence threshold
            nms_threshold: IoU threshold for NMS
        
        Returns:
            List of predictions for each image
        """
        self.eval()
        with torch.no_grad():
            pred_p3, pred_p4, pred_p5 = self.forward(x)
        
        predictions = []
        batch_size = x.size(0)
        
        for b in range(batch_size):
            all_preds = []
            
            # Process each pyramid level
            for pred, grid_size in [(pred_p3, 52), (pred_p4, 26), (pred_p5, 13)]:
                all_preds.extend(
                    self._decode_predictions(pred[b], grid_size, conf_threshold)
                )
            
            # NMS across all scales
            final_preds = self._nms(all_preds, nms_threshold)
            predictions.append(final_preds)
        
        return predictions
    
    def _decode_predictions(
        self,
        pred: torch.Tensor,
        grid_size: int,
        conf_threshold: float
    ) -> List[Dict]:
        """Decode predictions from one pyramid level"""
        preds = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                for box_idx in range(self.num_boxes):
                    box_start = box_idx * 5
                    
                    x_cell = pred[j, i, box_start + 0]
                    y_cell = pred[j, i, box_start + 1]
                    w = pred[j, i, box_start + 2]
                    h = pred[j, i, box_start + 3]
                    conf = torch.sigmoid(pred[j, i, box_start + 4])
                    
                    class_start = self.num_boxes * 5
                    class_probs = torch.softmax(
                        pred[j, i, class_start:class_start + self.num_classes],
                        dim=0
                    )
                    class_id = torch.argmax(class_probs).item()
                    class_prob = class_probs[class_id].item()
                    
                    final_conf = conf.item() * class_prob
                    
                    if final_conf > conf_threshold:
                        x_abs = (i + torch.sigmoid(x_cell).item()) / grid_size
                        y_abs = (j + torch.sigmoid(y_cell).item()) / grid_size
                        w_abs = torch.abs(w).item()
                        h_abs = torch.abs(h).item()
                        
                        preds.append({
                            "class_id": class_id,
                            "confidence": final_conf,
                            "bbox": [x_abs, y_abs, w_abs, h_abs],
                            "grid_size": grid_size  # Track which scale
                        })
        
        return preds
    
    def _nms(self, predictions: List[Dict], iou_threshold: float) -> List[Dict]:
        """Non-Maximum Suppression across all scales"""
        if len(predictions) == 0:
            return []
        
        # Group by class
        class_preds = {}
        for pred in predictions:
            cls_id = pred['class_id']
            if cls_id not in class_preds:
                class_preds[cls_id] = []
            class_preds[cls_id].append(pred)
        
        # NMS per class
        final_preds = []
        for cls_id, preds in class_preds.items():
            # Sort by confidence
            preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)
            
            keep = []
            while len(preds) > 0:
                keep.append(preds[0])
                preds = [
                    p for p in preds[1:]
                    if self._compute_iou(keep[-1]['bbox'], p['bbox']) < iou_threshold
                ]
            
            final_preds.extend(keep)
        
        return final_preds
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes [cx, cy, w, h]"""
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0


def create_fpn_model(config: Dict, use_attention: bool = True) -> PhobiaNetFPN:
    """Create PhobiaNetFPN from config"""
    return PhobiaNetFPN(config, use_attention=use_attention)
