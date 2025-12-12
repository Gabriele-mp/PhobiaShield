"""
PhobiaNetFPN - "V2" Architecture
Matched to FPN_epoch20_loss6.6473.pth weights.

Fixed by Member 3 (The Engineer) combining:
1. Teammate's Simplified Backbone (ConvBlock based)
2. Inference Logic (predict/nms/decode)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

class ConvBlock(nn.Module):
    """Simple Conv+BN+LeakyReLU block"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class FPNNeck(nn.Module):
    def __init__(self, backbone_channels=[128, 256, 512], fpn_channels=256):
        super().__init__()
        self.lateral_c3 = nn.Conv2d(backbone_channels[0], fpn_channels, 1)
        self.lateral_c4 = nn.Conv2d(backbone_channels[1], fpn_channels, 1)
        self.lateral_c5 = nn.Conv2d(backbone_channels[2], fpn_channels, 1)
        
        self.smooth_p3 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
    
    def forward(self, features):
        c3, c4, c5 = features
        
        # Top-down pathway
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral_c3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        
        return self.smooth_p3(p3), self.smooth_p4(p4), self.smooth_p5(p5)

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_boxes, num_classes):
        super().__init__()
        output_channels = num_boxes * (5 + num_classes)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, output_channels, 1)
        )
    def forward(self, x):
        return self.conv(x)

class PhobiaNetFPN(nn.Module):
    """
    FPN with simple ConvBlocks - 'V2' architecture.
    Matches weights: FPN_epoch20...pth
    """
    def __init__(self, config, use_attention=False):
        super().__init__()
        self.num_classes = config["output"]["num_classes"]
        self.num_boxes = config["architecture"]["num_boxes_per_cell"]
        
        # Simplified backbone with ConvBlock (Matches Teammate's Code)
        self.conv1 = ConvBlock(3, 16, stride=2)      # 416 -> 208
        self.conv2 = ConvBlock(16, 32, stride=2)     # 208 -> 104
        self.conv3 = ConvBlock(32, 64, stride=2)     # 104 -> 52
        self.conv4 = ConvBlock(64, 128, stride=1)    # 52 -> 52 (C3)
        self.conv5 = ConvBlock(128, 256, stride=2)   # 52 -> 26 (C4)
        self.conv6 = ConvBlock(256, 512, stride=2)   # 26 -> 13 (C5)
        
        # FPN
        self.fpn = FPNNeck([128, 256, 512], 256)
        
        # Detection heads
        self.head_p3 = DetectionHead(256, self.num_boxes, self.num_classes)
        self.head_p4 = DetectionHead(256, self.num_boxes, self.num_classes)
        self.head_p5 = DetectionHead(256, self.num_boxes, self.num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Backbone execution
        x = self.conv1(x)   # 16, 208
        x = self.conv2(x)   # 32, 104
        x = self.conv3(x)   # 64, 52
        c3 = self.conv4(x)  # 128, 52 - C3
        c4 = self.conv5(c3) # 256, 26 - C4
        c5 = self.conv6(c4) # 512, 13 - C5
        
        # FPN
        p3, p4, p5 = self.fpn((c3, c4, c5))
        
        # Heads
        pred_p3 = self.head_p3(p3).permute(0, 2, 3, 1)
        pred_p4 = self.head_p4(p4).permute(0, 2, 3, 1)
        pred_p5 = self.head_p5(p5).permute(0, 2, 3, 1)
        
        return pred_p3, pred_p4, pred_p5

    # --- INFERENCE METHODS (Added by Member 3 for Demo) ---
    
    def predict(self, x: torch.Tensor, conf_threshold: float = 0.5, nms_threshold: float = 0.4) -> List[List[Dict]]:
        self.eval()
        with torch.no_grad():
            pred_p3, pred_p4, pred_p5 = self.forward(x)
        
        predictions = []
        batch_size = x.size(0)
        
        for b in range(batch_size):
            all_preds = []
            # Process each pyramid level
            # P3 (52x52), P4 (26x26), P5 (13x13)
            for pred, grid_size in [(pred_p3, 52), (pred_p4, 26), (pred_p5, 13)]:
                all_preds.extend(self._decode_predictions(pred[b], grid_size, conf_threshold))
            
            final_preds = self._nms(all_preds, nms_threshold)
            predictions.append(final_preds)
        
        return predictions

    def _decode_predictions(self, pred, grid_size, conf_threshold):
        preds = []
        for i in range(grid_size):
            for j in range(grid_size):
                for box_idx in range(self.num_boxes):
                    box_start = box_idx * 5
                    
                    conf = torch.sigmoid(pred[j, i, box_start + 4])
                    class_start = self.num_boxes * 5
                    class_probs = torch.softmax(pred[j, i, class_start:class_start + self.num_classes], dim=0)
                    class_id = torch.argmax(class_probs).item()
                    final_conf = conf.item() * class_probs[class_id].item()
                    
                    if final_conf > conf_threshold:
                        x_cell = pred[j, i, box_start + 0]
                        y_cell = pred[j, i, box_start + 1]
                        w = pred[j, i, box_start + 2]
                        h = pred[j, i, box_start + 3]
                        
                        x_abs = (i + torch.sigmoid(x_cell).item()) / grid_size
                        y_abs = (j + torch.sigmoid(y_cell).item()) / grid_size
                        w_abs = torch.abs(w).item()
                        h_abs = torch.abs(h).item()
                        
                        preds.append({
                            "class_id": class_id,
                            "confidence": final_conf,
                            "bbox": [x_abs, y_abs, w_abs, h_abs]
                        })
        return preds

    def _nms(self, predictions, iou_threshold):
        if not predictions: return []
        
        class_preds = {}
        for pred in predictions:
            cid = pred['class_id']
            if cid not in class_preds: class_preds[cid] = []
            class_preds[cid].append(pred)
            
        final_preds = []
        for cid, preds in class_preds.items():
            preds.sort(key=lambda x: x['confidence'], reverse=True)
            keep = []
            while len(preds) > 0:
                curr = preds[0]
                keep.append(curr)
                preds = [p for p in preds[1:] if self._compute_iou(curr['bbox'], p['bbox']) < iou_threshold]
            final_preds.extend(keep)
        return final_preds

    def _compute_iou(self, box1, box2):
        x1_min = box1[0] - box1[2]/2
        y1_min = box1[1] - box1[3]/2
        x1_max = box1[0] + box1[2]/2
        y1_max = box1[1] + box1[3]/2
        
        x2_min = box2[0] - box2[2]/2
        y2_min = box2[1] - box2[3]/2
        x2_max = box2[0] + box2[2]/2
        y2_max = box2[1] + box2[3]/2
        
        inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        inter_area = inter_w * inter_h
        
        b1_area = box1[2] * box1[3]
        b2_area = box2[2] * box2[3]
        union_area = b1_area + b2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0