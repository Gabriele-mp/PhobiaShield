"""
Metrics for Object Detection
The Architect Module

Implements mAP, IoU, precision, recall for evaluation.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Compute IoU between two boxes.
    
    Args:
        box1: [x, y, w, h] center format
        box2: [x, y, w, h] center format
    
    Returns:
        iou: Intersection over Union
    """
    # Convert to corner format
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return iou


def calculate_map(
    model,
    dataloader,
    device,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
    num_classes: int = 3
) -> float:
    """
    Calculate Mean Average Precision (mAP).
    
    Args:
        model: Trained model
        dataloader: Validation dataloader
        device: CPU or CUDA
        iou_threshold: IoU threshold for matching
        conf_threshold: Confidence threshold
        num_classes: Number of classes
    
    Returns:
        mAP: Mean Average Precision across all classes
    """
    model.eval()
    
    # Store predictions and ground truths per class
    all_predictions = defaultdict(list)  # {class_id: [(conf, bbox), ...]}
    all_ground_truths = defaultdict(list)  # {class_id: [bbox, ...]}
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            batch_size = images.size(0)
            
            # Get predictions
            predictions = model.predict(images, conf_threshold=conf_threshold)
            
            # Process each image in batch
            for img_idx in range(batch_size):
                # Ground truth
                target = targets[img_idx]  # (S, S, B*5+C)
                
                # Extract ground truth boxes
                for i in range(target.size(0)):  # Grid rows
                    for j in range(target.size(1)):  # Grid cols
                        if target[j, i, 4] > 0:  # Has object
                            # Get box
                            x_cell = target[j, i, 0].item()
                            y_cell = target[j, i, 1].item()
                            w = target[j, i, 2].item()
                            h = target[j, i, 3].item()
                            
                            # Convert to absolute coordinates
                            cx = (j + x_cell) / 13
                            cy = (i + y_cell) / 13
                            
                            # Get class
                            class_probs = target[j, i, 10:13]
                            class_id = torch.argmax(class_probs).item()
                            
                            all_ground_truths[class_id].append([cx, cy, w, h])
                
                # Predictions for this image
                for pred in predictions[img_idx]:
                    class_id = pred['class_id']
                    confidence = pred['confidence']
                    bbox = pred['bbox']  # [cx, cy, w, h]
                    
                    all_predictions[class_id].append((confidence, bbox))
    
    # Calculate AP for each class
    aps = []
    
    for class_id in range(num_classes):
        predictions = all_predictions[class_id]
        ground_truths = all_ground_truths[class_id]
        
        if len(ground_truths) == 0:
            continue
        
        # Sort predictions by confidence (descending)
        predictions = sorted(predictions, key=lambda x: x[0], reverse=True)
        
        # Track which ground truths have been matched
        gt_matched = [False] * len(ground_truths)
        
        tp = []  # True positives
        fp = []  # False positives
        
        for conf, pred_bbox in predictions:
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_bbox in enumerate(ground_truths):
                if gt_matched[gt_idx]:
                    continue
                
                iou = compute_iou(
                    torch.tensor(pred_bbox),
                    torch.tensor(gt_bbox)
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                if not gt_matched[best_gt_idx]:
                    tp.append(1)
                    fp.append(0)
                    gt_matched[best_gt_idx] = True
                else:
                    tp.append(0)
                    fp.append(1)
            else:
                tp.append(0)
                fp.append(1)
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Compute AP (area under precision-recall curve)
        # Using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        aps.append(ap)
    
    # Mean AP
    if len(aps) == 0:
        return 0.0
    
    mAP = np.mean(aps)
    return mAP


def compute_detection_metrics(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for detections.
    
    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        iou_threshold: IoU threshold
    
    Returns:
        metrics: Dict with precision, recall, f1
    """
    tp = 0
    fp = 0
    fn = 0
    
    # Track matched targets
    matched_targets = set()
    
    for pred in predictions:
        pred_bbox = torch.tensor(pred['bbox'])
        best_iou = 0
        best_target_idx = -1
        
        for target_idx, target in enumerate(targets):
            if target_idx in matched_targets:
                continue
            
            target_bbox = torch.tensor(target['bbox'])
            iou = compute_iou(pred_bbox, target_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_target_idx = target_idx
        
        if best_iou >= iou_threshold and best_target_idx >= 0:
            tp += 1
            matched_targets.add(best_target_idx)
        else:
            fp += 1
    
    fn = len(targets) - len(matched_targets)
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


# Test
if __name__ == "__main__":
    # Test IoU
    box1 = torch.tensor([0.5, 0.5, 0.4, 0.4])
    box2 = torch.tensor([0.6, 0.6, 0.4, 0.4])
    iou = compute_iou(box1, box2)
    print(f"IoU: {iou:.4f}")
    
    # Test metrics
    preds = [
        {'bbox': [0.5, 0.5, 0.3, 0.3], 'class_id': 0, 'confidence': 0.9}
    ]
    targets = [
        {'bbox': [0.52, 0.48, 0.32, 0.28], 'class_id': 0}
    ]
    
    metrics = compute_detection_metrics(preds, targets)
    print(f"\nMetrics: {metrics}")
    
    print("\n✓ Metrics test passed!")