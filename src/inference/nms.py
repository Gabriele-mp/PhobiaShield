"""
Non-Maximum Suppression (NMS)

Responsabilità Membro C: Deployment & Demo Engineer

NMS is used to remove duplicate detections and keep only the best bounding box
for each object.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


def compute_iou_boxes(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes.
    
    Args:
        box1: [x, y, w, h] (center format)
        box2: [x, y, w, h] (center format)
    
    Returns:
        iou: Intersection over Union
    """
    # Convert to corner format
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou


def nms(
    detections: List[Dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.3
) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove duplicate detections.
    
    Algorithm:
    1. Filter detections by confidence threshold
    2. Sort detections by confidence (descending)
    3. For each detection:
        - Keep it if it doesn't overlap too much with already kept detections
        - Remove it otherwise
    
    Args:
        detections: List of detection dicts with keys:
                   - 'bbox': [x, y, w, h] (center format, normalized 0-1)
                   - 'confidence': float
                   - 'class_id': int
        iou_threshold: IoU threshold for suppression
        conf_threshold: Confidence threshold for filtering
    
    Returns:
        filtered_detections: List of kept detections after NMS
    """
    if not detections:
        return []
    
    # Filter by confidence threshold
    detections = [d for d in detections if d["confidence"] >= conf_threshold]
    
    if not detections:
        return []
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    
    # Group detections by class
    class_detections = {}
    for det in detections:
        class_id = det["class_id"]
        if class_id not in class_detections:
            class_detections[class_id] = []
        class_detections[class_id].append(det)
    
    # Apply NMS per class
    final_detections = []
    
    for class_id, class_dets in class_detections.items():
        kept_dets = []
        
        while class_dets:
            # Take the detection with highest confidence
            best_det = class_dets.pop(0)
            kept_dets.append(best_det)
            
            # Remove detections that overlap too much with best_det
            remaining_dets = []
            for det in class_dets:
                iou = compute_iou_boxes(
                    np.array(best_det["bbox"]),
                    np.array(det["bbox"])
                )
                
                if iou < iou_threshold:
                    # Keep this detection (low overlap)
                    remaining_dets.append(det)
                # else: suppress (high overlap)
            
            class_dets = remaining_dets
        
        final_detections.extend(kept_dets)
    
    return final_detections


def soft_nms(
    detections: List[Dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.3,
    sigma: float = 0.5,
) -> List[Dict]:
    """
    Apply Soft-NMS: Instead of removing overlapping boxes, reduce their confidence.
    
    This is a gentler version of NMS that can help when objects are very close.
    
    Args:
        detections: List of detection dicts
        iou_threshold: IoU threshold for suppression
        conf_threshold: Confidence threshold for filtering
        sigma: Gaussian sigma for confidence decay
    
    Returns:
        filtered_detections: List of kept detections after Soft-NMS
    """
    if not detections:
        return []
    
    # Filter by confidence threshold
    detections = [d for d in detections if d["confidence"] >= conf_threshold]
    
    if not detections:
        return []
    
    # Make a copy to avoid modifying original
    detections = [d.copy() for d in detections]
    
    # Group by class
    class_detections = {}
    for det in detections:
        class_id = det["class_id"]
        if class_id not in class_detections:
            class_detections[class_id] = []
        class_detections[class_id].append(det)
    
    final_detections = []
    
    for class_id, class_dets in class_detections.items():
        # Sort by confidence
        class_dets = sorted(class_dets, key=lambda x: x["confidence"], reverse=True)
        
        kept_dets = []
        
        while class_dets:
            # Take best detection
            best_det = class_dets.pop(0)
            kept_dets.append(best_det)
            
            # Decay confidence of overlapping detections
            for det in class_dets:
                iou = compute_iou_boxes(
                    np.array(best_det["bbox"]),
                    np.array(det["bbox"])
                )
                
                # Gaussian decay
                det["confidence"] *= np.exp(-(iou ** 2) / sigma)
            
            # Re-sort by updated confidence
            class_dets = sorted(class_dets, key=lambda x: x["confidence"], reverse=True)
            
            # Remove detections below threshold
            class_dets = [d for d in class_dets if d["confidence"] >= conf_threshold]
        
        final_detections.extend(kept_dets)
    
    return final_detections


def batch_nms(
    batch_detections: List[List[Dict]],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.3,
    method: str = "standard"
) -> List[List[Dict]]:
    """
    Apply NMS to a batch of detections.
    
    Args:
        batch_detections: List of detection lists (one per image)
        iou_threshold: IoU threshold
        conf_threshold: Confidence threshold
        method: "standard" or "soft"
    
    Returns:
        filtered_batch: List of filtered detection lists
    """
    nms_fn = soft_nms if method == "soft" else nms
    
    filtered_batch = []
    for detections in batch_detections:
        filtered = nms_fn(detections, iou_threshold, conf_threshold)
        filtered_batch.append(filtered)
    
    return filtered_batch


if __name__ == "__main__":
    # Test NMS
    print("Testing Non-Maximum Suppression")
    print("=" * 50)
    
    # Create dummy detections
    detections = [
        {"bbox": [0.5, 0.5, 0.2, 0.2], "confidence": 0.9, "class_id": 0},  # Spider
        {"bbox": [0.52, 0.52, 0.21, 0.19], "confidence": 0.85, "class_id": 0},  # Overlapping spider
        {"bbox": [0.3, 0.3, 0.15, 0.15], "confidence": 0.7, "class_id": 0},  # Another spider
        {"bbox": [0.8, 0.8, 0.25, 0.25], "confidence": 0.95, "class_id": 1},  # Snake
        {"bbox": [0.82, 0.82, 0.23, 0.23], "confidence": 0.6, "class_id": 1},  # Overlapping snake
    ]
    
    print(f"Original detections: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  {i+1}. Class {det['class_id']}, Conf: {det['confidence']:.2f}, Box: {det['bbox']}")
    
    # Apply standard NMS
    print("\nApplying Standard NMS (IoU=0.5, Conf=0.3):")
    filtered = nms(detections, iou_threshold=0.5, conf_threshold=0.3)
    print(f"Filtered detections: {len(filtered)}")
    for i, det in enumerate(filtered):
        print(f"  {i+1}. Class {det['class_id']}, Conf: {det['confidence']:.2f}, Box: {det['bbox']}")
    
    # Apply soft NMS
    print("\nApplying Soft-NMS (IoU=0.5, Conf=0.3, Sigma=0.5):")
    filtered_soft = soft_nms(detections, iou_threshold=0.5, conf_threshold=0.3, sigma=0.5)
    print(f"Filtered detections: {len(filtered_soft)}")
    for i, det in enumerate(filtered_soft):
        print(f"  {i+1}. Class {det['class_id']}, Conf: {det['confidence']:.2f}, Box: {det['bbox']}")
    
    # Test IoU
    print("\nTesting IoU computation:")
    box1 = np.array([0.5, 0.5, 0.4, 0.4])
    box2 = np.array([0.6, 0.6, 0.4, 0.4])
    iou = compute_iou_boxes(box1, box2)
    print(f"IoU between overlapping boxes: {iou:.4f}")
    
    box3 = np.array([0.2, 0.2, 0.3, 0.3])
    iou2 = compute_iou_boxes(box1, box3)
    print(f"IoU between non-overlapping boxes: {iou2:.4f}")
