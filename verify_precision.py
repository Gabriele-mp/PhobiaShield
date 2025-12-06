import cv2
import numpy as np
import os
from src.utils.visualization import Visualizer
from src.inference.nms import nms
from src.inference.blur import apply_blur

def test_precision():
    print("STARTING CALIBRATION PROTOCOL...")
    
    # Create a black frame 1000x1000
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    
    # DATA INJECTION
    # Simulate what would come out of the detector if it were perfect.
    # Expected format: [cx, cy, w, h] normalized (0.0 - 1.0)
    
    fake_detections = [
        # CASE A: Perfect box at center (Spider)
        # Center (0.5, 0.5), Width 0.2 (200px), Height 0.2 (200px)
        # Expecting box from x=400 to x=600, y=400 to y=600
        {'class_id': 2, 'confidence': 0.95, 'bbox': [0.5, 0.5, 0.2, 0.2]},
        
        # CASE B: NMS Test (Two almost identical overlapping boxes)
        # If NMS works, ONLY ONE must remain.
        {'class_id': 3, 'confidence': 0.90, 'bbox': [0.2, 0.2, 0.1, 0.1]}, # Strong Blood
        {'class_id': 3, 'confidence': 0.85, 'bbox': [0.21, 0.21, 0.1, 0.1]}, # Weak Blood (to be removed)
        
        # CASE C: Border Test (Box partially outside)
        # Center at bottom right corner (1.0, 1.0)
        {'class_id': 0, 'confidence': 0.88, 'bbox': [1.0, 1.0, 0.2, 0.2]},
    ]
    
    print(f"Input: {len(fake_detections)} raw predictions injected.")
    
    # NMS TEST
    # If nms.py works, 'clean_detections' must have length 3 (not 4).
    clean_detections = nms(fake_detections, iou_threshold=0.5, conf_threshold=0.5)
    
    print(f"NMS Output: {len(clean_detections)} remaining predictions.")
    
    if len(clean_detections) == 3:
        print("NMS TEST: PASSED (Duplicate removed).")
    else:
        print(f"NMS TEST: FAILED. Expected 3 boxes, found {len(clean_detections)}.")
        
    # BLURRING AND VISUALIZATION TEST
    vis = Visualizer()
    
    for det in clean_detections:
        # Apply Blur
        frame = apply_blur(frame, det['bbox'], intensity=15)
        
    # Draw
    frame = vis.draw_detections(frame, clean_detections)
    
    # SAVE FOR VISUAL INSPECTION
    cv2.imwrite("test_calibration_result.jpg", frame)
    print("Image saved: 'test_calibration_result.jpg'")
    
    # MATHEMATICAL COORDINATE CHECK (Pixel check)
    # Check if center pixel (500,500) is inside a black box (Spider ID 2)
    center_pixel = frame[500, 500]
    print(f"Center Pixel (BGR): {center_pixel}")
    
    if np.any(center_pixel > 0): # If not pure black, we drew something
        print("COORDINATE TEST: PASSED (Drawing detected at center).")
    else:
        print("COORDINATE TEST: CHECK VISUALLY (Center is empty).")

if __name__ == "__main__":
    test_precision()