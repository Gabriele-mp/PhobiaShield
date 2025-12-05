import cv2
import numpy as np
from typing import List, Dict, Tuple

class Visualizer:
    # Manages all drawing and annotation operations on frames
    
    def __init__(self):
        self.colors = {
            0: (255, 165, 0),   # Spider
            1: (0, 255, 0),     # Shark
            2: (255, 0, 255),   # Clown
            3: (0, 0, 255),     # Blood
            4: (0, 0, 255),     # Needle
            5: (128, 128, 128)  # Default
        }
        
        self.class_names = {
            0: "Clown",
            1: "Shark",
            2: "Spider",
            3: "Blood",
            4: "Needle"
        }

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        
        h, w, _ = frame.shape
        
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            cls_id = int(det['class_id'])
            
            # Coordinates (from YOLO 0-1 to Pixel)
            cx, cy, bw, bh = bbox
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            # Safety clipping
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Colour Selection
            color = self.colors.get(cls_id, self.colors[5])
            
            # Box design
            # If the object is large (>30% width), thicker border
            thickness = 2
            if (x2 - x1) > w * 0.3: thickness = 3
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            label = f"{self.class_names.get(cls_id, str(cls_id))} {conf:.0%}"
            
            # Calculate text size
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)
            
            
            cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
        return frame