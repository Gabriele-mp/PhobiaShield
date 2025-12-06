import cv2
import numpy as np
from typing import List, Dict

class Visualizer:
    def __init__(self):
        # OFFICIAL MAPPING FROM DATASET_FINAL_README.md
        # 0: Clown, 1: Shark, 2: Spider, 3: Blood, 4: Needle
        
        self.class_names = {
            0: "Clown",   # 🤡
            1: "Shark",   # 🦈
            2: "Spider",  # 🕷️
            3: "Blood",   # 🩸
            4: "Needle"   # 💉
        }

        # BGR colors for OpenCV
        self.colors = {
            0: (0, 165, 255),   # Clown -> Orange
            1: (192, 192, 192), # Shark -> Silver/grey
            2: (0, 0, 0),       # Spider -> Black
            3: (0, 0, 255),     # Blood -> Pure red
            4: (255, 255, 0),   # Needle -> cyan
            5: (255, 0, 255)    # Fallback -> magenta
        }

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        h, w, _ = frame.shape
        
        for det in detections:
            bbox = det['bbox']
            conf = det.get('confidence', 0.0)
            class_id = int(det['class_id'])
            
            # Convert coordinates
            cx, cy, bw, bh = bbox
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            # Clipping
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x1 >= x2 or y1 >= y2: continue

            # Specific color or default
            color = self.colors.get(class_id, self.colors[5])
            
            # Draw Box
            # If Spider (Black), add a thin white border for visibility
            if class_id == 2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Get class name
            label_text = self.class_names.get(class_id, f"ID {class_id}")
            label = f"{label_text} {conf:.0%}"
            
            # Label
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            
            # Text (White or Black depending on contrast)
            text_color = (255, 255, 255)
            if class_id in [1, 4]: # On light background (grey/cyan), use black text
                text_color = (0, 0, 0)
                
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            
        return frame