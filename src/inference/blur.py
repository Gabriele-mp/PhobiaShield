import cv2
import numpy as np

def apply_blur(frame: np.ndarray, bbox: list, intensity: int = 30) -> np.ndarray:
    """
    Applies a robust Gaussian Blur on the Region of Interest (ROI).
    bbox format: [cx, cy, w, h] normalized (0-1)
    """
    h, w, _ = frame.shape
    cx, cy, bw, bh = bbox
    
    # Convert from relative coordinates to pixels
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)
    
    # Safety clipping to avoid crashes if box exceeds image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # If box is invalid (e.g., width 0), return original frame
    if x1 >= x2 or y1 >= y2:
        return frame

    # Extract ROI
    roi = frame[y1:y2, x1:x2]
    
    # Apply Blur (Kernel size must be odd)
    k_size = (intensity*2 + 1, intensity*2 + 1)
    blurred_roi = cv2.GaussianBlur(roi, k_size, 30)
    
    # Reinsert ROI
    frame[y1:y2, x1:x2] = blurred_roi
    
    return frame