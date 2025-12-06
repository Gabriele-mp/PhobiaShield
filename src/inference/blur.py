import cv2
import numpy as np

def apply_blur(frame: np.ndarray, bbox: list, intensity: int = 30) -> np.ndarray:
    """
    Applica un Gaussian Blur robusto sulla Region of Interest (ROI).
    bbox format: [cx, cy, w, h] normalized (0-1)
    """
    h, w, _ = frame.shape
    cx, cy, bw, bh = bbox
    
    # Conversione da coordinate relative a pixel
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)
    
    # Clipping per evitare crash se la box esce dall'immagine
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Se la box è invalida (es. larghezza 0), ritorna frame originale
    if x1 >= x2 or y1 >= y2:
        return frame

    # Estrai ROI
    roi = frame[y1:y2, x1:x2]
    
    # Applica Blur (Kernel deve essere dispari)
    k_size = (intensity*2 + 1, intensity*2 + 1)
    blurred_roi = cv2.GaussianBlur(roi, k_size, 30)
    
    # Reinserisci ROI
    frame[y1:y2, x1:x2] = blurred_roi
    
    return frame