import cv2
import numpy as np
import time
import sys
import os
from typing import List, Dict, Optional


# Allows running the script from any location without import errors
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# Import internal modules with error handling
try:
    from src.inference.nms import nms, soft_nms
    print("NMS module loaded")
except ImportError:
    nms = None
    print("Warning: NMS module not found. Filtering disabled")

try:
    from src.utils.visualization import Visualizer
    print("Visualization module loaded")
except ImportError:
    Visualizer = None
    print("Warning: Visualization module not found. Basic graphical debug only")

class PhobiaVideoProcessor:
    
    def __init__(self, model=None, output_dir: str = "outputs/videos"):
        self.model = model
        self.output_dir = output_dir
        
        # Blur parameters
        self.blur_kernel_size = (51, 51) 
        self.blur_sigma = 30 
        
        # Initialize external visualizer (Separation of concerns)
        self.visualizer = Visualizer() if Visualizer else None
        
        print(f"VideoProcessor initialized. Kernel: {self.blur_kernel_size}")

    def apply_convolutional_blur(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        h, w, _ = frame.shape
        cx, cy, bw, bh = bbox
        
        # De-normalization and clipping
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return frame

        # Convolution
        blurred_roi = cv2.GaussianBlur(roi, self.blur_kernel_size, self.blur_sigma)
        frame[y1:y2, x1:x2] = blurred_roi
        return frame

    def generate_monte_carlo_detections(self) -> List[Dict]:
        # Stochastic simulation to test the pipeline without a model
        detections = []
        num_objects = np.random.randint(5, 15)
        
        for _ in range(num_objects):
            det = {
                "bbox": [
                    np.random.uniform(0.1, 0.9), # x
                    np.random.uniform(0.1, 0.9), # y
                    np.random.uniform(0.05, 0.2), # w
                    np.random.uniform(0.05, 0.2)  # h
                ],
                "confidence": np.random.uniform(0.4, 0.99),
                "class_id": np.random.randint(0, 5) # 5 classes
            }
            detections.append(det)
        return detections

    def process_video(self, input_path: str, output_name: str = "output.mp4", simulate: bool = False, debug: bool = True):
        
        if not os.path.exists(input_path):
            print(f"Error: Video file does not exist: {input_path}")
            return

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video stream: {input_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, output_name)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        print(f"Processing started: {input_path} -> {save_path}")
        print(f"Configuration: Simulate={simulate}, Debug={debug}")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            detections = []
            
            # INFERENCE / SIMULATION
            if simulate:
                detections = self.generate_monte_carlo_detections()
            elif self.model:
                # TODO: Future integration with real model
                pass

            # FILTERING (NMS)
            if nms and detections:
                # Use Soft-NMS if available (better for close objects)
                try:
                    detections = soft_nms(detections, iou_threshold=0.45, conf_threshold=0.25, sigma=0.5)
                except (NameError, ImportError):
                    detections = nms(detections, iou_threshold=0.45, conf_threshold=0.25)

            # TRANSFORMATION (Blur)
            for det in detections:
                frame = self.apply_convolutional_blur(frame, det['bbox'])
            
            # VISUALIZATION (Delegated)
            if debug and self.visualizer:
                frame = self.visualizer.draw_detections(frame, detections)

            out.write(frame)
            frame_idx += 1
            if frame_idx % 60 == 0: print(f"Processing frame {frame_idx}...")

        cap.release()
        out.release()
        print(f"Completed. Video saved in: {save_path}")

# UNIT TEST
if __name__ == "__main__":
    
    processor = PhobiaVideoProcessor()
    
    test_video = "assets/test_video.mp4"
    if not os.path.exists(test_video):
        test_video = "data_workspace/assets/test_video.mp4"
    
    processor.process_video(test_video, "test_result_final.mp4", simulate=True)