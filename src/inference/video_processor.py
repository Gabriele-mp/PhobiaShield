import cv2
import numpy as np
import time
import sys
import os
from typing import List, Dict, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# import NMS
try:
    from src.inference.nms import nms
    print("NMS module loaded")
except ImportError as e:
    nms = None
    print(f"Error: NMS not found. Reason: {e}")


class PhobiaVideoProcessor:
    
    def __init__(self, model=None, output_dir: str = "outputs/videos"):
        self.model = model
        self.output_dir = output_dir
        
        self.blur_kernel_size = (51, 51) 
        self.blur_sigma = 30 
        
        # Class Map
        self.class_names = {
            0: "Clown",
            1: "Shark",
            2: "Spider", 
            3: "Blood",
            4: "Needle",                                  
        }

        print(f"VideoProcessor initialised. Convolution Kernel: {self.blur_kernel_size}")

    def apply_convolutional_blur(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        
        h, w, _ = frame.shape
        
        # bbox input: [center_x, center_y, width, height] (Normalised YOLO format 0-1)
        cx, cy, bw, bh = bbox
        
        # Mapping in pixel space (De-normalisation)
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        # Clipping to ensure that the coordinates are within the image set
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, h)

        # Extraction of ROI (Sub-matrix)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return frame

        # CONVOLUTION APPLICATION (Blur)
        # Mathematically: (f * g)[n] = sum(f[m]g[n-m])
        blurred_roi = cv2.GaussianBlur(roi, self.blur_kernel_size, self.blur_sigma)
        
        # Reinsertion of the transformed matrix
        frame[y1:y2, x1:x2] = blurred_roi
        return frame

    def generate_monte_carlo_detections(self) -> List[Dict]:
        detections = []
        num_objects = np.random.randint(5, 15)
        
        for _ in range(num_objects):
            det = {
                "bbox": [
                    np.random.uniform(0.1, 0.9),
                    np.random.uniform(0.1, 0.9),
                    np.random.uniform(0.05, 0.2),
                    np.random.uniform(0.05, 0.2)
                ],
                "confidence": np.random.uniform(0.4, 0.99),
                "class_id": np.random.randint(0, 5)
            }
            detections.append(det)
        return detections

    def process_video(self, input_path: str, output_name: str = "output.mp4", simulate: bool = False, debug: bool = True):

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Unable to open {input_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        save_path = f"{self.output_dir}/{output_name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        print(f"Processing: {input_path} | Debug Mode: {debug}")
        
        # NMS configuration
        nms_iou_thresh = 0.45
        nms_conf_thresh = 0.25

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            detections = []
            
            if simulate:
                detections = self.generate_monte_carlo_detections()
            elif self.model:
                pass

            # FILTERING (Use Soft-NMS if available to manage overlaps)
            if nms and detections:
                try:
                    from src.inference.nms import soft_nms
                    detections = soft_nms(detections, iou_threshold=nms_iou_thresh, conf_threshold=nms_conf_thresh, sigma=0.5)
                except ImportError:
                    detections = nms(detections, iou_threshold=nms_iou_thresh, conf_threshold=nms_conf_thresh)

            # TRANSFORMATION & DEBUGGING
            for det in detections:
                bbox = det['bbox']
                
                # Apply Blur (Censorship)
                frame = self.apply_convolutional_blur(frame, bbox)
                
                # Debug Mode
                if debug:
                    h, w, _ = frame.shape
                    cx, cy, bw, bh = bbox
                    
                    # Coordinate denormalisation (from 0-1 to pixels)
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    # Safety clipping
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)


                    class_name = self.class_names.get(det['class_id'], f"Class {det['class_id']}")
                    
                    label = f"{class_name} {det['confidence']:.0%}"
                    
                    # BGR Format
                    cid = det['class_id']
                    
                    if cid == 2:
                        # Class 2
                        color = (0, 0, 255)
                        thickness = 2
                        
                    elif cid in [0, 1]:
                        # Class 0, 1
                        color = (74, 223, 45)
                        thickness = 2
                        
                    elif cid in [3, 4]:
                        # Class 3, 4
                        color = (182, 89, 170) 
                        thickness = 2
                        
                    else:
                        # Default for errors
                        color = (128, 128, 128)
                        thickness = 2
                    
                    # Draw the rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw the background for the text
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), color, -1)
                    
                    cv2.putText(frame, label, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out.write(frame)

        cap.release()
        out.release()
        print(f"Finished. Check {save_path}")

# Unit Test
if __name__ == "__main__":
    import os
    
    if not os.path.exists("outputs/videos"):
        os.makedirs("outputs/videos")
        
    processor = PhobiaVideoProcessor()
    
    video_test = "data_workspace/assets/test_video.mp4" 
    
    if os.path.exists(video_test):
        # Start in simulation mode (Monte Carlo)
        print("Starting Monte Carlo test...")
        processor.process_video(video_test, "test_result_montecarlo.mp4", simulate=True)
    else:
        print(f"Attention: File {video_test} not found")