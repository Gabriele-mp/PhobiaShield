import cv2
import time
import os
# Import custom modules
from src.inference.detector import PhobiaDetector
from src.inference.nms import nms
from src.inference.blur import apply_blur
from src.utils.visualization import Visualizer

class PhobiaVideoProcessor:
    def __init__(self, model_path=None, output_dir="outputs/videos"):
        self.output_dir = output_dir
        # Initialize real detector
        self.detector = PhobiaDetector(model_path=model_path)
        self.visualizer = Visualizer()
        
    def process_video(self, input_path, output_name="result.webm", conf_threshold=0.5, debug=True):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Setup video writer
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        save_path = os.path.join(self.output_dir, output_name)
        
        # WINDOWS ROBUSTNESS FIX
        # Using mp4v. It is the only codec guaranteeing file creation on Windows
        # without external DLLs. The browser might not play it, but the file is valid.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

        # Force .mp4 extension
        if not save_path.endswith(".mp4"):
             save_path = os.path.splitext(save_path)[0] + ".mp4"

        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("ERROR: VideoWriter failed to open even with mp4v.")
            return

        print(f"Processing started using REAL ENGINE...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # REAL INFERENCE (No simulation)
            # Pass frame to detector using PhobiaNet
            raw_detections = self.detector.detect(frame, conf_threshold=conf_threshold)
            
            # FILTERING (NMS)
            # Remove overlapping predictions
            clean_detections = nms(raw_detections, iou_threshold=0.4, conf_threshold=conf_threshold)
            
            # BLURRING & VISUALIZATION
            for det in clean_detections:
                # Apply real blur
                frame = apply_blur(frame, det['bbox'], intensity=15)
            
            if debug:
                frame = self.visualizer.draw_detections(frame, clean_detections)

            out.write(frame)

        cap.release()
        out.release()
        print(f"Done. Saved to {save_path}")