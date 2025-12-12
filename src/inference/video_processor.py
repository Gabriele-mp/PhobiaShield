import cv2
import time
import os
import numpy as np
from src.inference.detector import PhobiaDetector
from src.inference.nms import nms
from src.inference.blur import apply_blur
from src.utils.visualization import Visualizer

class PhobiaVideoProcessor:
    def __init__(self, model_path=None, output_dir="outputs/videos"):
        self.output_dir = output_dir
        self.detector = PhobiaDetector(model_path=model_path)
        self.visualizer = Visualizer()
        
    def process_video(self, input_path, output_name="result.webm", conf_threshold=0.3, debug=True):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # --- DEBUG CONFIGURATION ---
        # Creiamo percorso assoluto per evitare dubbi
        abs_output_dir = os.path.abspath(self.output_dir)
        frames_dir = os.path.join(abs_output_dir, "debug_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"🎬 Processing started...")
        print(f"   📂 SAVING TO ABSOLUTE PATH: {frames_dir}")
        
        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Check validità frame
                if frame is None or frame.size == 0:
                    print(f"❌ Frame {frame_count} is empty/invalid!")
                    continue

                if frame_count == 0:
                    print(f"   ℹ️ Source Resolution: {frame.shape}")
                
                # 1. INFERENCE
                try:
                    raw_detections = self.detector.detect(frame, conf_threshold=conf_threshold)
                    clean_detections = nms(raw_detections, iou_threshold=0.3, conf_threshold=conf_threshold)
                    
                    # Log detections del primo frame
                    if frame_count == 0:
                        print(f"   🔎 Frame 0 Detections found: {len(clean_detections)}")

                    for det in clean_detections:
                        frame = apply_blur(frame, det['bbox'], intensity=15)
                    
                    if debug:
                        frame = self.visualizer.draw_detections(frame, clean_detections)
                        
                except Exception as e_inf:
                    print(f"❌ Error during inference on frame {frame_count}: {e_inf}")
                    # Continuiamo per provare a salvare almeno il frame originale
                    pass

                # 2. SALVATAGGIO CON VERIFICA
                frame_name = f"frame_{frame_count:04d}.jpg"
                save_path = os.path.join(frames_dir, frame_name)
                
                success = cv2.imwrite(save_path, frame)
                
                if not success:
                    print(f"❌ FAILED to write: {save_path}")
                    # Se fallisce, proviamo nella cartella corrente per test
                    fallback = f"test_frame_{frame_count}.jpg"
                    cv2.imwrite(fallback, frame)
                    print(f"   -> Tried fallback save to local folder: {fallback}")
                
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"   Processed {frame_count} frames...", end='\r')
                    
                if frame_count >= 50: # Riduciamo a 50 per fare prima
                    print("\n🛑 Test limit reached (50 frames). Stopping.")
                    break

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted.")
        except Exception as e:
            print(f"\n❌ Global Error: {e}")
            
        cap.release()
        print(f"\n✅ Done. Go to this folder now:")
        print(f"👉 {frames_dir}")