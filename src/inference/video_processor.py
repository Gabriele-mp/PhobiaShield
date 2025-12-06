import cv2
import time
import os
# Importa i tuoi nuovi moduli
from src.inference.detector import PhobiaDetector
from src.inference.nms import nms # [cite: 371]
from src.inference.blur import apply_blur
from src.utils.visualization import Visualizer

class PhobiaVideoProcessor:
    def __init__(self, model_path=None, output_dir="outputs/videos"):
        self.output_dir = output_dir
        # Inizializza il Detector REALE
        self.detector = PhobiaDetector(model_path=model_path)
        self.visualizer = Visualizer()
        
    def process_video(self, input_path, output_name="result.webm", conf_threshold=0.5, debug=True):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Setup Video Writer (come avevi fatto prima, va bene)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        save_path = os.path.join(self.output_dir, output_name)
        
        # --- FIX ROBUSTEZZA WINDOWS ---
        # Torniamo a mp4v. È l'unico che garantisce la creazione del file su Windows
        # senza librerie esterne.
        # Il browser potrebbe non riprodurlo, ma il file sarà valido.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

        # Forza estensione .mp4
        if not save_path.endswith(".mp4"):
             save_path = os.path.splitext(save_path)[0] + ".mp4"

        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("❌ ERRORE CRITICO: Il VideoWriter non si è aperto nemmeno con mp4v.")
            return

        print(f"🎬 Processing started using REAL ENGINE...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 1. INFERENCE REALE (Nessuna simulazione!)
            # Passiamo il frame al detector che usa PhobiaNet
            raw_detections = self.detector.detect(frame, conf_threshold=0.1) # Soglia bassa per test
            
            # 2. FILTERING (NMS)
            # Pulisci le predizioni sovrapposte
            clean_detections = nms(raw_detections, iou_threshold=0.4, conf_threshold=conf_threshold)
            
            # 3. BLURRING & VISUALIZATION
            for det in clean_detections:
                # Applica blur reale
                frame = apply_blur(frame, det['bbox'], intensity=15)
            
            if debug:
                frame = self.visualizer.draw_detections(frame, clean_detections)

            out.write(frame)

        cap.release()
        out.release()
        print(f"✅ Done. Saved to {save_path}")