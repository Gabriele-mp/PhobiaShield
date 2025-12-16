import sys
import os
import cv2
import time
from tqdm import tqdm

# Setup path per importare i moduli src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.detector import PhobiaDetector

# --- CONFIGURAZIONE ---
MODEL_PATH = "weights/Yolov5s_raw_dataset.pt"  # Il file YOLO
INPUT_VIDEO = "input_videos/test_video.mp4"      # Il tuo video di test
OUTPUT_VIDEO = "yolo_test_result.mp4"            # Dove salverà il video

def main():
    # 1. Controllo File
    if not os.path.exists(MODEL_PATH):
        print(f"[ERRORE] Non trovo il modello: {MODEL_PATH}")
        return
    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERRORE] Non trovo il video: {INPUT_VIDEO}")
        return

    print(f"--- TEST STANDALONE YOLO ---")
    print(f"Modello: {MODEL_PATH}")
    
    # 2. Caricamento Modello
    try:
        # Nota: model_type='yolo' è fondamentale qui
        detector = PhobiaDetector(model_path=MODEL_PATH, model_type='yolo')
    except Exception as e:
        print(f"[FATAL] Errore caricamento YOLO: {e}")
        return

    # 3. Setup Video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"Inizio elaborazione video ({total_frames} frames)...")
    
    # 4. Loop Inferenza
    start_time = time.time()
    
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # PREDIZIONE
        detections = detector.predict(frame)

        # DISEGNO
        if len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                # Rettangolo Rosso per YOLO
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                # Label
                label = f"YOLO: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        out.write(frame)

    end_time = time.time()
    elapsed = end_time - start_time
    fps_avg = total_frames / elapsed

    cap.release()
    out.release()
    
    print(f"\n[SUCCESS] Video salvato: {OUTPUT_VIDEO}")
    print(f"Tempo totale: {elapsed:.2f}s | FPS Medi: {fps_avg:.2f}")

if __name__ == "__main__":
    main()