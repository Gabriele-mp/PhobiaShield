import sys
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from moviepy import VideoFileClip, AudioFileClip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.detector import PhobiaDetector

# --- CONFIGURAZIONE ---
PATH_YOLO = "weights/Yolov5s_raw_dataset.pt"

# --- NUOVO FILE ---
PATH_CUSTOM = "weights/fpn_ultra_e22_loss4.5031.pth" 
# ------------------

INPUT_VIDEO = "input_videos/final_test.mp4"
TEMP_VIDEO = "temp_video_mute.mp4"
OUTPUT_VIDEO = "confronto_finale_ULTRA.mp4" # Nome nuovo

def apply_blur_effect(frame, detections):
    img = frame.copy()
    h_img, w_img = img.shape[:2]

    if len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
            w_box = x2 - x1
            h_box = y2 - y1
            
            if w_box > 0 and h_box > 0:
                roi = img[y1:y2, x1:x2]
                try:
                    # Blur intenso
                    roi_blurred = cv2.GaussianBlur(roi, (99, 99), 30)
                    img[y1:y2, x1:x2] = roi_blurred
                except Exception:
                    pass
    return img

def main():
    if not os.path.exists(PATH_YOLO):
        print(f"[ERRORE] Manca YOLO in: {PATH_YOLO}")
        return
    if not os.path.exists(PATH_CUSTOM):
        print(f"[ERRORE] Manca FPN in: {PATH_CUSTOM}")
        return
    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERRORE] Manca il video in: {INPUT_VIDEO}")
        return

    print("--- GENERAZIONE VIDEO (MODELLO ULTRA) ---")
    
    det_yolo = PhobiaDetector(PATH_YOLO, model_type='yolo')
    det_custom = PhobiaDetector(PATH_CUSTOM, model_type='custom')

    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_width = width * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(TEMP_VIDEO, fourcc, fps, (out_width, height))
    
    if not out.isOpened(): return

    print(f"1/2: Elaborazione Grafica ({frames} frames)...")
    
    for _ in tqdm(range(frames)):
        ret, frame = cap.read()
        if not ret: break

        d_custom = det_custom.predict(frame)
        d_yolo = det_yolo.predict(frame)

        frame_custom = apply_blur_effect(frame, d_custom)
        frame_yolo = apply_blur_effect(frame, d_yolo)

        # TITOLI AGGIORNATI
        cv2.putText(frame_custom, "FPN ULTRA (Loss 4.5)", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_yolo, "YOLOv5 (Standard)", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        final_frame = cv2.hconcat([frame_custom, frame_yolo])
        cv2.line(final_frame, (width, 0), (width, height), (255, 255, 255), 6)

        out.write(final_frame)

    cap.release()
    out.release()

    print("2/2: Mix Audio...")
    try:
        original_clip = VideoFileClip(INPUT_VIDEO)
        if original_clip.audio is not None:
            generated_clip = VideoFileClip(TEMP_VIDEO)
            final_clip = generated_clip.with_audio(original_clip.audio)
            final_clip.write_videofile(OUTPUT_VIDEO, codec='libx264', audio_codec='aac')
            generated_clip.close()
        else:
            os.rename(TEMP_VIDEO, OUTPUT_VIDEO)
        original_clip.close()
        if os.path.exists(TEMP_VIDEO): os.remove(TEMP_VIDEO)
        print(f"\n[SUCCESS] VIDEO ULTRA PRONTO: {OUTPUT_VIDEO}")
    except Exception as e:
        print(f"Errore Audio: {e}")

if __name__ == "__main__":
    main()