import sys
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from moviepy import VideoFileClip, AudioFileClip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.detector import PhobiaDetector

# CONFIGURATION
PATH_YOLO = "weights/yolov8s_best.pt"
PATH_CUSTOM = "weights/fpn_ultra_e22_loss4.5031.pth" 

INPUT_VIDEO = "input_videos/final_test.mp4"
TEMP_VIDEO = "temp_video_mute.mp4"
OUTPUT_VIDEO = "final_comparison_YOLOv8_vs_FPN.mp4"

def apply_blur_effect(frame, detections):
    """
    Applies Gaussian Blur to detected areas and adds a thin red border.
    """
    img = frame.copy()
    h_img, w_img = img.shape[:2]

    if len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
            w_box = x2 - x1
            h_box = y2 - y1
            
            if w_box > 0 and h_box > 0:
                roi = img[y1:y2, x1:x2]
                try:
                    # Intense Blur (99x99 kernel)
                    roi_blurred = cv2.GaussianBlur(roi, (99, 99), 30)
                    img[y1:y2, x1:x2] = roi_blurred
                    
                    # Thin Red Border (BGR format: Blue=0, Green=0, Red=255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                except Exception:
                    pass
    return img

def main():
    # Verify file existence
    if not os.path.exists(PATH_YOLO):
        print(f"[ERROR] YOLOv8 missing at: {PATH_YOLO}")
        return
    if not os.path.exists(PATH_CUSTOM):
        print(f"[ERROR] FPN model missing at: {PATH_CUSTOM}")
        return
    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERROR] Input video missing at: {INPUT_VIDEO}")
        return

    print("--- VIDEO GENERATION: YOLOv8 (0.35) vs FPN (0.55) ---")
    
    # Initialize Detectors
    det_yolo = PhobiaDetector(PATH_YOLO, model_type='yolo')
    det_custom = PhobiaDetector(PATH_CUSTOM, model_type='custom')

    # Video setup
    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_width = width * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(TEMP_VIDEO, fourcc, fps, (out_width, height))
    
    if not out.isOpened(): return

    print(f"1/2: Processing Graphics ({frames} frames)...")
    
    for _ in tqdm(range(frames)):
        ret, frame = cap.read()
        if not ret: break

        # Inference
        d_custom = det_custom.predict(frame)
        d_yolo = det_yolo.predict(frame)

        # Apply Effects
        frame_custom = apply_blur_effect(frame, d_custom)
        frame_yolo = apply_blur_effect(frame, d_yolo)

        # Add Titles (showing configured confidence thresholds)
        cv2.putText(frame_custom, "FPN ULTRA (Conf 0.55)", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_yolo, "YOLOv8s (Conf 0.35)", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Merge and Draw Split Line
        final_frame = cv2.hconcat([frame_custom, frame_yolo])
        cv2.line(final_frame, (width, 0), (width, height), (255, 255, 255), 6)

        out.write(final_frame)

    cap.release()
    out.release()

    # Audio Mixing
    print("2/2: Mixing Audio...")
    try:
        original_clip = VideoFileClip(INPUT_VIDEO)
        if original_clip.audio is not None:
            generated_clip = VideoFileClip(TEMP_VIDEO)
            final_clip = generated_clip.with_audio(original_clip.audio)
            # Using libx264 codec for max compatibility
            final_clip.write_videofile(OUTPUT_VIDEO, codec='libx264', audio_codec='aac')
            generated_clip.close()
        else:
            os.rename(TEMP_VIDEO, OUTPUT_VIDEO)
        original_clip.close()
        if os.path.exists(TEMP_VIDEO): os.remove(TEMP_VIDEO)
        print(f"\n[SUCCESS] FINAL VIDEO READY: {OUTPUT_VIDEO}")
    except Exception as e:
        print(f"Audio Error: {e}")

if __name__ == "__main__":
    main()