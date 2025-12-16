import sys
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.detector import PhobiaDetector

# --- CONFIGURAZIONE ---
MODEL_PATH = "weights/FPN_epoch18_loss6.3037.pth"
INPUT_FOLDER = "test_images_input"
OUTPUT_FOLDER = "test_images_output"

# Mappa Classi
CLASS_MAP = {
    "clown": 0,
    "shark": 1,
    "spider": 2,
    "blood": 3,
    "needle": 4
}

COLORS = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERRORE] Modello mancante: {MODEL_PATH}")
        return
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    image_files = glob.glob(os.path.join(INPUT_FOLDER, "*.*"))
    print(f"--- TEST CALIBRATO (Soglia 0.15 | NMS 0.20) ---")
    
    try:
        detector = PhobiaDetector(model_path=MODEL_PATH, model_type='custom')
    except Exception as e:
        print(f"[FATAL] Errore init detector: {e}")
        return

    for img_path in tqdm(image_files):
        filename = os.path.basename(img_path).lower()
        img = cv2.imread(img_path)
        if img is None: continue

        target_class_id = -1
        target_name = "Sconosciuto"
        for name, id_cls in CLASS_MAP.items():
            if name in filename:
                target_class_id = id_cls
                target_name = name
                break
        
        filter_active = (target_class_id != -1)

        detections = detector.predict(img)

        found_correct = 0
        rejected_classes = [] # Per diagnostica

        if len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                cls = int(cls)
                
                # Se filtro attivo e classe sbagliata, segna e salta
                if filter_active and cls != target_class_id:
                    rejected_classes.append(cls)
                    continue

                found_correct += 1
                color = COLORS[cls] if cls < len(COLORS) else (0, 255, 0)
                
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                label_text = f"{target_name.upper()}: {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (int(x1), int(y1)-25), (int(x1)+w, int(y1)), color, -1)
                cv2.putText(img, label_text, (int(x1), int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # REPORT DIAGNOSTICO NEL TERMINALE
        msg = f" > {filename}: Cercavo ID {target_class_id}. Trovati: {found_correct}."
        if rejected_classes:
            # Conta le classi scartate
            unique, counts = np.unique(rejected_classes, return_counts=True)
            discard_info = dict(zip(unique, counts))
            msg += f" [SCARTATI: {discard_info}]"
        
        print(msg)

        save_path = os.path.join(OUTPUT_FOLDER, "filtered_" + filename)
        cv2.imwrite(save_path, img)

    print(f"\n[SUCCESS] Immagini salvate in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()