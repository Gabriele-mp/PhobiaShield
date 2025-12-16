import cv2
import os
import sys
from src.inference.detector import PhobiaDetector
from src.inference.nms import nms

def test_platinum_logic():
    print("💎 STARTING PLATINUM VERIFICATION (Auto-Thresholds)...")
    
    # 1. FILE PTH NUOVO
    weights_path = "outputs/checkpoints/FPN_epoch18_loss6.3037.pth"
    
    if not os.path.exists(weights_path):
        print(f"❌ Pesi non trovati: {weights_path}")
        return

    # 2. CARICA MODELLO
    # Nota: Caricherà PhobiaNetFPN_v2 grazie alla modifica fatta al passo 1
    print(f"🧠 Caricamento Modello {weights_path}...")
    try:
        detector = PhobiaDetector(model_path=weights_path)
    except Exception as e:
        print(f"❌ Errore caricamento: {e}")
        return

    class_names = {0: "Clown", 1: "Shark", 2: "Spider", 3: "Blood", 4: "Needle"}
    test_images = {
        0: "clown_test.jpg",
        1: "shark_test.jpg",
        2: "spider_test.jpg",
        3: "blood_test.jpg",
        4: "needle_test.jpg"
    }

    print("-" * 75)
    print(f"{'IMAGE':<15} | {'EXPECTED':<10} | {'PREDICTION (V10 FILTER)':<25} | {'STATUS'}")
    print("-" * 75)

    for class_id, img_name in test_images.items():
        if not os.path.exists(img_name): continue
        expected_class = class_names[class_id]
        
        img = cv2.imread(img_name)
        
        # --- IL SEGRETO È QUI ---
        # Passiamo conf_threshold=None.
        # Questo dice al detector: "Usa le tue soglie Platinum interne (0.32, 0.35...)"
        raw_detections = detector.detect(img, conf_threshold=None)
        
        # NMS Stretto (0.10) come richiesto dal compagno
        clean_detections = nms(raw_detections, iou_threshold=0.10)
        
        # Analisi
        if len(clean_detections) == 0:
            pred_str = "Cleaned (No Detection)"
            status = "⚠️ BLIND / FILTERED"
        else:
            best_det = max(clean_detections, key=lambda x: x['confidence'])
            p_id = int(best_det['class_id'])
            p_name = class_names.get(p_id, "?")
            conf = best_det['confidence']
            pred_str = f"{p_name} ({conf:.1%})"
            
            if p_id == class_id:
                status = "✅ CORRECT"
            else:
                status = "❌ WRONG CLASS"

        print(f"{img_name:<15} | {expected_class:<10} | {pred_str:<25} | {status}")

        # Disegno per verifica
        for det in clean_detections:
            x, y, w, h = det['bbox']
            H, W, _ = img.shape
            x1, y1 = int((x-w/2)*W), int((y-h/2)*H)
            x2, y2 = int((x+w/2)*W), int((y+h/2)*H)
            color = (0, 255, 0) if det['class_id'] == class_id else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            lbl = f"{class_names[det['class_id']]} {det['confidence']:.2f}"
            cv2.putText(img, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        cv2.imwrite(f"result_platinum_{expected_class}.jpg", img)

    print("-" * 75)
    print("💾 Controlla le immagini 'result_platinum_*.jpg'")

if __name__ == "__main__":
    test_platinum_logic()