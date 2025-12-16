import cv2
import torch
import os
import sys
import numpy as np

# Setup path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.inference.detector import PhobiaDetector
# --- AGGIUNTA FONDAMENTALE: Importiamo il filtro ---
from src.inference.nms import nms

def test_all_classes_clean():
    print("🔬 STARTING CLEAN VALIDATION (WITH NMS)...")
    
    class_names = {0: "Clown", 1: "Shark", 2: "Spider", 3: "Blood", 4: "Needle"}
    
    test_images = {
        0: "clown_test.jpg",
        1: "shark_test.jpg",
        2: "spider_test.jpg",
        3: "blood_test.jpg",
        4: "needle_test.jpg"
    }
    
    weights_path = "outputs/checkpoints/FPN_epoch18_loss6.3037.pth"
    
    print(f"🧠 Caricamento Modello...")
    detector = PhobiaDetector(model_path=weights_path)

    print("-" * 70)
    print(f"{'IMAGE':<15} | {'EXPECTED':<10} | {'BEST PREDICTION':<25} | {'STATUS'}")
    print("-" * 70)

    for class_id, img_name in test_images.items():
        expected_class = class_names[class_id]
        
        if not os.path.exists(img_name):
            continue
            
        img = cv2.imread(img_name)
        
        # 1. INFERENZA (Raw)
        # Teniamo la soglia bassa per vedere cosa pensa la rete
        raw_detections = detector.detect(img, conf_threshold=0.2)
        
        # 2. PULIZIA (NMS) - IL PASSAGGIO MAGICO
        # Questo rimuove i 50 rettangoli sovrapposti e ne lascia solo uno per oggetto
        clean_detections = nms(raw_detections, iou_threshold=0.3, conf_threshold=0.2)
        
        # Analisi
        if len(clean_detections) == 0:
            pred_str = "No detections"
            status = "⚠️ BLIND"
        else:
            # Prendiamo quella con confidenza più alta
            best_det = max(clean_detections, key=lambda x: x['confidence'])
            pred_id = int(best_det['class_id'])
            pred_name = class_names.get(pred_id, "Unknown")
            conf = best_det['confidence']
            
            pred_str = f"{pred_name} ({conf:.0%})"
            
            if pred_id == class_id:
                status = "✅ CORRECT"
            else:
                status = "❌ WRONG CLASS"
        
        print(f"{img_name:<15} | {expected_class:<10} | {pred_str:<25} | {status}")
        
        # 3. DISEGNO (Solo i box puliti)
        for det in clean_detections:
            bbox = det['bbox']
            p_id = int(det['class_id'])
            p_name = class_names.get(p_id, "?")
            p_conf = det['confidence']
            
            h, w, _ = img.shape
            cx, cy, bw, bh = bbox
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            # Verde se la classe è giusta, Rosso se è sbagliata
            color = (0, 255, 0) if p_id == class_id else (0, 0, 255)
            
            # Box spessore 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Etichetta con sfondo per leggerla meglio
            label = f"{p_name} {p_conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1-20), (x1+tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
        cv2.imwrite(f"result_{expected_class}_clean.jpg", img)

    print("-" * 70)
    print("💾 Guarda le immagini 'result_*_clean.jpg'. Ora dovrebbero essere leggibili.")

if __name__ == "__main__":
    test_all_classes_clean()