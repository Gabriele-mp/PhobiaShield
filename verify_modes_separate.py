import cv2
import os
import sys
from src.inference.detector import PhobiaDetector
from src.inference.nms import nms

def run_test_for_mode(mode_name):
    weights_path = "outputs/checkpoints/FPN_epoch18_loss6.3037.pth"
    
    print(f"\n💎 STARTING VALIDATION FOR MODE: {mode_name}")
    print("-" * 75)
    
    # Inizializza Detector con la modalità specifica
    try:
        detector = PhobiaDetector(model_path=weights_path, mode=mode_name)
    except Exception as e:
        print(f"❌ Errore init: {e}")
        return

    class_names = {0: "Clown", 1: "Shark", 2: "Spider", 3: "Blood", 4: "Needle"}
    test_images = {
        0: "clown_test.jpg",
        1: "shark_test.jpg",
        2: "spider_test.jpg",
        3: "blood_test.jpg",
        4: "needle_test.jpg"
    }

    print(f"{'IMAGE':<15} | {'EXPECTED':<10} | {'PREDICTION':<25} | {'STATUS'}")
    print("-" * 75)

    for class_id, img_name in test_images.items():
        if not os.path.exists(img_name): continue
        expected_class = class_names[class_id]
        
        img = cv2.imread(img_name)
        
        # 1. DETECT (Usa soglie interne V25 o V20)
        raw_detections = detector.detect(img, conf_threshold=None)
        
        # 2. NMS (Usa logica Cross-Class)
        clean_detections = nms(raw_detections, iou_threshold=0.10)
        
        # 3. ANALISI
        if len(clean_detections) == 0:
            pred_str = "No detections"
            status = "⚠️ BLIND"
        else:
            # Prendi la migliore
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
        
        # (Opzionale) Salva immagine se vuoi vederla
        # for det in clean_detections: ... (disegno) ...
        # cv2.imwrite(f"result_{mode_name}_{expected_class}.jpg", img)

    print("-" * 75)
    print(f"✅ {mode_name} TEST COMPLETE.\n")

if __name__ == "__main__":
    # Esegue prima V25
    run_test_for_mode("V25")
    
    # Esegue dopo V20
    run_test_for_mode("V20")