import cv2
import os
from src.inference.detector import PhobiaDetector
from src.inference.nms import nms

def run_comparison():
    weights = "outputs/checkpoints/FPN_epoch18_loss6.3037.pth" 
    
    # Classi e Immagini
    test_data = {
        0: ("Clown", "clown_test.jpg"),
        1: ("Shark", "shark_test.jpg"),
        2: ("Spider", "spider_test.jpg"),
        3: ("Blood", "blood_test.jpg"),
        4: ("Needle", "needle_test.jpg")
    }
    
    # Inizializza i due detector
    print("🔧 Initializing V25 (Big & Brave)...")
    det_v25 = PhobiaDetector(weights, mode="V25")
    
    print("🔧 Initializing V20 (Hybrid)...")
    det_v20 = PhobiaDetector(weights, mode="V20")
    
    print("\n" + "="*80)
    print(f"{'IMAGE':<15} | {'V25 PREDICTION':<30} | {'V20 PREDICTION':<30}")
    print("="*80)
    
    for cid, (cname, img_path) in test_data.items():
        if not os.path.exists(img_path): continue
        
        img = cv2.imread(img_path)
        
        # --- TEST V25 ---
        raw_v25 = det_v25.detect(img, conf_threshold=None) # Usa soglie interne
        nms_v25 = nms(raw_v25, iou_threshold=0.10)
        
        res_v25 = "No Detection"
        if nms_v25:
            best = max(nms_v25, key=lambda x: x['confidence'])
            # Mappa ID a Nome
            name = ["Clown", "Shark", "Spider", "Blood", "Needle"][best['class_id']]
            res_v25 = f"{name} ({best['confidence']:.1%})"
            
        # --- TEST V20 ---
        raw_v20 = det_v20.detect(img, conf_threshold=None)
        nms_v20 = nms(raw_v20, iou_threshold=0.10)
        
        res_v20 = "No Detection"
        if nms_v20:
            best = max(nms_v20, key=lambda x: x['confidence'])
            name = ["Clown", "Shark", "Spider", "Blood", "Needle"][best['class_id']]
            res_v20 = f"{name} ({best['confidence']:.1%})"
            
        print(f"{cname:<15} | {res_v25:<30} | {res_v20:<30}")
        
    print("="*80)

if __name__ == "__main__":
    run_comparison()