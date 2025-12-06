import cv2
import numpy as np
import os
from src.utils.visualization import Visualizer
from src.inference.nms import nms
from src.inference.blur import apply_blur

def test_precision():
    print("🔬 AVVIO PROTOCOLLO DI CALIBRAZIONE...")
    
    # 1. Creiamo un frame nero 1000x1000 (facile per i calcoli mentali)
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    
    # 2. INIEZIONE DATI NOTI
    # Simuliamo cosa uscirebbe dal detector se fosse perfetto.
    # Formato atteso: [cx, cy, w, h] normalizzati (0.0 - 1.0)
    
    fake_detections = [
        # CASO A: Box perfetta al centro (Ragno)
        # Centro (0.5, 0.5), Larghezza 0.2 (200px), Altezza 0.2 (200px)
        # Ci aspettiamo un box da x=400 a x=600, y=400 a y=600
        {'class_id': 2, 'confidence': 0.95, 'bbox': [0.5, 0.5, 0.2, 0.2]},
        
        # CASO B: Test NMS (Due box quasi identiche sovrapposte)
        # Se l'NMS funziona, ne deve rimanere SOLO UNA.
        {'class_id': 3, 'confidence': 0.90, 'bbox': [0.2, 0.2, 0.1, 0.1]}, # Sangue forte
        {'class_id': 3, 'confidence': 0.85, 'bbox': [0.21, 0.21, 0.1, 0.1]}, # Sangue debole (da eliminare)
        
        # CASO C: Test Bordi (Box parzialmente fuori)
        # Centro nell'angolo in basso a destra (1.0, 1.0)
        {'class_id': 0, 'confidence': 0.88, 'bbox': [1.0, 1.0, 0.2, 0.2]},
    ]
    
    print(f"📥 Input: {len(fake_detections)} predizioni grezze inserite.")
    
    # 3. TEST NMS
    # Se il tuo nms.py funziona, 'clean_detections' deve avere lunghezza 3 (non 4).
    clean_detections = nms(fake_detections, iou_threshold=0.5, conf_threshold=0.5)
    
    print(f"📤 Output NMS: {len(clean_detections)} predizioni rimaste.")
    
    if len(clean_detections) == 3:
        print("   ✅ NMS TEST: PASSATO (Il doppione è stato rimosso).")
    else:
        print(f"   ❌ NMS TEST: FALLITO. Attesi 3 box, trovati {len(clean_detections)}.")
        
    # 4. TEST BLURRING E VISUALIZATION
    vis = Visualizer()
    
    for det in clean_detections:
        # Applica Blur
        frame = apply_blur(frame, det['bbox'], intensity=15)
        
    # Disegna
    frame = vis.draw_detections(frame, clean_detections)
    
    # 5. SALVATAGGIO PER ISPEZIONE VISIVA
    cv2.imwrite("test_calibration_result.jpg", frame)
    print("💾 Immagine salvata: 'test_calibration_result.jpg'")
    
    # 6. VERIFICA MATEMATICA COORDINATE (Pixel check)
    # Controlliamo se il pixel centrale (500,500) è dentro un box nero (Ragno ID 2)
    # Nota: Visualizer disegna bordi colorati, ma controlliamo se c'è attività
    center_pixel = frame[500, 500]
    print(f"🔍 Pixel centrale (BGR): {center_pixel}")
    
    if np.any(center_pixel > 0): # Se non è nero puro, abbiamo disegnato qualcosa
        print("   ✅ COORDINATE TEST: PASSATO (Disegno rilevato al centro).")
    else:
        print("   ⚠️ COORDINATE TEST: DUBBIO (Il centro è vuoto, controlla l'immagine).")

if __name__ == "__main__":
    test_precision()