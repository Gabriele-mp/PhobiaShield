import cv2
import numpy as np
import time
import sys
import os
from typing import List, Dict, Optional

# --- FIX IMPORTAZIONI (Il trucco dell'Ingegnere) ---
# Aggiungiamo la root del progetto al Python Path per trovare i moduli ovunque
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/inference
project_root = os.path.dirname(os.path.dirname(current_dir)) # phobiashield root
if project_root not in sys.path:
    sys.path.append(project_root)

# Ora importiamo NMS con certezza assoluta
try:
    from src.inference.nms import nms
    print("✅ Modulo NMS caricato correttamente.")
except ImportError as e:
    nms = None
    print(f"❌ ERRORE CRITICO: NMS non trovato. Motivo: {e}")
    print("Il video mostrerà box sovrapposti (fase di debug).")

# ----------------------------------------------------

class PhobiaVideoProcessor:
    """
    Gestisce l'elaborazione video stocastica per il progetto PhobiaShield.
    
    Implementazione tecnica basata su:
    - SMDS_03: Discrete Convolution per il blurring (Gaussian Kernel)
    - SMDS_01: Set Theory per il filtraggio delle detection (NMS/IoU)
    - SMDS_03: Monte Carlo Simulation per il testing senza modello
    """
    
    def __init__(self, model=None, output_dir: str = "outputs/videos"):
        self.model = model
        self.output_dir = output_dir
        
        # Parametri del Kernel Gaussiano per la Convoluzione (Ref: SMDS_03)
        # Un sigma più alto = distribuzione più "piatta" = maggiore sfocatura
        self.blur_kernel_size = (51, 51) 
        self.blur_sigma = 30 
        
        # MAPPA UFFICIALE CLASSI (Per visualizzazione professionale)
        self.class_names = {
            0: "Clown",
            1: "Shark",
            2: "Spider", 
            3: "Blood",
            4: "Needles",                                  
        }

        print(f"VideoProcessor inizializzato. Kernel Convoluzione: {self.blur_kernel_size}")

    def apply_convolutional_blur(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Applica una convoluzione discreta (Gaussian Blur) sulla Region of Interest (ROI).
        Ref: SMDS_03_Simulations_MonteCarlo_Convolution_MGF.pdf
        """
        h, w, _ = frame.shape
        
        # bbox input: [center_x, center_y, width, height] (Formato YOLO normalizzato 0-1)
        cx, cy, bw, bh = bbox
        
        # Mapping nello spazio pixel (De-normalizzazione)
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        # Clipping per garantire che le coordinate siano dentro l'insieme immagine
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, h)

        # Estrazione della ROI (Sotto-matrice)
        roi = frame[y1:y2, x1:x2]
        
        # Gestione casi limite (ROI vuota)
        if roi.size == 0:
            return frame

        # APPLICAZIONE CONVOLUZIONE (Blur)
        # Matematicamente: (f * g)[n] = sum(f[m]g[n-m])
        blurred_roi = cv2.GaussianBlur(roi, self.blur_kernel_size, self.blur_sigma)
        
        # Reinserimento della matrice trasformata
        frame[y1:y2, x1:x2] = blurred_roi
        return frame

    def generate_monte_carlo_detections(self) -> List[Dict]:
        """
        SIMULAZIONE SCENARIO 'INVASIONE'.
        Genera molti oggetti (es. 10-15) sparsi ovunque per testare
        se il sistema li censura tutti senza crashare o rallentare.
        """
        detections = []
        # Generiamo tra 5 e 15 oggetti casuali ogni frame
        num_objects = np.random.randint(5, 15)
        
        for _ in range(num_objects):
            det = {
                "bbox": [
                    np.random.uniform(0.1, 0.9), # X sparso
                    np.random.uniform(0.1, 0.9), # Y sparso
                    np.random.uniform(0.05, 0.2), # Larghezza variabile
                    np.random.uniform(0.05, 0.2)  # Altezza variabile
                ],
                "confidence": np.random.uniform(0.4, 0.99),
                "class_id": np.random.randint(0, 5)
            }
            detections.append(det)
        return detections

    def process_video(self, input_path: str, output_name: str = "output.mp4", simulate: bool = False, debug: bool = True):
        """
        Processa il video. 
        Args:
            debug (bool): Se True, disegna un rettangolo rosso attorno al blur per verificare la precisione.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"ERRORE CRITICO: Impossibile aprire {input_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        save_path = f"{self.output_dir}/{output_name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        print(f"Elaborazione: {input_path} | Debug Mode: {debug}")
        
        # --- CONFIGURAZIONE NMS PER "FOLLA" ---
        # IoU più alto (0.45-0.5) evita di cancellare oggetti vicini ma distinti.
        # Confidenza bassa (0.25) per non perdere oggetti piccoli/difficili.
        nms_iou_thresh = 0.45
        nms_conf_thresh = 0.25

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            detections = []
            
            if simulate:
                # SIMULAZIONE CAOS: Generiamo tanti oggetti sparsi per testare la folla
                # (Sostituisci qui con la tua logica di simulazione preferita)
                detections = self.generate_monte_carlo_detections()
            elif self.model:
                # detections = self.model.predict(frame)
                pass

            # FASE 2: FILTRAGGIO (Usa Soft-NMS se disponibile per gestire sovrapposizioni)
            if nms and detections:
                try:
                    # Se nms.py ha soft_nms, usiamolo! È meglio per le folle.
                    from src.inference.nms import soft_nms
                    detections = soft_nms(detections, iou_threshold=nms_iou_thresh, conf_threshold=nms_conf_thresh, sigma=0.5)
                except ImportError:
                    # Fallback su NMS standard
                    detections = nms(detections, iou_threshold=nms_iou_thresh, conf_threshold=nms_conf_thresh)

            # FASE 3: TRASFORMAZIONE & DEBUG
            for det in detections:
                bbox = det['bbox']
                
                # 1. Applica Blur (La censura)
                frame = self.apply_convolutional_blur(frame, bbox)
                
                # 2. DISEGNA BORDO E ETICHETTE (Debug Mode)
                if debug:
                    h, w, _ = frame.shape
                    cx, cy, bw, bh = bbox
                    
                    # De-normalizzazione coordinate (da 0-1 a Pixel)
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    # Clipping di sicurezza
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    # Recupera il nome leggibile dal dizionario
                    class_name = self.class_names.get(det['class_id'], f"Class {det['class_id']}")
                    
                    # Crea l'etichetta
                    label = f"{class_name} {det['confidence']:.0%}"
                    
                    # --- COLOR CODING STRATEGICO (BGR Format) ---
                    cid = det['class_id']
                    
                    if cid == 2:
                        # Classe 2 -> ROSSO (0, 0, 255)
                        color = (0, 0, 255)
                        thickness = 2
                        
                    elif cid in [0, 1]:
                        # Classe 0, 1
                        color = (74, 223, 45)
                        thickness = 2
                        
                    elif cid in [3, 4]:
                        # Classe 3, 4
                        color = (182, 89, 170) 
                        thickness = 2
                        
                    else:
                        # Default per classe 5 o errori -> Grigio
                        color = (128, 128, 128)
                        thickness = 2
                    
                    # A. Disegna il rettangolo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # B. Disegna lo sfondo per il testo
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), color, -1)
                    
                    # C. Scrivi il testo in BIANCO
                    cv2.putText(frame, label, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out.write(frame)

        cap.release()
        out.release()
        print(f"✅ Finito. Controlla {save_path}")

# --- BLOCCO DI TEST (Unit Test) ---
if __name__ == "__main__":
    import os
    
    # 1. Verifica ambiente
    if not os.path.exists("outputs/videos"):
        os.makedirs("outputs/videos")
        
    # 2. Istanzia il processore
    processor = PhobiaVideoProcessor()
    
    # 3. Definisci il video di test (Devi averlo scaricato!)
    # Mettilo nella cartella 'assets' per non sporcare Git
    video_test = "data_workspace/assets/test_video.mp4" 
    
    if os.path.exists(video_test):
        # 4. Avvia in modalità SIMULAZIONE (Monte Carlo)
        print("Avvio test Monte Carlo...")
        processor.process_video(video_test, "test_result_montecarlo.mp4", simulate=True)
    else:
        print(f"⚠️ ATTENZIONE: File {video_test} non trovato. Scarica un trailer e rinominalo!")
        print("Suggerimento: Scarica il trailer di IT o SAW e mettilo in una cartella 'assets'.")