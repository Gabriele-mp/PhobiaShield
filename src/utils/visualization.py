import cv2
import numpy as np
from typing import List, Dict

class Visualizer:
    def __init__(self):
        # MAPPING UFFICIALE DA DATASET_FINAL_README.md
        # 0: Clown, 1: Shark, 2: Spider, 3: Blood, 4: Needle
        
        self.class_names = {
            0: "Clown",   # 🤡
            1: "Shark",   # 🦈
            2: "Spider",  # 🕷️
            3: "Blood",   # 🩸
            4: "Needle"   # 💉
        }

        # Colori BGR (Blue, Green, Red) per OpenCV
        self.colors = {
            0: (0, 165, 255),   # Clown -> Arancione
            1: (192, 192, 192), # Shark -> Grigio Argento
            2: (0, 0, 0),       # Spider -> Nero (o molto scuro)
            3: (0, 0, 255),     # Blood -> Rosso Puro
            4: (255, 255, 0),   # Needle -> Ciano/Azzurro
            5: (255, 0, 255)    # Fallback -> Magenta
        }

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        h, w, _ = frame.shape
        
        for det in detections:
            bbox = det['bbox']
            conf = det.get('confidence', 0.0)
            class_id = int(det['class_id'])
            
            # Conversione coordinate
            cx, cy, bw, bh = bbox
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            # Clipping
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x1 >= x2 or y1 >= y2: continue

            # Colore specifico o default
            color = self.colors.get(class_id, self.colors[5])
            
            # Disegna Box
            # Se è nero (Spider), aggiungi un bordo bianco sottile per visibilità
            if class_id == 2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Recupera nome classe corretto
            label_text = self.class_names.get(class_id, f"ID {class_id}")
            label = f"{label_text} {conf:.0%}"
            
            # Etichetta
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            
            # Testo (Bianco o Nero a seconda del contrasto)
            text_color = (255, 255, 255)
            if class_id in [1, 4]: # Su sfondo chiaro (grigio/ciano), testo nero
                text_color = (0, 0, 0)
                
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            
        return frame