import torch
import cv2
import numpy as np
import sys
import os
from typing import List, Dict

# Assicuriamo che Python trovi i moduli
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.models.phobia_net import PhobiaNet
from omegaconf import OmegaConf

class PhobiaDetector:
    def __init__(self, model_path=None, config_path="cfg/model/tiny_yolo.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Detector initializing on {self.device}...")

        # 1. Carica la Configurazione Reale (la stessa usata per il training)
        if os.path.exists(config_path):
            conf = OmegaConf.load(config_path)
            self.config = OmegaConf.to_container(conf, resolve=True)
        else:
            # Fallback se non trovi il file yaml, configurazione hardcoded di sicurezza
            self.config = {
                "output": {"num_classes": 5},
                "architecture": {
                    "grid_size": 13, 
                    "num_boxes_per_cell": 2, 
                    "in_channels": 3,
                    "layers": [{"filters": 512}] # Dummy per init
                },
                "init": {"type": "kaiming"}
            }

        # 2. Inizializza il Modello (Scheletro Vuoto)
        self.model = PhobiaNet(self.config).to(self.device)
        self.model.eval() # Modalità valutazione (spegne dropout, ecc)

        # 3. Carica i Pesi (Se esistono, altrimenti usa pesi random per test)
        if model_path and os.path.exists(model_path):
            print(f"📂 Loading weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            # Gestione robusta: a volte i checkpoint salvano l'intero stato, a volte solo i pesi
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("⚠️  WARNING: No weights loaded! Using Random Weights (Test Mode)")

    def preprocess(self, img):
        """Prepara l'immagine per la rete: Resize -> Normalize -> Tensor"""
        img_resized = cv2.resize(img, (416, 416))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb / 255.0
        # Da (H, W, C) a (C, H, W) e aggiungi dimensione batch
        img_tensor = torch.from_numpy(img_norm).float().permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)

    def detect(self, frame, conf_threshold=0.5):
        """Pipeline completa: Immagine -> Predizioni"""
        tensor = self.preprocess(frame)
        
        with torch.no_grad():
            # Qui chiami il metodo predict() che il Membro 1 ha già scritto in phobia_net.py
            # Questo è il punto chiave dell'integrazione!
            predictions = self.model.predict(tensor, conf_threshold=conf_threshold)
        
        # predict restituisce una lista di liste (batch), noi prendiamo il primo elemento
        return predictions[0]