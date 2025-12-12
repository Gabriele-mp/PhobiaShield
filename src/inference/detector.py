import torch
import cv2
import numpy as np
import sys
import os
from typing import List, Dict

# Ensure Python can find source modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# --- MODIFICA CRITICA: Importiamo la nuova classe FPN ---
from src.models.phobia_net_fpn import PhobiaNetFPN 
from omegaconf import OmegaConf

class PhobiaDetector:
    def __init__(self, model_path=None, config_path="cfg/model/tiny_yolo.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Detector initializing on {self.device}...")

        # 1. Load Configuration
        if os.path.exists(config_path):
            conf = OmegaConf.load(config_path)
            self.config = OmegaConf.to_container(conf, resolve=True)
        else:
            # Fallback safe config
            self.config = {
                "output": {"num_classes": 5},
                "architecture": {
                    "grid_size": 13, 
                    "num_boxes_per_cell": 2, 
                    "in_channels": 3,
                    "leaky_relu_slope": 0.1 # FPN needs this
                },
                "init": {"type": "kaiming"}
            }

        # 2. Initialize NEW FPN Model
        # Usiamo PhobiaNetFPN invece di PhobiaNet
        try:
            self.model = PhobiaNetFPN(self.config).to(self.device)
        except Exception as e:
            print(f"❌ Error initializing PhobiaNetFPN: {e}")
            print("   Ensure src/models/phobia_net.py was also updated via git pull.")
            raise e
            
        self.model.eval() 

        # 3. Load Weights
        if model_path and os.path.exists(model_path):
            print(f"📂 Loading FPN weights from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Logic to handle 'model_state_dict', 'state_dict' or direct dict
                if isinstance(checkpoint, dict):
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # FIX: Remove 'module.' prefix if present (DataParallel issue)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                
                self.model.load_state_dict(new_state_dict)
                print("✅ Weights loaded successfully!")
                
            except Exception as e:
                print(f"❌ Error loading weights: {e}")
                print("   Mismatch is likely due to FPN architecture differences.")
        else:
            print("⚠️  WARNING: No weights loaded! Using Random Weights.")

    def preprocess(self, img):
        """Prepare image: Resize -> Normalize -> Tensor"""
        # Dynamic size check
        target_size = self.config.get("architecture", {}).get("img_size", 416)
        
        img_resized = cv2.resize(img, (target_size, target_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb / 255.0
        img_tensor = torch.from_numpy(img_norm).float().permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)

    def detect(self, frame, conf_threshold=0.3):
        """Full pipeline"""
        tensor = self.preprocess(frame)
        
        with torch.no_grad():
            # PhobiaNetFPN.predict gestisce già l'NMS internamente, 
            # ma restituisce lo stesso formato List[List[Dict]].
            # Passiamo nms_threshold default 0.4
            predictions = self.model.predict(
                tensor, 
                conf_threshold=conf_threshold, 
                nms_threshold=0.4
            )
        
        return predictions[0]