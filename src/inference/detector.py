import torch
import cv2
import numpy as np
import sys
import os
from typing import List, Dict

# Ensure Python can find source modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.models.phobia_net import PhobiaNet
from omegaconf import OmegaConf

class PhobiaDetector:
    def __init__(self, model_path=None, config_path="cfg/model/tiny_yolo.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Detector initializing on {self.device}...")

        # Load real configuration
        if os.path.exists(config_path):
            conf = OmegaConf.load(config_path)
            self.config = OmegaConf.to_container(conf, resolve=True)
        else:
            # Fallback if yaml is missing: hardcoded safety config
            self.config = {
                "output": {"num_classes": 5},
                "architecture": {
                    "grid_size": 13, 
                    "num_boxes_per_cell": 2, 
                    "in_channels": 3,
                    "layers": [{"filters": 512}]
                },
                "init": {"type": "kaiming"}
            }

        # Initialize model (Empty skeleton)
        self.model = PhobiaNet(self.config).to(self.device)
        self.model.eval() # Eval mode (disable dropout, etc.)

        # Load weights (If available!!!!...otherwise use random weights for testing)
        if model_path and os.path.exists(model_path):
            print(f"Loading weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            # Robust loading: handle both full checkpoint state and model state dict
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("WARNING: No weights loaded!!!! Using Random Weights (Test Mode)")

    def preprocess(self, img):
        """Prepare image for the network: Resize -> Normalize -> Tensor"""
        target_size = self.config.get("architecture", {}).get("img_size", 416)
        
        img_resized = cv2.resize(img, (target_size, target_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb / 255.0
        img_tensor = torch.from_numpy(img_norm).float().permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)

    def detect(self, frame, conf_threshold=0.5):
        """Full pipeline: Image -> Predictions"""
        tensor = self.preprocess(frame)
        
        with torch.no_grad():
            # Call the predict() method implemented in phobia_net.py
            predictions = self.model.predict(tensor, conf_threshold=conf_threshold)
        
        # predict returns a list of lists (batch), we take the first element
        return predictions[0]