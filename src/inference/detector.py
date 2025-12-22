import torch
import cv2
import numpy as np
import pathlib
from src.models.phobia_net_fpn import PhobiaNetFPN 

# Try to import YOLO library (Ultralytics)
try:
    from ultralytics import YOLO
except ImportError:
    pass

class PhobiaDetector:
    def __init__(self, model_path, model_type='custom', device=None):
        """
        Initializes the detector.
        Args:
            model_path (str): Path to the .pt or .pth weight file.
            model_type (str): 'yolo' for YOLOv8, 'custom' for PhobiaNetFPN.
            device (str): 'cuda' or 'cpu'.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type.lower()
        
        # Standard Anchors (Used only for Custom FPN)
        self.anchors = {
            8:  [[10, 13], [16, 30]],   
            16: [[30, 61], [62, 45]],   
            32: [[116, 90], [156, 198]] 
        }
        self.strides = [8, 16, 32]
        self.num_classes = 5 
        
        if self.model_type == 'yolo':
            self._load_yolov8(model_path)
        else:
            self._load_custom(model_path)

    def _load_yolov8(self, model_path):
        print(f"[INFO] Loading YOLOv8 from: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("[SUCCESS] YOLOv8 Loaded.")
        except Exception as e:
            print(f"[FATAL ERROR] YOLOv8 Crash: {e}")
            raise e

    def _load_custom(self, model_path):
        print(f"[INFO] Loading Custom FPN from: {model_path}")
        self.model = PhobiaNetFPN(num_classes=self.num_classes)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Clean state_dict keys (remove _orig_mod prefix if present from compilation)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("_orig_mod.", "")
            new_state_dict[name] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        
        self.model.float() 
        for param in self.model.parameters():
            param.data = param.data.float()
            
        self.model.eval()
        self.img_size = 224
        print("[SUCCESS] Custom FPN Loaded.")

    def predict(self, frame):
        if self.model_type == 'yolo':
            return self._predict_yolov8(frame)
        else:
            return self._predict_custom(frame)

    def _predict_yolov8(self, frame):
        # YOLO CONFIGURATION: THRESHOLD 0.35
        # Using 0.35 as agreed for optimal detection/noise balance
        results = self.model(frame, verbose=False, conf=0.35, iou=0.45)
        
        final = []
        for r in results:
            boxes = r.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                final.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
        
        return np.array(final) if len(final) > 0 else np.empty((0, 6))

    def _predict_custom(self, frame):
        h_orig, w_orig = frame.shape[:2]
        
        # Preprocessing
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        dtype_model = next(self.model.parameters()).dtype
        t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device).to(dtype=dtype_model)

        with torch.no_grad():
            features = self.model(t)

        all_boxes = []
        
        # Decode FPN outputs
        for i, feature_map in enumerate(features):
            stride = self.strides[i]
            anchors_level = self.anchors[stride]
            bs, h, w, channels = feature_map.shape
            feat = feature_map.reshape(bs, h, w, 2, 5 + self.num_classes).cpu().numpy()[0]
            
            for grid_y in range(h):
                for grid_x in range(w):
                    for anchor_idx in range(2): 
                        data = feat[grid_y, grid_x, anchor_idx]
                        tx, ty, tw, th, obj_raw = data[:5]
                        class_scores = data[5:]
                        
                        # CUSTOM CONFIGURATION: THRESHOLD 0.55
                        # Higher threshold to reduce False Positives
                        obj_score = 1 / (1 + np.exp(-np.clip(obj_raw, -100, 100)))
                        if obj_score < 0.55: continue 
                        
                        shift_scores = class_scores - np.max(class_scores)
                        exp_scores = np.exp(shift_scores)
                        class_probs = exp_scores / np.sum(exp_scores)
                        class_id = np.argmax(class_probs)
                        final_score = obj_score * class_probs[class_id]
                        
                        if final_score < 0.55: continue

                        # Coordinate decoding
                        sig_tx = 1 / (1 + np.exp(-np.clip(tx, -100, 100)))
                        sig_ty = 1 / (1 + np.exp(-np.clip(ty, -100, 100)))
                        cx = (sig_tx + grid_x) * stride
                        cy = (sig_ty + grid_y) * stride
                        bw = np.exp(np.clip(tw, -20, 20)) * anchors_level[anchor_idx][0]
                        bh = np.exp(np.clip(th, -20, 20)) * anchors_level[anchor_idx][1]
                        
                        x1 = (cx - bw/2) * (w_orig / self.img_size)
                        y1 = (cy - bh/2) * (h_orig / self.img_size)
                        x2 = (cx + bw/2) * (w_orig / self.img_size)
                        y2 = (cy + bh/2) * (h_orig / self.img_size)
                        
                        all_boxes.append([x1, y1, x2, y2, final_score, class_id])

        if len(all_boxes) == 0:
            return np.empty((0, 6))
            
        all_boxes = np.array(all_boxes)
        
        if len(all_boxes) > 500:
            indices_sort = np.argsort(all_boxes[:, 4])[::-1]
            all_boxes = all_boxes[indices_sort[:500]]

        boxes = all_boxes[:, :4]
        scores = all_boxes[:, 4]
        
        # NMS (Non-Maximum Suppression) with threshold 0.55
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.55, nms_threshold=0.05)
        
        if len(indices) > 0:
            return all_boxes[indices.flatten()]
        else:
            return np.empty((0, 6))