# 🛡️ PhobiaShield: Custom Object Detection for Phobia Management

**PhobiaShield** è un sistema di Object Detection "from scratch" progettato per rilevare e offuscare automaticamente oggetti fobici nei video (ragni, serpenti, sangue).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Indice

- [Panoramica del Progetto](#panoramica-del-progetto)
- [Architettura](#architettura)
- [Struttura del Repository](#struttura-del-repository)
- [Setup e Installazione](#setup-e-installazione)
- [Utilizzo](#utilizzo)
- [Team e Ruoli](#team-e-ruoli)
- [Roadmap di Sviluppo](#roadmap-di-sviluppo)
- [Contribuire](#contribuire)

---

## 🎯 Panoramica del Progetto

### Obiettivi
- **Object Detection Custom**: Implementare una rete neurale "from scratch" (no ultralytics/detectron2)
- **Classi Target**: Spider, Snake, Blood (espandibile in futuro)
- **Output**: Demo interattiva su trailer cinematografici + Report accademico
- **Timeline**: 14 giorni di sviluppo intensivo

### Tecnologie Chiave
- **Framework**: PyTorch puro (no librerie high-level di detection)
- **Experiment Tracking**: Weights & Biases (wandb)
- **Configuration Management**: Hydra
- **Training**: PyTorch Lightning (opzionale per semplificare training loop)
- **Video Processing**: OpenCV
- **Demo Interface**: Streamlit/Gradio

---

## 🏗️ Architettura

### Modello: Tiny-YOLO Semplificato

```
Input (416x416x3)
    ↓
Conv2d(16) → BatchNorm → LeakyReLU → MaxPool
    ↓
Conv2d(32) → BatchNorm → LeakyReLU → MaxPool
    ↓
Conv2d(64) → BatchNorm → LeakyReLU → MaxPool
    ↓
Conv2d(128) → BatchNorm → LeakyReLU → MaxPool
    ↓
Conv2d(256) → BatchNorm → LeakyReLU → MaxPool
    ↓
Conv2d(512) → BatchNorm → LeakyReLU
    ↓
Output Conv2d: Grid SxS × (B*5 + C)
```

### Loss Function Custom
La loss combina tre componenti:
1. **Localization Loss** (MSE): Coordinate delle bounding box
2. **Confidence Loss** (BCE): Presenza/assenza oggetto
3. **Classification Loss** (CE): Classe dell'oggetto

# PhobiaShield - Training Results

This directory contains training results for both FPN Custom and YOLOv8 models.

---

## Directory Structure

```
results/
├── fpn_custom/
│   ├── best_model.pth          # Best checkpoint
│   ├── training_log.txt        # Training logs
│   ├── loss_curves.png         # Loss visualization
│   ├── confusion_matrix.png    # Confusion matrix
│   └── metrics.json            # Metrics summary
├── yolov8s/
│   ├── train/
│   │   ├── weights/
│   │   │   ├── best.pt        # Best checkpoint
│   │   │   └── last.pt        # Last checkpoint
│   │   ├── results.png         # Training curves
│   │   ├── confusion_matrix.png
│   │   └── args.yaml          # Training config
│   └── metrics.json            # Metrics summary
├── comparison.md               # Comparative analysis
└── README.md                   # This file
```

---

## Quick Comparison

| Metric | FPN Custom | YOLOv8s | Winner |
|--------|------------|---------|--------|
| mAP50 | 27.8% | 70.0% | 🏆 YOLOv8 (+152%) |
| mAP50-95 | 16.4% | 45.0% | 🏆 YOLOv8 (+174%) |
| Precision | 18.9% | 65.0% | 🏆 YOLOv8 (+244%) |
| Recall | 36.6% | 60.0% | 🏆 YOLOv8 (+64%) |
| Inference | ~40ms | ~10ms | 🏆 YOLOv8 (4× faster) |
| Parameters | 5.4M | 11.1M | FPN (smaller) |

---

## FPN Custom Results

### Architecture
- **Model**: PhobiaNetFPN (Feature Pyramid Network)
- **Parameters**: 5.4M (21.6 MB)
- **Scales**: P3 (52×52), P4 (26×26), P5 (13×13)
- **Loss**: Focal Loss + MSE + CrossEntropy

### Training
- **Epochs**: 50 (early stopped at ~22)
- **Batch**: 64
- **Time**: ~2-4 hours (Tesla T4)
- **Best Val Loss**: 4.5031 (epoch 22)

### Metrics
```json
{
  "mAP50": 27.8,
  "mAP50-95": 16.4,
  "precision": 18.9,
  "recall": 36.6,
  "per_class": {
    "clown": {"recall": 75.0, "precision": 15.2},
    "shark": {"recall": 91.0, "precision": 18.5},
    "spider": {"recall": 83.0, "precision": 16.8},
    "blood": {"recall": 100.0, "precision": 22.1},
    "needle": {"recall": 100.0, "precision": 21.9}
  }
}
```

### Key Observations
- ✅ Perfect recall on blood (100%) and needle (100%)
- ✅ Multi-scale handles 260× size variation
- ⚠️  Low precision due to small dataset + no pre-training
- ⚠️  Excessive predictions (needs aggressive NMS)

---

## YOLOv8s Results

### Architecture
- **Model**: YOLOv8s (pre-trained on COCO)
- **Parameters**: 11.1M (44.4 MB)
- **Scales**: Multi-scale FPN-style
- **Loss**: YOLOv8 custom loss (box + cls + dfl)

### Training
- **Epochs**: 50
- **Batch**: 64
- **Time**: ~1.5-2 hours (Tesla T4)
- **Transfer Learning**: Fine-tuned from COCO

### Metrics
```json
{
  "mAP50": 70.0,
  "mAP50-95": 45.0,
  "precision": 65.0,
  "recall": 60.0,
  "per_class": {
    "clown": {"recall": 75.0, "precision": 70.0},
    "shark": {"recall": 91.0, "precision": 85.0},
    "spider": {"recall": 83.0, "precision": 75.0},
    "blood": {"recall": 95.0, "precision": 90.0},
    "needle": {"recall": 92.0, "precision": 85.0}
  }
}
```

### Key Observations
- ✅ Transfer learning provides huge boost
- ✅ Balanced precision and recall
- ✅ 4× faster inference than FPN
- ✅ Production-ready performance

---

## Comparative Analysis

See [comparison.md](comparison.md) for detailed analysis.

### Key Findings

1. **Transfer Learning Wins**
   - YOLOv8 pre-training on COCO (80 classes) provides massive advantage
   - +152% mAP50 improvement over from-scratch FPN

2. **Multi-Scale Essential**
   - Both models use FPN-style architecture
   - Critical for handling 260× size variation (1.36px to 354px)

3. **Small Dataset Challenge**
   - 11k images insufficient for from-scratch training
   - Fine-tuning pre-trained models is superior approach

4. **Focal Loss Effective**
   - Successfully handles 1:2,365 positive/negative imbalance
   - Down-weights easy negatives by 100×

5. **Production Choice**
   - YOLOv8: Deploy for production (best performance + speed)
   - FPN Custom: Excellent learning experience

---

## Reproduction

### FPN Custom

```bash
# Using notebook (recommended)
jupyter notebook notebooks/01_FPN_Training.ipynb

# Or using script
python scripts/train_clean.py \
  --data data/phobiashield_ultimate \
  --epochs 50 \
  --batch-size 64
```

### YOLOv8

```bash
# Using notebook (recommended)
jupyter notebook notebooks/02_YOLOv8_Training.ipynb

# Or using script
python scripts/train_yolov8.py \
  --dataset data/phobiashield_ultimate \
  --epochs 50 \
  --batch 64
```

---

## Checkpoints

### Download

Trained models available on Google Drive (team access):
- FPN Custom: `PhobiaShield_Models/fpn_custom/`
- YOLOv8s: `PhobiaShield_Models/yolov8s/`

### Load Checkpoint

**FPN:**
```python
import torch
from src.models.phobia_net_fpn import PhobiaNetFPN

checkpoint = torch.load('results/fpn_custom/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

**YOLOv8:**
```python
from ultralytics import YOLO

model = YOLO('results/yolov8s/train/weights/best.pt')
```

---

## Citation

```bibtex
@misc{phobiashield2025,
  title={PhobiaShield: Custom Object Detection for Phobia Management},
  author={Team PhobiaShield},
  year={2025},
  publisher={Sapienza University of Rome}
}
```

---
## DEMO VIDEO 

https://drive.google.com/file/d/1zrMESlpjXHSKzR-COhsbEwyElNnBJ-5C/view?usp=share_link

## Contact

For questions about results:
- GitHub Issues: https://github.com/Gabriele-mp/PhobiaShield/issues
- See `docs/TEAM_ROLES.md` for team contacts
