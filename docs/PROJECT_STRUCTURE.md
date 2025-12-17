# PhobiaShield - Project Structure

Complete directory structure and file descriptions for the PhobiaShield repository.

---

## Overview

```
PhobiaShield/
├── cfg/                    # Configuration files (Hydra)
├── src/                    # Source code (models, data, utils)
├── scripts/                # Training and utility scripts
├── notebooks/              # Jupyter notebooks for training
├── docs/                   # Documentation
├── presentation/           # Final presentation materials
├── results/                # Training results and analysis
├── data/                   # Dataset (gitignored)
├── outputs/                # Model checkpoints (gitignored)
└── README.md               # Main documentation
```

---

## Detailed Structure

### 📂 Root Directory

```
PhobiaShield/
├── README.md              # Project overview and quickstart
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── LICENSE               # Project license
└── setup.py              # Package installation script (optional)
```

---

### 📂 `cfg/` - Configuration Files

Hydra-based configuration for model training.

```
cfg/
├── config.yaml           # Main config (imports model/data/training)
├── model/
│   ├── tiny_yolo_5class.yaml      # Model architecture config
│   └── phobia_net_fpn.yaml        # FPN config
├── data/
│   └── phobiashield.yaml          # Dataset paths and classes
└── training/
    ├── default.yaml               # Default training params
    └── fast_test.yaml             # Quick test config
```

**Usage:**
```python
from omegaconf import OmegaConf
config = OmegaConf.load('cfg/model/tiny_yolo_5class.yaml')
```

**Example config:**
```yaml
# cfg/model/tiny_yolo_5class.yaml
output:
  num_classes: 5
  class_names: [clown, shark, spider, blood, needle]

architecture:
  grid_size: 13
  num_boxes_per_cell: 2
  in_channels: 3
  layers:
    - {filters: 16, pool: true}
    - {filters: 32, pool: true}
    # ... more layers
```

---

### 📂 `src/` - Source Code

Core implementation of PhobiaShield.

```
src/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── phobia_net.py              # Base single-scale model
│   ├── phobia_net_fpn.py          # Multi-scale FPN model ⭐
│   ├── loss.py                    # Basic YOLO loss
│   └── loss_fpn.py                # FPN loss with Focal Loss ⭐
├── data/
│   ├── __init__.py
│   └── phobia_dataset.py          # PyTorch Dataset class ⭐
├── training/
│   ├── __init__.py
│   └── metrics.py                 # mAP, precision, recall
├── inference/
│   ├── __init__.py
│   ├── predictor.py               # Inference wrapper
│   └── nms.py                     # Non-Maximum Suppression ⭐
└── utils/
    ├── __init__.py
    ├── visualization.py           # Plot utilities
    └── logging.py                 # Training logger
```

#### Key Files

**`src/models/phobia_net_fpn.py`**
- PhobiaNetFPN class (multi-scale detection)
- FPN neck (P3, P4, P5 scales)
- Detection heads
- **Owner:** Gabriele (Model Architect)

**`src/models/loss_fpn.py`**
- FPNLoss class
- Focal Loss implementation
- Multi-scale loss computation
- **Owner:** Gabriele

**`src/data/phobia_dataset.py`**
- PhobiaDataset class
- YOLO format label parsing
- Data augmentation
- **Owner:** Member A (Data Specialist)

**`src/inference/nms.py`**
- Non-Maximum Suppression
- Multi-class NMS
- IoU calculation
- **Owner:** Gabriele + Member C

---

### 📂 `scripts/` - Utility Scripts

Standalone scripts for training, evaluation, and data processing.

```
scripts/
├── train.py                       # Single-scale training
├── train_clean.py                 # Clean training script ⭐
├── train_yolov8.py                # YOLOv8 baseline training ⭐
├── evaluate.py                    # Model evaluation
├── merge_final_dataset.py         # Dataset merging script
├── setup_dataset.py               # Dataset setup automation ⭐
├── visualize_dataset.py           # Dataset visualization
└── download_from_roboflow.py      # Roboflow downloader
```

#### Key Scripts

**`scripts/train_clean.py`** ⭐
- FPN custom training
- Optimized hyperparameters
- Early stopping
- Weights & Biases logging (optional)

**Usage:**
```bash
python scripts/train_clean.py \
  --data data/phobiashield_ultimate \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.000346
```

**`scripts/train_yolov8.py`** ⭐
- YOLOv8 baseline training
- Transfer learning from COCO
- Automatic data.yaml generation

**Usage:**
```bash
python scripts/train_yolov8.py \
  --dataset data/phobiashield_ultimate \
  --epochs 50 \
  --batch 64
```

**`scripts/merge_final_dataset.py`**
- Merges multiple dataset sources
- Stratified train/val/test split (70/15/15)
- Duplicate removal
- Class ID remapping

**`scripts/setup_dataset.py`** ⭐
- Downloads dataset from Google Drive
- Extracts and organizes files
- Verifies integrity

---

### 📂 `notebooks/` - Training Notebooks

Google Colab notebooks for reproducible training.

```
notebooks/
├── 01_FPN_Training.ipynb          # FPN custom training ⭐
├── 02_YOLOv8_Training.ipynb       # YOLOv8 baseline ⭐
├── 03_Evaluation.ipynb            # Comparative analysis ⭐
├── 04_Inference_Demo.ipynb        # Video demo
└── 05_Dataset_Analysis.ipynb      # Dataset statistics
```

**Features:**
- Google Colab compatible
- Mount Drive automatically
- GPU (T4) accelerated
- Self-contained (includes setup)

**Usage:**
1. Open in Google Colab
2. Run all cells
3. Models saved to Drive

---

### 📂 `docs/` - Documentation

Project documentation and guides.

```
docs/
├── DATASET_ULTIMATE_README.md     # Dataset documentation ⭐
├── GIT_WORKFLOW.md                # Git collaboration guide ⭐
├── PROJECT_STRUCTURE.md           # This file ⭐
├── TEAM_ROLES.md                  # Team responsibilities ⭐
├── TRAINING_GUIDE.md              # Training best practices
└── API_REFERENCE.md               # Code documentation
```

---

### 📂 `presentation/` - Final Presentation

Materials for course presentation.

```
presentation/
├── phobiashield_slides.tex        # LaTeX Beamer slides ⭐
├── speaker_notes.md               # Speaker scripts ⭐
├── COMPILATION.md                 # Compilation guide ⭐
└── figures/
    ├── architecture_diagram.png
    ├── results_comparison.png
    └── demo_screenshot.png
```

**Slides Structure:**
1. Title & Introduction
2. Dataset Challenge
3. FPN Architecture
4. NMS Post-Processing
5. FPN Results
6. YOLOv8 Approach
7. Comparative Results
8. Conclusions + Demo

---

### 📂 `results/` - Training Results

Training outputs and analysis.

```
results/
├── README.md                      # Results overview ⭐
├── comparison.md                  # Detailed comparison ⭐
├── fpn_custom/
│   ├── best_model.pth            # Best checkpoint
│   ├── training_log.txt          # Loss curves
│   ├── confusion_matrix.png
│   └── metrics.json              # mAP, precision, recall
└── yolov8s/
    ├── train/
    │   ├── weights/
    │   │   ├── best.pt
    │   │   └── last.pt
    │   ├── results.png
    │   └── confusion_matrix.png
    └── metrics.json
```

---

### 📂 `data/` - Datasets (Gitignored)

Dataset storage (not in Git due to size).

```
data/
├── phobiashield_ultimate/         # DATASET_ULTIMATE_COMPLETE ⭐
│   ├── train/
│   │   ├── images/               # 7,593 images
│   │   └── labels/               # YOLO format
│   ├── val/
│   │   ├── images/               # 1,624 images
│   │   └── labels/
│   └── test/
│       ├── images/               # 1,634 images
│       └── labels/
├── old_dataset/                  # phobiashield_final (archived)
└── raw/                          # Raw downloads
```

**Download:**
- Google Drive (team access only)
- Use `scripts/setup_dataset.py` for automatic setup

---

### 📂 `outputs/` - Model Checkpoints (Gitignored)

Training outputs (not in Git due to size).

```
outputs/
├── checkpoints/
│   ├── best_model.pth            # Best model
│   ├── checkpoint_epoch_10.pth
│   └── checkpoint_epoch_20.pth
└── logs/
    ├── train_log.txt
    └── tensorboard/              # TensorBoard logs
```

---

## File Naming Conventions

### Python Files

- **Modules**: `lowercase_with_underscores.py`
- **Classes**: `CapitalizedWords` (PascalCase)
- **Functions**: `lowercase_with_underscores`
- **Constants**: `UPPERCASE_WITH_UNDERSCORES`

**Examples:**
```python
# File: src/models/phobia_net_fpn.py

class PhobiaNetFPN(nn.Module):  # PascalCase
    def __init__(self):
        self.num_classes = NUM_CLASSES  # UPPERCASE constant
    
    def forward(self, x):  # lowercase function
        return self._process_features(x)  # Private method
```

### Configuration Files

- Format: `name_version.yaml`
- Examples:
  - `tiny_yolo_5class.yaml`
  - `phobia_net_fpn_v2.yaml`

### Checkpoints

- Format: `{model}_{metric}_{value}.pth`
- Examples:
  - `fpn_best_e22_loss4.5031.pth`
  - `yolov8s_best.pt`

### Notebooks

- Format: `{number}_{description}.ipynb`
- Examples:
  - `01_FPN_Training.ipynb`
  - `02_YOLOv8_Training.ipynb`

---

## Important Files Reference

### Core Implementation Files ⭐

| File | Lines | Description | Owner |
|------|-------|-------------|-------|
| `src/models/phobia_net_fpn.py` | ~400 | Multi-scale FPN | Gabriele |
| `src/models/loss_fpn.py` | ~300 | Focal Loss + MSE | Gabriele |
| `src/data/phobia_dataset.py` | ~200 | Dataset class | Member A |
| `scripts/train_clean.py` | ~250 | Training script | Gabriele |
| `scripts/train_yolov8.py` | ~150 | YOLOv8 training | Member C |

### Documentation Files ⭐

| File | Purpose |
|------|---------|
| `README.md` | Project overview |
| `docs/DATASET_ULTIMATE_README.md` | Dataset info |
| `docs/GIT_WORKFLOW.md` | Git guide |
| `docs/TEAM_ROLES.md` | Team structure |
| `results/comparison.md` | Results analysis |

---

## Adding New Files

### Adding a New Model

1. Create file: `src/models/my_model.py`
2. Implement `nn.Module` subclass
3. Add config: `cfg/model/my_model.yaml`
4. Update `src/models/__init__.py`
5. Document in `docs/API_REFERENCE.md`

### Adding a New Script

1. Create file: `scripts/my_script.py`
2. Add argparse for CLI
3. Add docstring with usage
4. Update `README.md` with example

### Adding a Notebook

1. Create: `notebooks/0X_Title.ipynb`
2. Add Colab badge at top
3. Include setup cells
4. Document in `README.md`

---

## Dependencies

### Core Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.7.0
Pillow>=9.5.0
albumentations>=1.3.0
omegaconf>=2.3.0
tqdm>=4.65.0
```

### Optional Dependencies

```txt
ultralytics>=8.0.0    # For YOLOv8
wandb>=0.15.0         # For experiment tracking
tensorboard>=2.13.0   # For visualization
jupyter>=1.0.0        # For notebooks
```

### Installation

```bash
# Basic installation
pip install -r requirements.txt

# With all optional dependencies
pip install -r requirements-full.txt
```

---

## Build System

We use standard Python packaging:

```bash
# Install in editable mode
pip install -e .

# This allows:
from src.models import PhobiaNetFPN
from src.data import PhobiaDataset
```

**setup.py:**
```python
from setuptools import setup, find_packages

setup(
    name='phobiashield',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        # ... other deps
    ]
)
```

---

## Testing Structure (Future)

```
tests/
├── test_models.py          # Model tests
├── test_dataset.py         # Dataset tests
├── test_loss.py            # Loss function tests
└── test_nms.py             # NMS tests
```

**Run tests:**
```bash
pytest tests/
```

---

## CI/CD (Future)

```
.github/
└── workflows/
    ├── test.yml           # Run tests on push
    ├── lint.yml           # Code quality checks
    └── deploy.yml         # Deploy docs
```

---

## Size Estimates

| Directory | Size | Notes |
|-----------|------|-------|
| `data/` | 1.4 GB | Gitignored |
| `outputs/` | 500 MB | Gitignored |
| `src/` | 100 KB | Tracked |
| `notebooks/` | 5 MB | Tracked |
| `docs/` | 1 MB | Tracked |
| **Total (Git)** | ~10 MB | Excluding data/outputs |

---

## Quick Navigation

- 🏠 **Start here**: `README.md`
- 📊 **Dataset info**: `docs/DATASET_ULTIMATE_README.md`
- 🚀 **Training**: `notebooks/01_FPN_Training.ipynb`
- 📈 **Results**: `results/comparison.md`
- 🎤 **Presentation**: `presentation/phobiashield_slides.tex`
- 🔧 **Git guide**: `docs/GIT_WORKFLOW.md`
- 👥 **Team**: `docs/TEAM_ROLES.md`

---

## Contact

For questions about project structure:
- GitHub Issues: https://github.com/Gabriele-mp/PhobiaShield/issues
- See `docs/TEAM_ROLES.md` for team contacts
