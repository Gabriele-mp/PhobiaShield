# Dataset ULTIMATE_COMPLETE

## Overview

**DATASET_ULTIMATE_COMPLETE** is the final, merged dataset for PhobiaShield training.

- **Total images**: 11,425
- **Classes**: 5 (Clown, Shark, Spider, Blood, Needle)
- **Format**: YOLO (normalized coordinates)
- **Split**: Stratified 70% train / 15% val / 15% test by dominant class
- **Location**: Google Drive (team access only)

---

## Download & Setup

### Option 1: Manual Download (Team Members)

1. Access Google Drive: `PhobiaShield_Models/PhobiaShield/DATASET_ULTIMATE_COMPLETE.zip`
2. Download zip file (1.36 GB)
3. Extract to `data/phobiashield_ultimate/`

### Option 2: Automated Setup (Recommended)

```bash
python scripts/setup_dataset.py
```

This script will:
- Check if dataset exists locally
- Prompt for Drive download if needed
- Extract and organize files
- Verify image counts

---

## Statistics

### Split Distribution

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 7,593  | 66.4%      |
| Val   | 1,624  | 14.2%      |
| Test  | 1,634  | 14.3%      |
| **Total** | **11,425** | **100%** |

### Class Distribution

| Class  | Train | Val | Test | Total |
|--------|-------|-----|------|-------|
| Clown  | 2,119 | 454 | 479  | 3,052 |
| Shark  | 1,862 | 399 | 422  | 2,683 |
| Spider | 1,531 | 328 | 347  | 2,206 |
| Blood  | 1,288 | 276 | 292  | 1,856 |
| Needle | 738   | 158 | 166  | 1,062 |

**Note:** Stratified split ensures balanced class distribution across splits.

---

## Size Analysis

### Object Sizes (Normalized)

| Class  | Min     | Mean  | Max   | Std Dev |
|--------|---------|-------|-------|---------|
| Needle | 0.00327 | 0.034 | 0.098 | 0.028   |
| Spider | 0.015   | 0.089 | 0.234 | 0.067   |
| Blood  | 0.023   | 0.156 | 0.445 | 0.112   |
| Clown  | 0.042   | 0.312 | 0.756 | 0.184   |
| Shark  | 0.089   | 0.487 | 0.850 | 0.223   |

### Pixel-Level Analysis (at 416×416)

| Class  | Min (px) | Mean (px) | Max (px) |
|--------|----------|-----------|----------|
| Needle | 1.36     | 14.1      | 40.8     |
| Spider | 6.24     | 37.0      | 97.3     |
| Blood  | 9.57     | 64.9      | 185.1    |
| Clown  | 17.5     | 129.8     | 314.5    |
| Shark  | 37.0     | 202.6     | 353.6    |

**Key insight:** 260× size variation (1.36px to 354px) - this is why multi-scale detection (FPN) is essential.

---

## Format

### YOLO Format

Each label file (`*.txt`) contains one line per object:

```
<class_id> <center_x> <center_y> <width> <height>
```

All coordinates normalized to [0, 1].

### Class Mapping

```python
CLASS_NAMES = {
    0: 'clown',
    1: 'shark',
    2: 'spider',
    3: 'blood',
    4: 'needle'
}
```

---

## Directory Structure

```
DATASET_ULTIMATE_COMPLETE/
├── train/
│   ├── images/
│   │   ├── clown_0001.jpg
│   │   ├── shark_0023.jpg
│   │   └── ...
│   └── labels/
│       ├── clown_0001.txt
│       ├── shark_0023.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## Data Quality

### Quality Assurance

- ✅ All images validated (loadable by OpenCV)
- ✅ All coordinates clipped to [0, 1]
- ✅ Minimum object size: 0.001 (filters noise)
- ✅ No duplicate images across splits
- ✅ Stratified split by dominant class

### Known Issues

1. **Clown variation**: High appearance diversity (traditional, horror, makeup styles)
2. **Shark context**: 78% open-mouth/aggressive poses (may bias detection)
3. **Needle ambiguity**: Mix of syringes (90%) and medical needles (10%)

These issues represent real-world phobia triggers and are acceptable for the application.

---

## Dataset Sources

The ULTIMATE_COMPLETE dataset is a merger of:

1. **Old dataset** (`phobiashield_final.zip`)
2. **New dataset** (`FINAL_DATASET.zip`)
3. **Ragni (spiders)** - additional spider images
4. **Clown** - Roboflow dataset
5. **Background images** - negative examples

Merged using `scripts/merge_final_dataset.py` with:
- Duplicate removal
- Class ID remapping
- Stratified splitting
- Quality filtering

---

## Usage in Training

### PyTorch Example

```python
from src.data.phobia_dataset import PhobiaDataset

train_dataset = PhobiaDataset(
    img_dir='data/phobiashield_ultimate/train/images',
    label_dir='data/phobiashield_ultimate/train/labels',
    img_size=416,
    num_classes=5,
    augment=True
)
```

### YOLOv8 Example

```yaml
# data.yaml
path: data/phobiashield_ultimate
train: train/images
val: val/images
test: test/images

nc: 5
names: ['clown', 'shark', 'spider', 'blood', 'needle']
```

---

## Citation

If you use this dataset, please cite:

```
@misc{phobiashield2025,
  title={PhobiaShield: Custom Object Detection for Phobia Management},
  author={Team PhobiaShield},
  year={2025},
  publisher={Sapienza University of Rome}
}
```

---

## Version History

- **v1.0** (Dec 2025): Initial ULTIMATE_COMPLETE release
  - 11,425 images
  - 5 classes
  - Stratified 70/15/15 split

---

## Contact

For dataset access or questions:
- GitHub Issues: https://github.com/Gabriele-mp/PhobiaShield/issues
- Team: See `docs/TEAM_ROLES.md`
