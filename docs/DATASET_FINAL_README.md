# PhobiaShield Final Dataset

## 📦 Overview

Dataset completo con 5 classi per PhobiaShield:
- **Clown** (ID: 0)
- **Shark** (ID: 1)
- **Spider** (ID: 2)
- **Blood** (ID: 3)
- **Needle** (ID: 4)

---

## 📊 Statistics

```
Total: 2354 images
Train: 1647 (70%)
Val:   353 (15%)
Test:  354 (15%)

Class Distribution:
- Clown:  1056 objects (10%)
- Shark:   503 objects (5%)
- Spider:  644 objects (6%)
- Blood:  8387 objects (79%)
- Needle:   94 objects (1%)
```

---

## 🔧 How to Generate

### Prerequisites

1. Scarica i dataset individuali nel Desktop:
   ```
   ~/Desktop/Marco_Data/
     Blood_ID3/
       images/
       labels/
     Needles_ID4/
       images/
       labels/
   
   ~/Desktop/Phobia/
     images/
     labels/
   ```

2. Assicurati che clown e shark siano in `data/raw/`:
   ```bash
   python scripts/download_clown.py
   python scripts/download_shark.py
   ```

### Generate Dataset

```bash
python scripts/merge_final_dataset.py
```

Output: `data/phobiashield_final/`

---

## 📁 Structure

```
data/phobiashield_final/
  train/
    images/
      clown_img001.jpg
      shark_img002.jpg
      spider_img003.jpg
      blood_img004.jpg
      needle_img005.jpg
      ...
    labels/
      clown_img001.txt
      shark_img002.txt
      ...
  val/
    images/
    labels/
  test/
    images/
    labels/
```

---

## 📝 Label Format (YOLO)

```
<class_id> <center_x> <center_y> <width> <height>
```

All values normalized to [0, 1]

Example:
```
3 0.5123 0.6234 0.1234 0.2345
```
→ Blood object at center (0.51, 0.62) with size (0.12, 0.23)

---

## ⚠️ Important Notes

### Class Filtering

The merge script automatically:
- **Blood dataset:** Keeps ONLY class_id=3 (blood), discards 30+ other COCO classes
- **Needle dataset:** Keeps ONLY class_id=4 (needles)
- This ensures clean, single-class annotations

### Dataset Imbalance

Blood dominates (79% of objects). For training:
- Use class weighting in loss function
- Consider data augmentation for minority classes (Needle especially)
- Monitor per-class metrics during validation

---

## 🎯 Usage in Training

Update config:

```yaml
# cfg/data/phobia_final.yaml
data:
  root: "data/phobiashield_final"
  num_classes: 5
```

Train:

```bash
python scripts/train_complete.py \
    model=tiny_yolo_5class \
    data=phobia_final \
    training.epochs=50
```

---

## 📊 Quality Checks

After generation, verify:

```bash
# Count images per split
ls data/phobiashield_final/train/images | wc -l    # Should be ~1647
ls data/phobiashield_final/val/images | wc -l      # Should be ~353
ls data/phobiashield_final/test/images | wc -l     # Should be ~354

# Check class distribution
python scripts/analyze_dataset.py data/phobiashield_final
```

---

## 🔄 Updating Dataset

If you add more images to source datasets:

1. Delete old merged dataset:
   ```bash
   rm -rf data/phobiashield_final
   ```

2. Re-run merge:
   ```bash
   python scripts/merge_final_dataset.py
   ```

The script will create a new random split (70/15/15).

---

## 📞 Support

- **Issues:** https://github.com/Gabriele-mp/PhobiaShield/issues
- **Team:** PhobiaShield Project
- **Module:** The Architect (dataset & model architecture)

---

**Last Updated:** December 2024
**Version:** 1.0 (5 classes, 2354 images)
