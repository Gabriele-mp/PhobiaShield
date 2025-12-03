# PhobiaShield - Complete Training Guide
## The Architect Module - Training & Testing

---

## 📋 OVERVIEW

Questa guida ti porta da **zero a model trained** in 3 step:

1. **Download mini dataset** (50 img/classe, ~150 totali)
2. **Train PhobiaNet** (5-50 epochs)
3. **Test su dati reali** (inference + mAP)

---

## 🚀 QUICK START - Google Colab (RECOMMENDED)

### Step 1: Apri Notebook

1. Vai su https://colab.research.google.com
2. Upload: `complete_training_pipeline.ipynb`
3. Runtime > Change runtime type > **GPU (T4)**
4. Run All (esegui tutte le celle)

**Tempo totale:** ~20-30 minuti per training completo

---

## 📦 DATASET - Opzioni

### Opzione 1: Mini COCO Dataset (AUTO)

Script scarica automaticamente:
- **COCO val2017** subset
- **50 immagini per classe**
- Classi proxy: person, cat, dog → clown, shark, spider

```bash
python scripts/download_mini_dataset.py
```

**Output:**
```
data/mini_dataset/
  train/
    images/  (105 imgs)
    labels/  (105 labels)
  val/
    images/  (22 imgs)
    labels/  (22 labels)
  test/
    images/  (23 imgs)
    labels/  (23 labels)
```

### Opzione 2: Dataset Reale (MANUALE)

Se hai dataset vero per clown/shark/spider:

1. **Struttura richiesta:**
```
data/my_dataset/
  train/
    images/
      img001.jpg
      img002.jpg
    labels/
      img001.txt  # YOLO format
      img002.txt
  val/
    images/
    labels/
  test/
    images/
    labels/
```

2. **Label format (YOLO):**
```
<class_id> <center_x> <center_y> <width> <height>
```
Valori normalizzati [0,1]

Esempio:
```
0 0.5 0.5 0.3 0.4  # clown al centro, size 0.3x0.4
```

3. **Modifica config:**
```yaml
# cfg/data/custom.yaml
data:
  root: "data/my_dataset"
  num_classes: 3
```

---

## 🎯 TRAINING

### Fast Test (5 epochs, ~5 min)

```bash
python scripts/train_complete.py training=fast_test
```

Output atteso:
```
Epoch 1: Train Loss=240.06, Val Loss=235.12
Epoch 2: Train Loss=185.23, Val Loss=180.45
...
Epoch 5: Train Loss=98.34, Val Loss=95.67
✓ Best model saved
```

### Full Training (50 epochs, ~30 min)

```bash
python scripts/train_complete.py training.epochs=50
```

### Con Weights & Biases Logging

```bash
# Login W&B (first time)
wandb login

# Train con logging
python scripts/train_complete.py logging.use_wandb=true
```

---

## 📊 MONITORING TRAINING

### 1. Terminal Output

```
Epoch 5 Summary:
  Train Loss: 142.67
  Val Loss:   138.92
  LR: 0.000100

✓ Best model saved (val_loss: 138.92)
```

### 2. W&B Dashboard (se abilitato)

- Real-time loss curves
- Learning rate schedule
- Gradient norms
- Model parameters

Dashboard: https://wandb.ai/your-username/phobiashield

### 3. Saved Checkpoints

```
outputs/checkpoints/
  best_model.pth          # Migliore su validation
  checkpoint_epoch_10.pth
  checkpoint_epoch_20.pth
  ...
```

---

## 🧪 TESTING

### Test Script (mAP calculation)

```python
# test.py
from omegaconf import OmegaConf
from src.models.phobia_net import PhobiaNet
import torch

# Load model
model_cfg = OmegaConf.load('cfg/model/tiny_yolo.yaml')
model = PhobiaNet(model_cfg).to('cuda')

# Load weights
checkpoint = torch.load('outputs/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Test
from src.training.metrics import calculate_map
map_score = calculate_map(model, test_loader, 'cuda')
print(f"mAP@0.5: {map_score:.4f}")
```

### Visual Inference

```python
# Predict on single image
predictions = model.predict(img_tensor, conf_threshold=0.5)

# predictions[0] = [
#   {'class_id': 0, 'confidence': 0.85, 'bbox': [0.5, 0.3, 0.2, 0.4]},
#   ...
# ]
```

---

## 📈 PERFORMANCE METRICS

### What to expect:

| Metric | Fast Test (5 ep) | Full Train (50 ep) |
|--------|------------------|---------------------|
| Training Time | ~5 min | ~30 min |
| Final Loss | ~100-120 | ~20-40 |
| mAP@0.5 | ~0.15-0.25 | ~0.40-0.60 |
| FPS (GPU) | ~30 | ~30 |

### Loss Components Breakdown:

```
Epoch 50:
  coord_loss: 0.45      (localizzazione)
  conf_loss_obj: 1.23   (confidence oggetto)
  conf_loss_noobj: 38.2 (confidence background)
  class_loss: 0.31      (classification)
  total_loss: 41.2
```

**Good signs:**
- ✅ Total loss scende monotonicamente
- ✅ conf_noobj scende (model impara "no")
- ✅ coord_loss si stabilizza ~0.3-0.5
- ✅ Val loss segue train loss (no overfit)

**Red flags:**
- ❌ Loss aumenta o oscilla wildly
- ❌ Val loss >> Train loss (overfit)
- ❌ coord_loss > 2.0 (coordinate sbagliate)

---

## 🔧 HYPERPARAMETER TUNING

### Learning Rate

```bash
# Default: 0.0001
python scripts/train_complete.py training.learning_rate=0.0005

# Se loss oscilla: riduci
python scripts/train_complete.py training.learning_rate=0.00005

# Se loss piatta: aumenta
python scripts/train_complete.py training.learning_rate=0.0002
```

### Batch Size

```bash
# Default: 8
python scripts/train_complete.py training.batch_size=16  # Se hai RAM GPU

# Se out of memory: riduci
python scripts/train_complete.py training.batch_size=4
```

### Lambda Weights

```bash
# Se coordinate imprecise: aumenta lambda_coord
python scripts/train_complete.py loss.lambda_coord=7.0

# Se troppi falsi positivi: aumenta lambda_noobj
python scripts/train_complete.py loss.lambda_noobj=0.7
```

---

## 📂 FILES OVERVIEW

```
scripts/
  download_mini_dataset.py    # Download COCO subset
  train_complete.py           # Training script principale
  test.py                     # Test e valutazione

notebooks/
  complete_training_pipeline.ipynb  # All-in-one Colab

src/
  models/
    phobia_net.py             # Model architecture
    loss.py                   # YOLO loss function
  data/
    phobia_dataset.py         # Dataset class
  training/
    metrics.py                # mAP, IoU, etc.

cfg/
  config.yaml                 # Main config
  model/
    tiny_yolo.yaml            # Model config
  training/
    default.yaml              # Training config
    fast_test.yaml            # Quick test config
  data/
    coco_phobia.yaml          # Data config
```

---

## 🎓 CHECKLIST COMPLETA

### Pre-Training:
- [ ] GPU available (Colab T4 o locale)
- [ ] Dataset scaricato (mini COCO o custom)
- [ ] Repo clonata e installata
- [ ] Config verificato

### Training:
- [ ] Fast test (5 ep) completato
- [ ] Loss scende correttamente
- [ ] Best model salvato
- [ ] Checkpoints presenti

### Testing:
- [ ] Model caricato
- [ ] mAP calcolato (>0.15 per fast test)
- [ ] Inference visiva funziona
- [ ] Detections sensate

---

## 🚨 TROUBLESHOOTING

### Out of Memory (GPU)
```bash
# Riduci batch size
training.batch_size=4

# Riduci workers
training.num_workers=0
```

### Loss NaN
```bash
# Riduci learning rate drasticamente
training.learning_rate=0.00001

# Abilita gradient clipping
training.grad_clip=1.0
```

### Loss non scende
```bash
# Verifica dataset
python scripts/check_dataset.py

# Aumenta learning rate
training.learning_rate=0.0005

# Aumenta epochs
training.epochs=100
```

---

## 📞 SUPPORT

- **Issues:** https://github.com/Gabriele-mp/PhobiaShield/issues
- **Docs:** `/docs` folder
- **Team:** The Architect module

---

**The Architect** 🏗️
PhobiaShield Project - 14 Days Challenge
