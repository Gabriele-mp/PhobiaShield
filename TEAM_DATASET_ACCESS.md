# 📦 PhobiaShield - Dataset Access for Team

## ⚡ Quick Access

**Dataset Finale Merged (5 classi):**
- 📊 **Total Images:** 2354
- 🏷️ **Classes:** Clown, Shark, Spider, Blood, Needle
- 📂 **Format:** YOLO (txt labels)
- 📏 **Split:** 70% train / 15% val / 15% test

---

## 🚀 Download & Setup (3 minuti)

### Opzione 1: Google Drive (RACCOMANDATO - veloce)

#### 1. Download dal Drive Condiviso

**Link Google Drive:** `[INSERISCI LINK QUI]`

📥 File: `phobiashield_final.zip` (171 MB)

#### 2. Setup nel Progetto

```bash
# Nel tuo PhobiaShield directory
cd ~/Desktop/PhobiaShield

# Unzip
unzip phobiashield_final.zip -d data/

# Verifica
ls data/phobiashield_final/train/images | wc -l  # Should be 1647
ls data/phobiashield_final/val/images | wc -l    # Should be 353
ls data/phobiashield_final/test/images | wc -l   # Should be 354
```

#### 3. Ready!

```python
# Test caricamento
from src.data.phobia_dataset import PhobiaDataset

dataset = PhobiaDataset(
    'data/phobiashield_final/train/images',
    'data/phobiashield_final/train/labels',
    img_size=416,
    grid_size=13,
    num_boxes=2,
    num_classes=5
)

print(f"Dataset size: {len(dataset)}")  # Should be 1647
```

---

## 📁 Struttura Dataset

```
data/phobiashield_final/
├── train/
│   ├── images/           # 1647 immagini (70%)
│   │   ├── clown_001.jpg
│   │   ├── shark_042.jpg
│   │   └── ...
│   └── labels/           # 1647 file .txt
│       ├── clown_001.txt
│       ├── shark_042.txt
│       └── ...
├── val/
│   ├── images/           # 353 immagini (15%)
│   └── labels/           # 353 file .txt
└── test/
    ├── images/           # 354 immagini (15%)
    └── labels/           # 354 file .txt
```

### Formato Label (YOLO)

Ogni `.txt` file contiene una riga per oggetto:

```
<class_id> <center_x> <center_y> <width> <height>
```

**Esempio** (`clown_001.txt`):
```
0 0.512 0.345 0.234 0.456
```

- `class_id`: 0=Clown, 1=Shark, 2=Spider, 3=Blood, 4=Needle
- Tutte le coordinate sono normalizzate [0, 1]

---

## 📊 Statistiche Dataset

### Distribuzione Classi (Objects)

| Class | Train | Val | Test | Total | Percentage |
|-------|-------|-----|------|-------|------------|
| Clown (0) | 739 | 158 | 159 | 1056 | 10% |
| Shark (1) | 352 | 75 | 76 | 503 | 5% |
| Spider (2) | 451 | 96 | 97 | 644 | 6% |
| Blood (3) | 5871 | 1258 | 1258 | 8387 | 79% |
| Needle (4) | 66 | 14 | 14 | 94 | 1% |
| **TOTAL** | **7479** | **1601** | **1604** | **10684** | **100%** |

⚠️ **Dataset sbilanciato:** Blood domina (79%), Needle scarso (1%)

### Distribuzione Immagini

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 1647 | 70% |
| Val | 353 | 15% |
| Test | 354 | 15% |
| **TOTAL** | **2354** | **100%** |

### Provenienza Dataset

- **Clown:** Roboflow (794 images)
- **Shark:** Open Images Dataset (400 images)
- **Spider:** Dataset Marco (634 images)
- **Blood:** Dataset Marco (471 images)
- **Needle:** Dataset Marco (55 images)

---

## 💻 Utilizzo nel Codice

### 1. Training con train.py (Trainer)

```bash
# Training completo
python scripts/train.py \
    model=tiny_yolo_5class \
    data=phobia_final \
    training.epochs=50 \
    training.batch_size=8
```

### 2. Training con train_clean.py (The Architect)

```bash
# Quick testing
python scripts/train_clean.py \
    --data data/phobiashield_final \
    --config cfg/model/tiny_yolo_5class.yaml \
    --epochs 10 \
    --batch-size 8
```

### 3. Custom DataLoader

```python
from torch.utils.data import DataLoader
from src.data.phobia_dataset import PhobiaDataset

# Create dataset
train_dataset = PhobiaDataset(
    'data/phobiashield_final/train/images',
    'data/phobiashield_final/train/labels',
    img_size=416,
    grid_size=13,
    num_boxes=2,
    num_classes=5,
    augment=True  # Augmentation per training
)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=PhobiaDataset.collate_fn  # Importante!
)

# Iterate
for images, targets in train_loader:
    # images: [8, 3, 416, 416]
    # targets: [8, 13, 13, 20]
    pass
```

---

## 🔧 Configurazione

### Config File: `cfg/data/phobia_final.yaml`

```yaml
data:
  root: "data/phobiashield_final"
  num_classes: 5
  class_names: ["Clown", "Shark", "Spider", "Blood", "Needle"]
  img_size: 416
  grid_size: 13

dataloader:
  batch_size: 8
  num_workers: 4
  shuffle: true

augmentation:
  train:
    horizontal_flip: 0.5
    brightness_contrast: 0.2
    hue_saturation: 0.1
  val:
    # No augmentation
```

### Config Model: `cfg/model/tiny_yolo_5class.yaml`

```yaml
architecture:
  input_size: 416
  grid_size: 13
  num_boxes_per_cell: 2

output:
  num_classes: 5
  class_names: ["Clown", "Shark", "Spider", "Blood", "Needle"]

loss:
  lambda_coord: 5.0
  lambda_obj: 1.0
  lambda_noobj: 0.5
  lambda_class: 1.0
```

---

## 🐛 Troubleshooting

### Problema: "ValueError: bbox out of range"

**Soluzione:** Assicurati di usare `phobia_dataset.py` con i bbox fixes (già su main):

```python
from src.data.phobia_dataset import PhobiaDataset  # ✅ Questa versione ha i fix
```

### Problema: "IndexError: class_id"

**Soluzione:** Già fixato in `phobia_dataset.py` (cast to int). Assicurati di aver pullato da main.

### Problema: Dataset non trovato

```bash
# Verifica path
ls data/phobiashield_final/train/images | head -5

# Se vuoto, unzip di nuovo
unzip phobiashield_final.zip -d data/
```

### Problema: Troppo lento su CPU

```python
# Disabilita augmentation per velocizzare
dataset = PhobiaDataset(..., augment=False)

# Riduci num_workers
train_loader = DataLoader(..., num_workers=0)  # Single thread
```

---

## 🔄 Opzione 2: Rigenerare Localmente (Avanzato)

Se hai accesso ai dataset originali:

### Prerequisiti

```
~/Desktop/Marco_Data/
  Blood_ID3/
    images/ (471 images)
    labels/
  Needles_ID4/
    images/ (55 images)
    labels/

~/Desktop/Phobia/
  images/ (634 spider images)
  labels/

PhobiaShield/data/raw/
  clown/ (da download Roboflow - 794 images)
  shark/ (da download Open Images - 400 images)
```

### Step 1: Download Clown e Shark

```bash
cd ~/Desktop/PhobiaShield

# Clown dataset (Roboflow)
# Scarica manualmente da Roboflow e metti in data/raw/clown/

# Shark dataset (Open Images)
python scripts/download_shark.py  # Se hai lo script
```

### Step 2: Merge

```bash
# Esegui merge script
python scripts/merge_final_dataset.py

# Output: data/phobiashield_final/ con tutti i file
```

**Tempo:** ~10 minuti (dipende da velocità disco)

---

## 📈 Performance Attese

### Training 10 Epochs (Quick Test)

```
Tempo: ~15 min su Tesla T4 (Google Colab)
Val Loss: ~17.5
mAP@0.5: 0.00-0.10 (normale, troppo poco training)
```

### Training 50 Epochs (Production)

```
Tempo: ~75 min su Tesla T4
Val Loss: ~8-12
mAP@0.5: 0.40-0.60 (BUONO)
```

### Training 100 Epochs (Ottimale)

```
Tempo: ~150 min su Tesla T4
Val Loss: ~5-8
mAP@0.5: 0.60-0.75 (ECCELLENTE)
```

---

## 🎯 Tips & Best Practices

### 1. Class Balancing

Dataset sbilanciato (Blood 79%, Needle 1%). Considera:

```python
# Opzione A: Class weights nella loss
class_weights = torch.tensor([1.0, 2.0, 2.0, 0.5, 10.0])  # Penalizza Blood, boost Needle

# Opzione B: Weighted sampling
from torch.utils.data import WeightedRandomSampler
# Implementa sampling che bilancia le classi
```

### 2. Augmentation Strategy

```python
# Training: Aggressive augmentation
train_dataset = PhobiaDataset(..., augment=True)

# Validation/Test: No augmentation
val_dataset = PhobiaDataset(..., augment=False)
```

### 3. Monitoring

```python
import wandb

wandb.init(project="phobiashield", name="my-experiment")

for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss = validate()
    
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch": epoch
    })
```

### 4. Checkpoint Management

```python
# Salva best model
if val_loss < best_val_loss:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, 'outputs/checkpoints/best_model.pth')
```

---

## 📚 Documentazione Componenti

### PhobiaDataset (The Architect)

**File:** `src/data/phobia_dataset.py`

**Features:**
- ✅ Bbox clipping [0, 1]
- ✅ Overflow prevention
- ✅ class_id cast to int
- ✅ Albumentations integration
- ✅ collate_fn per DataLoader

**Usage:**
```python
dataset = PhobiaDataset(
    images_dir='data/phobiashield_final/train/images',
    labels_dir='data/phobiashield_final/train/labels',
    img_size=416,
    grid_size=13,
    num_boxes=2,
    num_classes=5,
    augment=True
)
```

### PhobiaNet (The Architect)

**File:** `src/models/phobia_net.py`

**Architecture:**
- 6 convolutional layers
- 1.58M parameters (~6.3 MB)
- Input: [B, 3, 416, 416]
- Output: [B, 13, 13, 20] (grid predictions)

### PhobiaLoss (The Architect)

**File:** `src/models/loss.py`

**Components:**
- Coordinate Loss (MSE): λ=5.0
- Objectness Loss (BCE): λ=1.0
- No-object Loss (BCE): λ=0.5
- Classification Loss (CE): λ=1.0

---

## 🤝 Chi Ha Fatto Cosa

### The Architect (Gabriele)
- ✅ PhobiaNet architecture
- ✅ PhobiaLoss implementation
- ✅ PhobiaDataset with fixes
- ✅ Metrics (mAP, IoU)
- ✅ Dataset merge script
- ✅ train_clean.py

### The Trainer (Compagno)
- ✅ train.py (professional pipeline)
- ✅ Hydra configuration
- ✅ W&B integration
- ✅ dataset.py (alternative)

### The Demo Engineer (Compagno 3)
- ⏳ NMS implementation
- ⏳ Video processing
- ⏳ Streamlit UI
- ⏳ Final demo

---

## ❓ FAQ

**Q: Posso usare dataset.py invece di phobia_dataset.py?**  
A: Sì, ma phobia_dataset.py ha i bbox fixes testati. Usa quella per evitare crash.

**Q: Come carico un checkpoint salvato?**  
A:
```python
checkpoint = torch.load('outputs/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

**Q: Il training è troppo lento su CPU?**  
A: Usa Google Colab con GPU gratuita. Vedi `notebooks/PhobiaShield_Training_CLEAN.ipynb`

**Q: mAP è 0 dopo 10 epochs, è normale?**  
A: SÌ! YOLO from scratch richiede 50+ epochs per convergere. 10 epochs = solo early learning.

**Q: Come aggiungo una nuova classe?**  
A: Modifica `num_classes` in config, aggiungi dataset con class_id corretto, ri-merge.

---

## 📞 Supporto

**Issues su GitHub:** https://github.com/Gabriele-mp/PhobiaShield/issues

**Team Contact:**
- The Architect (Model): @Gabriele
- The Trainer (Pipeline): @CompagnoTrainer
- The Engineer (Demo): @CompagnoDemoEngineer

---

## ✅ Checklist Rapida

Prima di iniziare il training:

- [ ] Dataset scaricato e unzippato in `data/phobiashield_final/`
- [ ] Verificato: 1647 train + 353 val + 354 test images
- [ ] Config file aggiornati (`cfg/model/tiny_yolo_5class.yaml`, `cfg/data/phobia_final.yaml`)
- [ ] Virtual environment attivato
- [ ] Dipendenze installate (`pip install -r requirements.txt`)
- [ ] W&B configurato (`wandb login`)
- [ ] GPU disponibile (Colab o locale)
- [ ] `phobia_dataset.py` usato (con fix)

---

**Dataset creato da:** The Architect  
**Ultimo aggiornamento:** Dicembre 6, 2025  
**Versione:** 1.0 (final merge, 5 classes)

🎯 **Ready to train! Buon lavoro team!** 🚀
