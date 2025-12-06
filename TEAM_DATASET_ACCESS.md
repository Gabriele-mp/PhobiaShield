# 📦 PhobiaShield - Dataset Access for Team

## ⚡ Quick Access

**Dataset Finale Merged (5 classi):**
- 📊 **Total Images:** 2354
- 🏷️ **Classes:** Clown, Shark, Spider, Blood, Needle
- 📂 **Format:** YOLO (txt labels)
- 📏 **Split:** 70% train / 15% val / 15% test

---

## 🚀 Download & Setup (3 minuti)

### 📥 Download dal Google Drive

**Link Dataset:** https://drive.google.com/file/d/1AhfNK2S9RJ_i-m_A4FZngCBr6Ea_3RaX/view?usp=share_link

📦 File: `phobiashield_final.zip` (171 MB)

---

## 💻 Setup su Kaggle (RACCOMANDATO)

### Perché Kaggle?

| | **Kaggle** | **Google Colab** |
|---|-----------|------------------|
| GPU Time | **30h/settimana** 🏆 | 4h/giorno ⏱️ |
| GPU Type | P100 (16GB) / T4 | T4 (15GB) |
| Storage | 20GB persistent | Session-based |
| Best for | **Training lungo (50-100 epochs)** | Quick testing |

**Conclusione:** Usa Kaggle per production training! ✅

### Setup Kaggle Notebook

#### 1. Crea Nuovo Notebook

1. Vai su https://www.kaggle.com
2. Click **"Code"** → **"New Notebook"**
3. Settings (a destra):
   - **Accelerator:** GPU T4 x2 (o P100 se disponibile)
   - **Persistence:** Files only
   - **Internet:** ON

#### 2. Upload Dataset su Kaggle

**Opzione A: Crea Dataset Kaggle (consigliato)**

```python
# Nel notebook Kaggle, cella 1:

# Download da Google Drive
!pip install -q gdown
!gdown 1AhfNK2S9RJ_i-m_A4FZngCBr6Ea_3RaX -O phobiashield_final.zip

# Unzip
!unzip -q phobiashield_final.zip -d /kaggle/working/data/

# Verifica
!ls /kaggle/working/data/phobiashield_final/train/images | wc -l  # Should be 1647
```

**Opzione B: Upload come Kaggle Dataset**

1. Vai su https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload `phobiashield_final.zip`
4. Titolo: "PhobiaShield Final Dataset"
5. Publish (private o public)
6. Nel notebook: Add Data → tuo dataset

#### 3. Clone Repository

```python
# Cella 2: Clone repo
!git clone https://github.com/Gabriele-mp/PhobiaShield.git
%cd PhobiaShield

# Install dependencies
!pip install -q -r requirements.txt
!pip install -e .
```

#### 4. Verifica GPU

```python
# Cella 3: Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

#### 5. Training

```python
# Cella 4: Start training
!python scripts/train_clean.py \
    --data /kaggle/working/data/phobiashield_final \
    --config cfg/model/tiny_yolo_5class.yaml \
    --epochs 50 \
    --batch-size 8 \
    --lr 0.0001
```

**Tempo stimato:** ~75 min per 50 epochs su T4

---

## 🔧 Setup Locale (Mac/Linux)

```bash
# Nel tuo PhobiaShield directory
cd ~/Desktop/PhobiaShield

# Download da Google Drive (manuale o con gdown)
# Opzione 1: Download browser, poi:
cp ~/Downloads/phobiashield_final.zip .

# Opzione 2: Con gdown
pip install gdown
gdown 1AhfNK2S9RJ_i-m_A4FZngCBr6Ea_3RaX

# Unzip
unzip phobiashield_final.zip -d data/

# Verifica
ls data/phobiashield_final/train/images | wc -l  # Should be 1647
ls data/phobiashield_final/val/images | wc -l    # Should be 353
ls data/phobiashield_final/test/images | wc -l   # Should be 354
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

---

## 💻 Utilizzo nel Codice

### 1. Training con train.py (Trainer)

```bash
python scripts/train.py \
    model=tiny_yolo_5class \
    data=phobia_final \
    training.epochs=50 \
    training.batch_size=8
```

### 2. Training con train_clean.py (The Architect)

```bash
python scripts/train_clean.py \
    --data data/phobiashield_final \
    --config cfg/model/tiny_yolo_5class.yaml \
    --epochs 50 \
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

## 📈 Performance Attese

### Training su Kaggle (T4 GPU)

| Epochs | Tempo | Val Loss | mAP@0.5 | Status |
|--------|-------|----------|---------|--------|
| 10 | ~15 min | ~17.5 | 0.00-0.10 | Early learning ⏳ |
| 30 | ~45 min | ~12-15 | 0.20-0.35 | Learning bbox 📈 |
| 50 | ~75 min | ~8-12 | 0.40-0.60 | **GOOD** ✅ |
| 100 | ~150 min | ~5-8 | 0.60-0.75 | **EXCELLENT** 🏆 |

**Raccomandazione:** 50 epochs minimum per production model

### Limiti GPU

| Platform | GPU Time | Best For |
|----------|----------|----------|
| **Kaggle** | 30h/settimana | Production training (50-100 epochs) |
| **Google Colab** | 4h/giorno | Quick testing (10-20 epochs) |
| **Locale** | Illimitato (se hai GPU) | Development & debugging |

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

**Soluzione:** Assicurati di usare `phobia_dataset.py` con i bbox fixes:

```python
from src.data.phobia_dataset import PhobiaDataset  # ✅ Questa versione ha i fix
```

### Problema: "IndexError: class_id"

**Soluzione:** Già fixato in `phobia_dataset.py`. Pull da main:

```bash
git checkout main
git pull origin main
```

### Problema: Kaggle dataset non trovato

```python
# Su Kaggle, usa path assoluto
dataset = PhobiaDataset(
    '/kaggle/working/data/phobiashield_final/train/images',
    '/kaggle/working/data/phobiashield_final/train/labels',
    ...
)
```

### Problema: Google Drive download lento

```python
# Usa gdown con progress bar
!pip install gdown
!gdown --fuzzy 1AhfNK2S9RJ_i-m_A4FZngCBr6Ea_3RaX
```

### Problema: Out of Memory su GPU

```python
# Riduci batch size
!python scripts/train_clean.py --batch-size 4  # Invece di 8

# Oppure disabilita augmentation temporaneamente
dataset = PhobiaDataset(..., augment=False)
```

---

## 📓 Kaggle Notebook Template

Copia-incolla questo template per iniziare velocemente:

```python
# ============================================================
# CELL 1: Setup
# ============================================================
!pip install -q gdown
!gdown 1AhfNK2S9RJ_i-m_A4FZngCBr6Ea_3RaX -O phobiashield_final.zip
!unzip -q phobiashield_final.zip -d /kaggle/working/data/

# ============================================================
# CELL 2: Clone Repo
# ============================================================
!git clone https://github.com/Gabriele-mp/PhobiaShield.git
%cd PhobiaShield
!pip install -q -r requirements.txt
!pip install -e .

# ============================================================
# CELL 3: Verify GPU
# ============================================================
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Verify dataset
!ls /kaggle/working/data/phobiashield_final/train/images | wc -l

# ============================================================
# CELL 4: Start Training (50 epochs)
# ============================================================
!python scripts/train_clean.py \
    --data /kaggle/working/data/phobiashield_final \
    --config cfg/model/tiny_yolo_5class.yaml \
    --epochs 50 \
    --batch-size 8 \
    --lr 0.0001

# ============================================================
# CELL 5: Evaluate
# ============================================================
# Load best model and calculate mAP
# (usa la cella metrics che hai già testato)

# ============================================================
# CELL 6: Download Model
# ============================================================
from IPython.display import FileLink
FileLink('outputs/checkpoints/best_model.pth')
```

**Tempo totale:** ~90 minuti (incluso setup)

---

## 🎯 Workflow Consigliato

### Per Quick Testing (1-2 ore)
```
Platform: Google Colab
Epochs: 10-20
Batch Size: 8
Goal: Verificare che tutto funziona
```

### Per Production Training (2-3 ore)
```
Platform: Kaggle
Epochs: 50-100
Batch Size: 8
Goal: Model production-ready (mAP > 0.5)
```

### Per Development
```
Platform: Locale (Mac/Linux)
Task: Debug codice, test features
No GPU needed: Usa batch_size=1, poche immagini
```

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

**Q: Meglio Kaggle o Colab?**  
A: **Kaggle** per training lungo (50+ epochs). Colab solo per quick test.

**Q: Come carico un checkpoint salvato?**  
A:
```python
checkpoint = torch.load('outputs/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

**Q: mAP è 0 dopo 10 epochs, è normale?**  
A: SÌ! YOLO from scratch richiede 50+ epochs. 10 epochs = early learning.

**Q: Posso scaricare il model da Kaggle?**  
A: SÌ! Usa:
```python
from IPython.display import FileLink
FileLink('outputs/checkpoints/best_model.pth')
```

**Q: Kaggle session scade?**  
A: Sì dopo 12h inattività. Ma i file in `/kaggle/working/` rimangono. Salva checkpoint spesso!

**Q: Come uso più GPU su Kaggle?**  
A: Settings → Accelerator → "GPU T4 x2" (doppia velocità!)

---

## 📞 Supporto

**Issues GitHub:** https://github.com/Gabriele-mp/PhobiaShield/issues

**Dataset Google Drive:** https://drive.google.com/file/d/1AhfNK2S9RJ_i-m_A4FZngCBr6Ea_3RaX/view

**Team Contact:**
- The Architect (Model): @Gabriele
- The Trainer (Pipeline): @CompagnoTrainer
- The Engineer (Demo): @CompagnoDemoEngineer

---

## ✅ Checklist Rapida

Prima di iniziare il training:

- [ ] Dataset scaricato da Google Drive
- [ ] Unzippato in `data/phobiashield_final/` (locale) o `/kaggle/working/data/` (Kaggle)
- [ ] Verificato: 1647 train + 353 val + 354 test images
- [ ] GPU attiva (Kaggle T4 o P100)
- [ ] Repository clonata
- [ ] Dipendenze installate
- [ ] Config files verificati
- [ ] `phobia_dataset.py` usato (con fix)
- [ ] Tempo GPU disponibile: almeno 2h per 50 epochs

---

**Dataset creato da:** The Architect  
**Ultimo aggiornamento:** Dicembre 6, 2025  
**Versione:** 1.0 (final merge, 5 classes)

🎯 **Ready to train su Kaggle! Buon lavoro team!** 🚀
