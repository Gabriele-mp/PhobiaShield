# ⚡ PhobiaShield - Quick Start Guide

**Inizia subito a lavorare sul progetto in 5 minuti!**

---

## 🎯 Prerequisiti

- ✅ Python 3.8+
- ✅ Git installato
- ✅ Account GitHub
- ✅ Account Weights & Biases (gratuito)
- ✅ (Opzionale) Account Google Colab per GPU

---

## 🚀 Setup Rapido

### 1. Clone & Install (2 minuti)

```bash
# Clone repository
git clone https://github.com/your-team/PhobiaShield.git
cd PhobiaShield

# Crea virtual environment
conda create -n phobiashield python=3.10
conda activate phobiashield

# Installa dipendenze
pip install -r requirements.txt
pip install -e .

# Login W&B
wandb login
```

### 2. Crea il Tuo Branch (30 secondi)

```bash
# Scegli il branch in base al tuo ruolo:

# Membro A: Data Specialist
git checkout -b feature/data-pipeline

# Membro B: Model Architect  
git checkout -b feature/model-architecture

# Membro C: Deployment Engineer
git checkout -b feature/inference-demo
```

### 3. Verifica Setup (1 minuto)

```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import wandb; print('W&B: OK')"

# Test CUDA (se disponibile)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📋 Task per Ruolo (Strategia "ALL-IN")

### 🗓️ FASE CONDIVISA: TUTTI (Day 0-2 + Day 10-14)

**Day 0-2: Caccia al Dato**
- **Membro 1**: Scarica/pulisce Spider dataset
- **Membro 2**: Scarica/pulisce Snake dataset
- **Membro 3**: Scarica/pulisce Blood dataset
- **Risultato**: `all_phobias.zip`

**Day 10-14: Report**
- **Membro 1**: "Proposed Method" (Architettura + Loss)
- **Membro 2**: "Experimental Setup" (Augmentation + Training)
- **Membro 3**: "Application Results" (NMS + Demo)
- **TUTTI**: Introduction + Conclusions

---

### 🏗️ Membro 1: THE ARCHITECT (Rete & Matematica)

**File da creare/modificare:**
- `src/models/phobia_net.py` ✅ (già creato, da testare)
- `src/models/loss.py` ✅ (già creato, da debuggare)
- `src/training/metrics.py` 🔨 (TODO: mAP, IoU)
- `cfg/model/*.yaml` ✅ (già creato)

**Task prioritari:**
1. **Giorno 1-4**: Testare e debuggare PhobiaNet
   ```bash
   python src/models/phobia_net.py
   ```

2. **Giorno 5-8**: Debuggare e ottimizzare Loss Function
   ```bash
   python src/models/loss.py
   ```

3. **Giorno 9-11**: Analisi risultati e calcolo mAP
   ```bash
   python src/training/metrics.py
   ```

**Test veloce:**
```python
from src.models.phobia_net import create_model
from src.models.loss import PhobiaLoss
from omegaconf import OmegaConf
import torch

# Test model
config = OmegaConf.load("cfg/model/tiny_yolo.yaml")
config = OmegaConf.to_container(config, resolve=True)
model = create_model(config)

# Test forward
x = torch.randn(2, 3, 416, 416)
output = model(x)
print(f"Output shape: {output.shape}")

# Test loss
criterion = PhobiaLoss(grid_size=13, num_boxes=2, num_classes=3)
target = torch.zeros(2, 13, 13, 2*5+3)
loss, loss_dict = criterion(output, target)
print(f"Loss: {loss.item()}")
```

---

### 🔄 Membro 2: THE TRAINER (Pipeline & Ottimizzazione)

**File da creare/modificare:**
- `src/data/dataset.py` ✅ (già creato, da testare)
- `src/data/augmentation.py` 🔨 (TODO: implementare pipeline)
- `scripts/train.py` ✅ (già creato, da usare)
- `cfg/data/*.yaml` ✅ (già creato)

**Task prioritari:**
1. **Giorno 1-4**: Dataset e Augmentation
   ```bash
   python src/data/dataset.py
   python src/data/augmentation.py
   ```

2. **Giorno 5-8**: Training Loop
   ```bash
   # Test veloce
   python scripts/train.py training=fast_test
   
   # Training completo
   python scripts/train.py model=tiny_yolo training=default
   ```

3. **Giorno 9-11**: Passare pesi a Membro 3
   ```bash
   # Il file best_model.pth sarà in:
   outputs/checkpoints/best_model.pth
   ```

**Test veloce:**
```python
from src.data.dataset import PhobiaDataset, get_transforms

dataset = PhobiaDataset(
    root_dir="data/raw",
    annotations_file="data/annotations/train.json",
    image_size=(416, 416),
    transform=get_transforms(cfg, mode="train")
)

print(f"Dataset size: {len(dataset)}")
image, target = dataset[0]
print(f"Image: {image.shape}, Target: {target.shape}")
```

**Augmentation da implementare:**
```python
# src/data/augmentation.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

---

### 🎬 Membro 3: THE ENGINEER (Inference & Demo)

**File da creare/modificare:**
- `src/inference/nms.py` ✅ (già creato, da testare)
- `src/inference/detector.py` 🔨 (TODO: inference engine)
- `src/inference/video_processor.py` 🔨 (TODO: video processing)
- `src/inference/blur.py` 🔨 (TODO: ROI blurring)
- `app/streamlit_app.py` 🔨 (TODO: interfaccia)

**Task prioritari:**
1. **Giorno 1-4**: Testare NMS e utils
   ```bash
   python src/inference/nms.py
   ```

2. **Giorno 5-8**: Implementare video processing e blurring
   ```bash
   python src/inference/video_processor.py
   python src/inference/blur.py
   ```

3. **Giorno 9-11**: Demo video con pesi di Membro 2
   ```bash
   python scripts/demo.py \
       --video trailer.mp4 \
       --checkpoint outputs/checkpoints/best_model.pth \
       --output outputs/videos/demo_blurred.mp4
   ```

4. **Giorno 12-14**: Interfaccia Streamlit
   ```bash
   streamlit run app/streamlit_app.py
   ```

**Test veloce:**
```python
from src.inference.nms import nms

detections = [
    {"bbox": [0.5, 0.5, 0.2, 0.2], "confidence": 0.9, "class_id": 0},
    {"bbox": [0.52, 0.52, 0.21, 0.19], "confidence": 0.85, "class_id": 0},
    {"bbox": [0.3, 0.3, 0.15, 0.15], "confidence": 0.7, "class_id": 0},
]

filtered = nms(detections, iou_threshold=0.5, conf_threshold=0.3)
print(f"Detections: {len(detections)} → {len(filtered)}")
```

**Video Processor da implementare:**
```python
# src/inference/video_processor.py
import cv2
from src.models.phobia_net import create_model
from src.inference.nms import nms

class VideoProcessor:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
    
    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        # ... processamento frame-by-frame
        # 1. Detect
        # 2. NMS
        # 3. Blur ROI
        # 4. Save
```

---

## 🔥 Training su Google Colab

### Setup Colab (1 minuto)

1. Apri `notebooks/training_colab.ipynb`
2. Upload su Google Drive
3. Apri con Google Colab
4. Runtime → Change runtime type → GPU (T4)
5. Esegui le celle sequenzialmente

### Training Veloce (Test)

```python
# Nel notebook Colab
!python scripts/train.py \
    model=baseline \
    training=fast_test \
    training.device=cuda \
    wandb.name=test-run
```

### Training Completo

```python
!python scripts/train.py \
    model=tiny_yolo \
    training=default \
    training.epochs=100 \
    training.device=cuda \
    wandb.name=full-training-v1
```

---

## 📊 Monitor Training

### Weights & Biases

1. Vai su: https://wandb.ai/your-username/phobiashield
2. Visualizza:
   - Loss curves (train/val)
   - Learning rate
   - Epoch time
   - Sample predictions

### Locale

```bash
# Visualizza log
tail -f outputs/logs/training.log

# Lista checkpoints
ls -lh outputs/checkpoints/

# Tensorboard (opzionale)
tensorboard --logdir outputs/logs/
```

---

## 🔄 Git Workflow Quotidiano

### Mattina (inizio lavoro)

```bash
# Update main
git checkout main
git pull origin main

# Update tuo branch
git checkout feature/your-branch
git merge main

# Inizia a lavorare!
```

### Sera (fine lavoro)

```bash
# Controlla modifiche
git status

# Aggiungi file
git add src/your-files.py

# Commit
git commit -m "feat: description of work"

# Push
git push origin feature/your-branch
```

### Ogni 2-3 giorni

```bash
# Apri Pull Request su GitHub
# Base: main, Compare: feature/your-branch
# Richiedi review ai compagni
```

---

## 🧪 Test Rapidi

### Test Dataset

```bash
python -c "
from src.data.dataset import PhobiaDataset
ds = PhobiaDataset('data/raw', 'data/annotations/train.json')
print(f'✓ Dataset: {len(ds)} samples')
"
```

### Test Model

```bash
python -c "
from src.models.phobia_net import create_model
from omegaconf import OmegaConf
cfg = OmegaConf.to_container(OmegaConf.load('cfg/model/tiny_yolo.yaml'), resolve=True)
model = create_model(cfg)
print(f'✓ Model: {sum(p.numel() for p in model.parameters()):,} params')
"
```

### Test Loss

```bash
python src/models/loss.py
```

### Test NMS

```bash
python src/inference/nms.py
```

---

## 📚 Documentazione Completa

- **README.md**: Panoramica completa del progetto
- **docs/GIT_WORKFLOW.md**: Guida dettagliata Git
- **notebooks/training_colab.ipynb**: Tutorial training
- **cfg/**: Tutte le configurazioni

---

## 🆘 Problemi Comuni

### "ModuleNotFoundError"

```bash
# Assicurati di aver installato il package
pip install -e .
```

### "CUDA out of memory"

```python
# Riduci batch size
python scripts/train.py training.batch_size=8
```

### "Git conflict"

```bash
# Vedi docs/GIT_WORKFLOW.md sezione "Risoluzione Conflitti"
```

### "W&B login error"

```bash
# Re-login
wandb login --relogin
```

---

## 🎯 Prossimi Passi

### Settimana 1 (Giorni 1-4)

- [ ] Setup completo
- [ ] Primo commit su proprio branch
- [ ] Test di base funzionanti
- [ ] Dataset scaricato

### Settimana 2 (Giorni 5-9)

- [ ] Training loop funzionante
- [ ] Primo modello addestrato
- [ ] Demo di base funzionante
- [ ] Prima Pull Request

### Settimana 3 (Giorni 10-14)

- [ ] Training ottimizzato
- [ ] Demo completa
- [ ] Report scritto
- [ ] Presentazione pronta

---

## 💬 Comunicazione Team

### Canali

- **GitHub Issues**: Per task e bug
- **GitHub Projects**: Per tracking avanzamento
- **Pull Requests**: Per review codice
- **Slack/Discord**: Per comunicazione veloce

### Daily Sync (5 min)

Ogni giorno, condividere:
1. Cosa ho fatto ieri
2. Cosa farò oggi
3. Eventuali blocchi/problemi

---

## 🎉 Checklist Primo Giorno

- [ ] Repository clonata
- [ ] Environment creato
- [ ] Dipendenze installate
- [ ] W&B configurato
- [ ] Branch creato
- [ ] Primo commit fatto
- [ ] Test di base eseguiti
- [ ] Documentazione letta

---

**Sei pronto! Inizia a lavorare e buona fortuna! 🚀**

Per domande: apri un Issue su GitHub o chiedi al team!
