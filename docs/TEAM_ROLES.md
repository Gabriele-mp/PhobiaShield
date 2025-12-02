# 👥 PhobiaShield - Team Roles & Responsibilities

**Strategia "ALL-IN": Collaborazione Totale + Coding Ripartito**

Dicembre 2025

---

## 🎯 Nuova Strategia

### Fase Condivisa: DATI E REPORT
**Responsabilità**: TUTTI i membri

### Fase Tecnica: CODING
**Divisione**: 3 ruoli specializzati ma collaborativi

---

## 📊 FASE CONDIVISA: DATI E REPORT (TUTTI)

### 🗓️ Day 0-2: Caccia al Dato

Ogni membro sceglie **una classe diversa** e lavora in parallelo:

- **Membro 1**: 🕷️ Spider (Ragni)
- **Membro 2**: 🐍 Snake (Serpenti)  
- **Membro 3**: 🩸 Blood (Sangue)

**Task per ognuno:**
1. Scaricare dataset per la propria classe
2. Pulire e validare immagini
3. Convertire annotazioni in formato YOLO
4. Creare file `class_name.zip` (es. `spiders.zip`)

**Risultato finale**: 3 dataset parziali uniti in `all_phobias.zip`

**Comandi:**
```bash
# Ogni membro sul proprio branch
git checkout -b data/spiders    # Membro 1
git checkout -b data/snakes     # Membro 2
git checkout -b data/blood      # Membro 3

# Struttura directory
data/
├── raw/
│   ├── spiders/
│   ├── snakes/
│   └── blood/
└── annotations/
    ├── spiders.json
    ├── snakes.json
    └── blood.json
```

---

### 🗓️ Day 10-14: Report & Slide

**Divisione sezioni del report:**

#### 📝 Membro 1: "Proposed Method"
- Architettura della rete (diagramma PhobiaNet)
- Spiegazione matematica della Loss Function
- Scelte di design (perché Tiny-YOLO?)

**File da scrivere:**
- `docs/report_method.tex` (sezione LaTeX)
- `docs/slides_architecture.pptx`

---

#### 📝 Membro 2: "Experimental Setup"
- Data Augmentation utilizzata
- Training procedure e hyperparameters
- Grafici loss curves (train/val)

**File da scrivere:**
- `docs/report_experiments.tex` (sezione LaTeX)
- `docs/slides_training.pptx`

---

#### 📝 Membro 3: "Application Results"
- NMS algorithm e parametri
- Demo video risultati
- Analisi qualitativa (dove funziona/fallisce)

**File da scrivere:**
- `docs/report_results.tex` (sezione LaTeX)
- `docs/slides_demo.pptx`

---

#### 📝 TUTTI: Introduction & Conclusions
- Introduzione: problema delle fobie
- Conclusioni: cosa abbiamo imparato
- Future work

**Modalità**: Meeting collaborativo per scrivere insieme

---

## 💻 FASE TECNICA: CODING (3 RUOLI)

---

## 🏗️ MEMBRO 1: THE ARCHITECT (Rete & Matematica)

### Focus
Definire la **struttura statica del cervello**

### Task Principale
✅ Scrivere la classe `PhobiaNet`

### Dettagli
- Progettare sequenza di layer (Conv2d, BatchNorm, LeakyReLU)
- Calcolare dimensioni esatte tensori
- Assicurare output compatibile con griglia detection (7×7×30 o 13×13×24)

### Task Critico ⚠️
**Implementare la Loss Function**
- Tradurre matematica (MSE + BCE + CE) in codice PyTorch
- Componenti:
  - Localization Loss (MSE su bbox coordinates)
  - Confidence Loss (BCE su objectness)
  - Classification Loss (CE su classi)

### File di Responsabilità
```
src/models/
├── phobia_net.py          ✅ (già implementato, da testare/debuggare)
├── loss.py                ✅ (già implementato, da debuggare)
├── backbone.py            🔨 TODO: separare backbone
└── detection_head.py      🔨 TODO: separare detection head
```

### Workflow
```bash
# Branch
git checkout -b feature/model-architecture

# Day 1-2: Setup
python src/models/phobia_net.py  # Test model

# Day 3-4: Loss function
python src/models/loss.py  # Test loss

# Day 5-8: Debug & tuning
# Collaborare con Membro 2 per integrare nel training

# Day 9-11: Analisi risultati
# Calcolare mAP sul test set
# Analizzare dove il modello sbaglia
```

### Metrics da Implementare
```python
# src/training/metrics.py
def compute_map(predictions, targets):
    """Mean Average Precision"""
    pass

def compute_iou(box1, box2):
    """Intersection over Union"""
    pass

def analyze_errors(predictions, targets):
    """Error analysis: FP, FN per class"""
    pass
```

---

## 🔄 MEMBRO 2: THE TRAINER (Pipeline & Ottimizzazione)

### Focus
**Insegnare al cervello** e gestire i dati in ingresso

### Task Principale
✅ Scrivere il `Training Loop` e il `DataLoader`

### Dettagli
- Scrivere ciclo `for epoch in epochs`
- Gestire passaggio dati alla GPU
- Salvare checkpoint pesi
- Monitorare metriche (loss, mAP)

### Task Critico ⚠️
**Implementare Data Augmentation**
- Rotazioni
- Zoom
- Color jittering
- Gaussian blur/noise
- **Al volo durante training** (non pre-processing)

### File di Responsabilità
```
src/data/
├── dataset.py             ✅ (già implementato)
├── augmentation.py        🔨 TODO: implementare pipeline completa
└── preprocessing.py       🔨 TODO: preprocessing utils

scripts/
└── train.py               ✅ (già implementato, da testare)

src/training/
├── trainer.py             🔨 TODO (opzionale, già in scripts/train.py)
└── validator.py           🔨 TODO: validation logic
```

### Workflow
```bash
# Branch
git checkout -b feature/training-pipeline

# Day 1-2: Dataset & Augmentation
python src/data/dataset.py  # Test dataset
python src/data/augmentation.py  # Test augmentation

# Day 3-4: Integrare con Model (da Membro 1)
# Collaborare per far girare primo epoch

# Day 5-8: Training Loop
# Avviare training completo su Colab
python scripts/train.py training=fast_test  # Test
python scripts/train.py model=tiny_yolo     # Full training

# Monitorare su W&B:
# - Loss curves (train/val)
# - Learning rate
# - Sample predictions

# Day 9-11: Passare pesi a Membro 3
# outputs/checkpoints/best_model.pth
```

### Augmentation Pipeline
```python
# src/data/augmentation.py
import albumentations as A

def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.2, 
            rotate_limit=15, 
            p=0.3
        ),
        A.HueSaturationValue(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo', 
        label_fields=['class_labels']
    ))
```

---

## 🎬 MEMBRO 3: THE ENGINEER (Inference & Demo)

### Focus
Rendere i **numeri visibili** e creare l'applicazione

### Task Principale
✅ Scrivere logica di **Post-Processing** (NMS) e **Blurring**

### Dettagli
- Il modello di Membro 1 produce numeri grezzi (bbox, confidence, class)
- Scrivere algoritmo NMS per filtrare box sovrapposte
- Disegnare/sfumare rettangoli sull'immagine

### Task Critico ⚠️
**Creare interfaccia e Video Trailer**
- Interfaccia Gradio/Streamlit
- Processare video frame-by-frame
- Applicare blur su ROI (Region of Interest)
- Montare demo finale

### File di Responsabilità
```
src/inference/
├── nms.py                 ✅ (già implementato)
├── detector.py            🔨 TODO: inference engine
├── video_processor.py     🔨 TODO: video processing
└── blur.py                🔨 TODO: ROI blurring

src/utils/
├── visualization.py       🔨 TODO: plot bboxes
└── bbox_utils.py          🔨 TODO: bbox operations

app/
├── streamlit_app.py       🔨 TODO: Streamlit interface
└── utils.py               🔨 TODO: App helpers

scripts/
└── demo.py                🔨 TODO: Demo script
```

### Workflow
```bash
# Branch
git checkout -b feature/inference-demo

# Day 1-4: NMS & Utils
python src/inference/nms.py  # Test NMS
python src/utils/bbox_utils.py  # Test IoU, bbox utils

# Day 5-8: Blurring su immagini statiche
# Testare effetto blur con modello "dummy"
python src/inference/blur.py

# Creare funzioni:
# - draw_boxes()
# - blur_roi()
# - overlay_label()

# Day 9-11: Video Processing
# Ricevere best_model.pth da Membro 2
# Applicare su video trailer

python scripts/demo.py \
    --video trailer.mp4 \
    --checkpoint outputs/checkpoints/best_model.pth \
    --output outputs/videos/demo_blurred.mp4

# Day 12-14: Interfaccia Streamlit
streamlit run app/streamlit_app.py
```

### Video Processing Pipeline
```python
# src/inference/video_processor.py
import cv2
from src.models.phobia_net import create_model
from src.inference.nms import nms
from src.inference.blur import blur_roi

class VideoProcessor:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = load_model(model_path)
        self.conf_threshold = conf_threshold
    
    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Detect objects
            detections = self.model.predict(frame)
            
            # 2. Apply NMS
            filtered = nms(detections, iou_threshold=0.5)
            
            # 3. Blur ROIs
            for det in filtered:
                frame = blur_roi(frame, det['bbox'], intensity=15)
            
            # 4. (Optional) Draw boxes
            frame = draw_boxes(frame, filtered)
            
            out.write(frame)
        
        cap.release()
        out.release()
```

### Streamlit App
```python
# app/streamlit_app.py
import streamlit as st
from src.inference.detector import PhobiaDetector

st.title("🛡️ PhobiaShield - Object Detection Demo")

# Sidebar
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)
blur_intensity = st.sidebar.slider("Blur Intensity", 0, 50, 15)

# Upload
uploaded_file = st.file_uploader("Upload Image or Video", 
                                 type=['jpg', 'png', 'mp4'])

if uploaded_file:
    # Process
    detector = PhobiaDetector(model_path="best_model.pth")
    result = detector.process(uploaded_file, 
                             conf=conf_threshold,
                             blur=blur_intensity)
    
    # Display
    st.image(result) or st.video(result)
```

---

## 📅 ROADMAP INTEGRATA (GANTT)

### 🏗️ GIORNI 1-4: FONDAZIONI

**TUTTI:**
- Trovare dataset
- Definire specifiche (416×416 input)
- Setup environment

**Membro 1 (Architect):**
```bash
# Scrivere model.py
cd src/models
# Implementare PhobiaNet class
# Test: python phobia_net.py
```

**Membro 2 (Trainer):**
```bash
# Scrivere dataset.py
cd src/data
# Implementare PhobiaDataset
# Implementare augmentation
# Test: python dataset.py
```

**Membro 3 (Engineer):**
```bash
# Scrivere utils.py
cd src/utils
# Funzioni bbox (draw, IoU)
# Test blurring
# Test: python bbox_utils.py
```

---

### 🔥 GIORNI 5-8: TRAINING LOGIC

**Membro 1 (Architect):**
- Completare Loss Function (parte più difficile!)
- Debug con batch dummy
- Collaborare con Membro 2 per integrare

**Membro 2 (Trainer):**
- Assemblare tutto in `train.py`
- Primo training su Colab
- Monitorare loss curves su W&B

**Membro 3 (Engineer):**
- Scrivere NMS da zero
- Testare blurring su immagini statiche
- Usare modello "dummy" per test visivi

---

### 🎬 GIORNI 9-11: INTEGRAZIONE DEMO

**Membro 2 → Membro 3:**
- Passare file `best_model.pth` (pesi addestrati)

**Membro 3:**
- Caricare pesi in script inferenza
- Applicare modello al **Trailer Video**
- Processare frame-by-frame con blur

**Membro 1:**
- Analizzare risultati
- Dove sbaglia il modello?
- Calcolare **mAP** sul test set
- Creare confusion matrix

---

### 📊 GIORNI 12-14: PRESENTAZIONE (TUTTI)

**Membro 1:**
- Slide su Architettura
- Slide su Loss Function
- Diagrammi tecnici

**Membro 2:**
- Slide su Dataset e Augmentation
- Grafici training curves
- Slide su hyperparameters

**Membro 3:**
- Montare video demo
- Slide "Live Demo"
- Preparare interfaccia Streamlit

**TUTTI:**
- Review finale report LaTeX
- Prove orali (5 min per sezione)
- Q&A preparation

---

## ✅ CHECKLIST FINALE PER I 30 E LODE

### 📦 Dataset
- [ ] Dati delle 3 classi uniti correttamente?
- [ ] Annotazioni in formato YOLO valide?
- [ ] Split train/val/test bilanciati?

### 💻 Code
- [ ] Membro 1: `model.py` e `loss.py` pushati?
- [ ] Membro 2: `dataset.py` e `train.py` pushati?
- [ ] Membro 3: `nms.py`, `video_processor.py`, `app.py` pushati?
- [ ] Codice ben commentato?
- [ ] README aggiornato?

### 📝 Report
- [ ] 3 sezioni tecniche (Method, Setup, Results) coerenti?
- [ ] Grafici di alta qualità?
- [ ] Citazioni corrette?
- [ ] Abstract chiaro?

### 🎥 Demo
- [ ] Video si vede bene ed è fluido?
- [ ] Blur applicato correttamente?
- [ ] Interfaccia funzionante?
- [ ] Side-by-side comparison (original vs processed)?

---

## 🤝 Comunicazione Team

### Daily Sync (5 min ogni giorno)
1. Cosa ho fatto ieri?
2. Cosa farò oggi?
3. Blocchi/problemi?

### Integration Points
- **Day 4**: Membro 1 + 2 integrano model + dataset
- **Day 8**: Membro 2 + 3 testano inference con pesi
- **Day 11**: TUTTI testano demo completa

### Git Strategy
```bash
# Branch per fase
main
├── data/spiders (Membro 1 - dataset)
├── data/snakes (Membro 2 - dataset)
├── data/blood (Membro 3 - dataset)
├── feature/model-architecture (Membro 1)
├── feature/training-pipeline (Membro 2)
└── feature/inference-demo (Membro 3)

# Merge regolari
# Day 4: Merge data branches → main
# Day 8: Merge technical branches → main
# Day 11: Final integration
```

---

## 🎯 Summary per Membro

| Membro | Focus | File Chiave | Task Critico |
|--------|-------|-------------|--------------|
| **1: Architect** | Rete & Math | `phobia_net.py`, `loss.py` | Loss Function |
| **2: Trainer** | Pipeline & Opt | `dataset.py`, `train.py`, `augmentation.py` | Data Augmentation |
| **3: Engineer** | Inference & Demo | `nms.py`, `video_processor.py`, `streamlit_app.py` | Video Demo |

---

**Tutti contribuiscono a Dataset e Report!**

**Buon lavoro! 🚀**
