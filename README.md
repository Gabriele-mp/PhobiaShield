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

---

## 📁 Struttura del Repository

```
PhobiaShield/
├── README.md                   # Questo file
├── requirements.txt            # Dipendenze Python
├── setup.py                    # Setup del package
├── .gitignore                 # File da ignorare in git
│
├── cfg/                       # 🔧 Configurazioni Hydra
│   ├── config.yaml           # Config principale
│   ├── model/                # Config modello
│   │   ├── tiny_yolo.yaml
│   │   └── baseline.yaml
│   ├── data/                 # Config dataset
│   │   ├── coco_phobia.yaml
│   │   └── augmentation.yaml
│   └── training/             # Config training
│       ├── default.yaml
│       └── fast_test.yaml
│
├── src/                      # 💻 Codice sorgente principale
│   ├── __init__.py
│   │
│   ├── data/                 # 📊 Data Management (Membro A)
│   │   ├── __init__.py
│   │   ├── dataset.py        # PhobiaDataset class
│   │   ├── augmentation.py   # Custom augmentations
│   │   ├── preprocessing.py  # Data preprocessing
│   │   └── download.py       # Script download datasets
│   │
│   ├── models/               # 🧠 Model Architecture (Membro B)
│   │   ├── __init__.py
│   │   ├── phobia_net.py     # PhobiaNet class
│   │   ├── backbone.py       # CNN backbone
│   │   ├── detection_head.py # Detection head
│   │   └── loss.py           # Custom loss function
│   │
│   ├── training/             # 🏋️ Training Logic (Membro B)
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training loop
│   │   ├── validator.py      # Validation logic
│   │   └── metrics.py        # mAP, IoU, etc.
│   │
│   ├── inference/            # 🎬 Deployment & Demo (Membro C)
│   │   ├── __init__.py
│   │   ├── detector.py       # Inference engine
│   │   ├── nms.py            # Non-Maximum Suppression
│   │   ├── video_processor.py # Video frame processing
│   │   └── blur.py           # ROI blurring
│   │
│   └── utils/                # 🛠️ Utility functions
│       ├── __init__.py
│       ├── visualization.py  # Plot bboxes, loss curves
│       ├── logger.py         # Logging setup
│       └── bbox_utils.py     # IoU, NMS utilities
│
├── scripts/                  # 📜 Script eseguibili
│   ├── download_data.sh      # Download datasets
│   ├── train.py              # Script training principale
│   ├── evaluate.py           # Valutazione modello
│   └── demo.py               # Demo interattiva
│
├── notebooks/                # 📓 Jupyter Notebooks (solo per analisi)
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_testing.ipynb
│   └── 03_results_analysis.ipynb
│
├── tests/                    # 🧪 Unit tests
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_loss.py
│
├── data/                     # 📦 Dataset (gitignored)
│   ├── raw/
│   ├── processed/
│   └── annotations/
│
├── outputs/                  # 📈 Training outputs (gitignored)
│   ├── checkpoints/
│   ├── logs/
│   └── videos/
│
├── docs/                     # 📚 Documentazione
│   ├── report.tex            # Report LaTeX
│   ├── slides.pptx           # Presentazione
│   └── architecture.png      # Diagrammi
│
└── app/                      # 🌐 Demo App
    ├── streamlit_app.py      # Interfaccia Streamlit
    └── utils.py              # Helper per app
```

---

## 🚀 Setup e Installazione

### 1. Clona il Repository
```bash
git clone https://github.com/your-team/PhobiaShield.git
cd PhobiaShield
```

### 2. Crea Virtual Environment
```bash
# Con conda (consigliato)
conda create -n phobiashield python=3.10
conda activate phobiashield

# Con venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Installa Dipendenze
```bash
pip install -r requirements.txt
pip install -e .  # Installa package in modalità development
```

### 4. Configura Weights & Biases
```bash
wandb login
# Inserisci la tua API key quando richiesto
```

### 5. Download Dataset
```bash
bash scripts/download_data.sh
```

---

## 💻 Utilizzo

### Training

#### Modalità Base (con config di default)
```bash
python scripts/train.py
```

#### Con Hydra Configuration
```bash
# Training completo
python scripts/train.py model=tiny_yolo data=coco_phobia training=default

# Test veloce (poche epoch)
python scripts/train.py model=baseline training=fast_test

# Override parametri
python scripts/train.py training.epochs=50 training.batch_size=16 training.lr=0.001
```

#### Training su Google Colab (con GPU)
```python
# Nel notebook Colab
!git clone https://github.com/your-team/PhobiaShield.git
%cd PhobiaShield
!pip install -r requirements.txt
!pip install -e .

# Training
!python scripts/train.py training.device=cuda
```

### Evaluation
```bash
# Valuta il modello sul test set
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth

# Calcola mAP
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth --metric map
```

### Demo Interattiva
```bash
# Avvia interfaccia Streamlit
streamlit run app/streamlit_app.py

# Oppure con Gradio
python scripts/demo.py --video path/to/video.mp4
```

### Inferenza su Video
```bash
python scripts/demo.py \
    --video data/videos/trailer.mp4 \
    --checkpoint outputs/checkpoints/best_model.pth \
    --output outputs/videos/blurred_trailer.mp4 \
    --blur-intensity 15
```

---

## 👥 Team e Ruoli (Strategia "ALL-IN")

**Nuova organizzazione**: Collaborazione totale su Dataset & Report + Coding specializzato

### 📊 FASE CONDIVISA (TUTTI)

#### 🗓️ Day 0-2: Caccia al Dato
- **Membro 1**: 🕷️ Spider dataset
- **Membro 2**: 🐍 Snake dataset
- **Membro 3**: 🩸 Blood dataset

Ognuno scarica/pulisce/converte la propria classe → merge in `all_phobias.zip`

#### 🗓️ Day 10-14: Report & Slide
- **Membro 1**: Sezione "Proposed Method" (Architettura + Loss)
- **Membro 2**: Sezione "Experimental Setup" (Augmentation + Training)
- **Membro 3**: Sezione "Application Results" (NMS + Demo)
- **TUTTI**: Introduction + Conclusions

---

### 💻 FASE TECNICA (CODING)

### 🏗️ Membro 1: THE ARCHITECT (Rete & Matematica)
**Focus**: Definire la struttura statica del cervello

**Tasks Principali**:
- ✅ Scrivere classe `PhobiaNet`
- ✅ Progettare layer sequence (Conv2d, BatchNorm, LeakyReLU)
- ⚠️ **Task Critico**: Implementare **Loss Function** (MSE + BCE + CE)
- ✅ Analisi risultati e calcolo mAP

**File**: `src/models/phobia_net.py`, `src/models/loss.py`, `src/training/metrics.py`

**Branch**: `feature/model-architecture`

---

### 🔄 Membro 2: THE TRAINER (Pipeline & Ottimizzazione)
**Focus**: Insegnare al cervello e gestire dati in ingresso

**Tasks Principali**:
- ✅ Scrivere Training Loop e DataLoader
- ✅ Gestire ciclo `for epoch in epochs`
- ⚠️ **Task Critico**: Implementare **Data Augmentation** (rotations, zoom, color jitter)
- ✅ Monitorare training su W&B

**File**: `src/data/dataset.py`, `src/data/augmentation.py`, `scripts/train.py`

**Branch**: `feature/training-pipeline`

---

### 🎬 Membro 3: THE ENGINEER (Inference & Demo)
**Focus**: Rendere i numeri visibili e creare l'applicazione

**Tasks Principali**:
- ✅ Scrivere Post-Processing (NMS) e Blurring
- ✅ Filtrare box sovrapposte
- ⚠️ **Task Critico**: Creare **interfaccia Streamlit** e **Video Trailer**
- ✅ Montare demo finale

**File**: `src/inference/nms.py`, `src/inference/video_processor.py`, `app/streamlit_app.py`

**Branch**: `feature/inference-demo`

---

**📚 Documentazione Completa**: Vedi `docs/TEAM_ROLES.md` per dettagli workflow

---

## 📅 Roadmap di Sviluppo (14 Giorni)

### Fase 1: Setup e Architettura (Giorni 1-4)
- [x] **Giorno 1**: Setup repo, ambiente, download dataset
- [ ] **Giorno 2-3**: Implementazione Loss Function + DataLoader
- [ ] **Giorno 4**: First training run (anche se modello non impara)

### Fase 2: Training e Integrazione (Giorni 5-9)
- [ ] **Giorno 5-6**: Debug training, monitoring loss
- [ ] **Giorno 7-8**: Overfitting check, model saving
- [ ] **Giorno 9**: Demo prep, video processing

### Fase 3: Showtime e Report (Giorni 10-14)
- [ ] **Giorno 10**: Benchmark (mAP calculation)
- [ ] **Giorno 11-12**: Slide presentation
- [ ] **Giorno 13-14**: Final polish, report LaTeX

---

## 🤝 Contribuire

### Git Workflow

1. **Crea il tuo branch**:
```bash
git checkout -b feature/nome-feature
```

2. **Lavora sul tuo codice**:
```bash
git add .
git commit -m "feat: descrizione significativa"
```

3. **Push al tuo branch**:
```bash
git push origin feature/nome-feature
```

4. **Apri Pull Request** su GitHub quando pronto

### Commit Messages Convention
Usa [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` - Nuova feature
- `fix:` - Bug fix
- `docs:` - Documentazione
- `refactor:` - Refactoring codice
- `test:` - Test
- `chore:` - Maintenance

### Best Practices
- ✅ Testa il codice prima di fare commit
- ✅ Scrivi commit message descrittive
- ✅ Fai pull di `main` prima di creare nuovi branch
- ✅ Risolvi i conflitti localmente
- ✅ Usa `.gitignore` per non committare file pesanti

---

## 📊 Experiment Tracking con W&B

Il progetto usa Weights & Biases per tracciare esperimenti:

```python
import wandb

# Login (una sola volta)
wandb.login()

# Nel training script
wandb.init(
    project="phobiashield",
    name="tiny-yolo-v1",
    config={
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 16
    }
)

# Log metriche
wandb.log({"loss": loss, "mAP": map_score})
```

Dashboard W&B: `https://wandb.ai/your-team/phobiashield`

---

## 📝 Note Importanti

### ⚠️ Vincoli "From Scratch"
- ❌ NO ultralytics, detectron2, o librerie high-level detection
- ✅ SI PyTorch/TensorFlow puro per rete e loss
- ✅ Implementazione manuale di NMS
- ✅ Custom training loop

### 🎯 Dataset Consigliati
- [COCO Subset (Spider, Snake)](https://cocodataset.org/)
- [Kaggle: Spider Detection Dataset](https://www.kaggle.com/)
- [Roboflow: Blood Detection](https://roboflow.com/)

### 🔥 GPU Recommendations
- **Google Colab**: Free T4 GPU (consigliato per training)
- **Kaggle Notebooks**: Free P100 GPU
- **Local**: NVIDIA GPU con CUDA support

---

## 📜 License

Questo progetto è rilasciato sotto licenza MIT. Vedi `LICENSE` per dettagli.

---

## 🙏 Acknowledgments

- Ispirato dalla repository [MNIST-FDS](https://github.com/Mamiglia/MNIST-FDS)
- Dataset: COCO, Kaggle, Roboflow
- Framework: PyTorch, Weights & Biases, Hydra

---

## 📧 Contatti

Per domande o suggerimenti, apri un Issue su GitHub!

**Team PhobiaShield** - Dicembre 2025
