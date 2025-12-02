# 🏗️ PhobiaShield - Project Structure

**Documentazione completa della struttura del progetto**

Generato: Dicembre 2025

---

## 📁 Struttura Completa

```
PhobiaShield/
│
├── 📄 README.md                     # Documentazione principale
├── 📄 QUICKSTART.md                 # Guida rapida per iniziare
├── 📄 LICENSE                       # Licenza MIT
├── 📄 requirements.txt              # Dipendenze Python
├── 📄 setup.py                      # Package setup
├── 📄 .gitignore                    # File da ignorare in Git
│
├── 📂 cfg/                          # ⚙️ Configurazioni Hydra
│   ├── config.yaml                 # Config principale
│   ├── model/                      # Config modelli
│   │   ├── tiny_yolo.yaml         # Tiny-YOLO architecture
│   │   └── baseline.yaml          # Baseline model
│   ├── data/                       # Config dataset
│   │   ├── coco_phobia.yaml       # Dataset configuration
│   │   └── augmentation.yaml     # Data augmentation
│   └── training/                   # Config training
│       ├── default.yaml           # Default training
│       └── fast_test.yaml         # Fast test config
│
├── 📂 src/                          # 💻 Codice sorgente
│   ├── __init__.py
│   │
│   ├── 📂 data/                    # 📊 Data Management (Membro A)
│   │   ├── __init__.py
│   │   ├── dataset.py             # ✅ PhobiaDataset class
│   │   ├── augmentation.py        # 🔨 TODO: Data augmentation
│   │   ├── preprocessing.py       # 🔨 TODO: Preprocessing
│   │   └── download.py            # 🔨 TODO: Download script
│   │
│   ├── 📂 models/                  # 🧠 Model Architecture (Membro B)
│   │   ├── __init__.py
│   │   ├── phobia_net.py          # ✅ PhobiaNet model
│   │   ├── loss.py                # ✅ Custom loss function
│   │   ├── backbone.py            # 🔨 TODO: CNN backbone
│   │   └── detection_head.py      # 🔨 TODO: Detection head
│   │
│   ├── 📂 training/                # 🏋️ Training Logic (Membro B)
│   │   ├── __init__.py
│   │   ├── trainer.py             # 🔨 TODO: Training loop
│   │   ├── validator.py           # 🔨 TODO: Validation
│   │   └── metrics.py             # 🔨 TODO: mAP, IoU, etc.
│   │
│   ├── 📂 inference/               # 🎬 Inference & Demo (Membro C)
│   │   ├── __init__.py
│   │   ├── nms.py                 # ✅ Non-Maximum Suppression
│   │   ├── detector.py            # 🔨 TODO: Inference engine
│   │   ├── video_processor.py     # 🔨 TODO: Video processing
│   │   └── blur.py                # 🔨 TODO: ROI blurring
│   │
│   └── 📂 utils/                   # 🛠️ Utilities
│       ├── __init__.py
│       ├── visualization.py       # 🔨 TODO: Plotting
│       ├── logger.py              # 🔨 TODO: Logging
│       └── bbox_utils.py          # 🔨 TODO: Box utilities
│
├── 📂 scripts/                      # 📜 Executable Scripts
│   ├── train.py                   # ✅ Training script
│   ├── evaluate.py                # 🔨 TODO: Evaluation
│   ├── demo.py                    # 🔨 TODO: Demo
│   └── download_data.sh           # 🔨 TODO: Data download
│
├── 📂 notebooks/                    # 📓 Jupyter Notebooks
│   ├── training_colab.ipynb       # ✅ Colab training notebook
│   ├── 01_data_exploration.ipynb  # 🔨 TODO: Data analysis
│   ├── 02_model_testing.ipynb     # 🔨 TODO: Model testing
│   └── 03_results_analysis.ipynb  # 🔨 TODO: Results analysis
│
├── 📂 tests/                        # 🧪 Unit Tests
│   ├── test_dataset.py            # 🔨 TODO: Dataset tests
│   ├── test_model.py              # 🔨 TODO: Model tests
│   └── test_loss.py               # 🔨 TODO: Loss tests
│
├── 📂 data/                         # 📦 Dataset (gitignored)
│   ├── raw/                       # Raw images
│   ├── processed/                 # Processed data
│   └── annotations/               # Annotation files
│
├── 📂 outputs/                      # 📈 Training Outputs (gitignored)
│   ├── checkpoints/               # Model checkpoints
│   ├── logs/                      # Training logs
│   └── videos/                    # Processed videos
│
├── 📂 docs/                         # 📚 Documentation
│   ├── GIT_WORKFLOW.md            # ✅ Git workflow guide
│   ├── report.tex                 # 🔨 TODO: LaTeX report
│   ├── slides.pptx                # 🔨 TODO: Presentation
│   └── architecture.png           # 🔨 TODO: Architecture diagram
│
└── 📂 app/                          # 🌐 Demo Application
    ├── streamlit_app.py           # 🔨 TODO: Streamlit interface
    └── utils.py                   # 🔨 TODO: App utilities

Legenda:
✅ = File creato e completo
🔨 = TODO - Da implementare
📄 = File di configurazione/documentazione
📂 = Directory
```

---

## 📊 Status File per Membro

### 🔬 Membro A: Data Specialist

**File Pronti:**
- ✅ `src/data/dataset.py` - PhobiaDataset class implementata
- ✅ `cfg/data/coco_phobia.yaml` - Config dataset
- ✅ `cfg/data/augmentation.yaml` - Config augmentation

**Da Completare:**
- 🔨 `src/data/augmentation.py` - Implementare augmentation pipeline
- 🔨 `src/data/preprocessing.py` - Preprocessing functions
- 🔨 `src/data/download.py` - Script download dataset
- 🔨 `scripts/download_data.sh` - Bash script download

**Priority:**
1. Download dataset (spider, snake, blood)
2. Testare PhobiaDataset con dati reali
3. Implementare augmentation avanzata

---

### 🧠 Membro B: Model Architect

**File Pronti:**
- ✅ `src/models/phobia_net.py` - PhobiaNet implementato
- ✅ `src/models/loss.py` - Custom loss function
- ✅ `cfg/model/tiny_yolo.yaml` - Config Tiny-YOLO
- ✅ `cfg/model/baseline.yaml` - Config baseline
- ✅ `scripts/train.py` - Training script completo

**Da Completare:**
- 🔨 `src/training/trainer.py` - Alternative trainer (opzionale)
- 🔨 `src/training/validator.py` - Validation logic
- 🔨 `src/training/metrics.py` - mAP, IoU metrics

**Priority:**
1. Testare loss function con dati dummy
2. Debug training loop (primo epoch)
3. Tuning iperparametri

---

### 🎬 Membro C: Deployment Engineer

**File Pronti:**
- ✅ `src/inference/nms.py` - NMS implementato

**Da Completare:**
- 🔨 `src/inference/detector.py` - Inference engine
- 🔨 `src/inference/video_processor.py` - Video processing
- 🔨 `src/inference/blur.py` - ROI blurring
- 🔨 `app/streamlit_app.py` - Demo interface
- 🔨 `scripts/demo.py` - Demo script

**Priority:**
1. Testare NMS con detections dummy
2. Implementare video frame processing
3. Creare demo Streamlit

---

## 🔧 Configurazioni Disponibili

### Model Configs

1. **tiny_yolo.yaml**
   - Architettura: 6 layer CNN
   - Grid size: 13x13
   - Input: 416x416
   - Parametri: ~500K

2. **baseline.yaml**
   - Architettura: 3 layer CNN
   - Grid size: 7x7
   - Input: 224x224
   - Parametri: ~100K
   - Uso: Test rapidi

### Training Configs

1. **default.yaml**
   - Epochs: 100
   - Batch size: 16
   - Optimizer: Adam (lr=0.001)
   - Scheduler: StepLR
   - Mixed precision: enabled

2. **fast_test.yaml**
   - Epochs: 5
   - Batch size: 8
   - Subset: 10% data
   - Uso: Debug veloce

### Data Configs

1. **coco_phobia.yaml**
   - Classes: spider, snake, blood
   - Format: YOLO
   - Splits: 70/15/15 (train/val/test)

2. **augmentation.yaml**
   - HorizontalFlip: 50%
   - Brightness/Contrast: 50%
   - Gaussian Blur: 20%
   - Rotation: 30%

---

## 🚀 Comandi Quick Reference

### Setup

```bash
# Clone
git clone https://github.com/your-team/PhobiaShield.git
cd PhobiaShield

# Install
pip install -r requirements.txt
pip install -e .

# Login W&B
wandb login
```

### Training

```bash
# Fast test
python scripts/train.py training=fast_test

# Full training
python scripts/train.py model=tiny_yolo training=default

# Custom
python scripts/train.py training.epochs=50 training.lr=0.001
```

### Testing

```bash
# Test dataset
python src/data/dataset.py

# Test model
python src/models/phobia_net.py

# Test loss
python src/models/loss.py

# Test NMS
python src/inference/nms.py
```

### Git

```bash
# Create branch
git checkout -b feature/your-feature

# Commit
git add .
git commit -m "feat: description"
git push origin feature/your-feature

# Update
git pull origin main
git merge main
```

---

## 📈 Progress Tracking

### Week 1 (Days 1-4) - Setup & Architecture
- [x] Repository structure created
- [x] Configuration files setup
- [x] Core classes implemented (Dataset, Model, Loss, NMS)
- [ ] Dataset downloaded
- [ ] First training run

### Week 2 (Days 5-9) - Training & Integration
- [ ] Data pipeline tested
- [ ] Model training working
- [ ] Loss converging
- [ ] First checkpoints saved
- [ ] Demo prototype

### Week 3 (Days 10-14) - Finalization
- [ ] Model optimized
- [ ] Demo polished
- [ ] Report written
- [ ] Presentation ready
- [ ] Code review completed

---

## 🎯 Key Milestones

1. **Day 4**: First successful training epoch ✨
2. **Day 7**: Model loss starts decreasing 📉
3. **Day 9**: First video demo working 🎬
4. **Day 12**: Report draft completed 📝
5. **Day 14**: Final presentation 🎉

---

## 📚 Documentation Links

- **README.md**: Panoramica progetto
- **QUICKSTART.md**: Setup rapido
- **docs/GIT_WORKFLOW.md**: Guida Git dettagliata
- **notebooks/training_colab.ipynb**: Tutorial training

---

## 🔗 External Resources

- [PyTorch Docs](https://pytorch.org/docs/)
- [Hydra Docs](https://hydra.cc/)
- [W&B Docs](https://docs.wandb.ai/)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)

---

## 📞 Support

Per domande o problemi:
1. Controlla la documentazione
2. Cerca in GitHub Issues
3. Chiedi al team
4. Apri una nuova Issue

---

**Buon lavoro! 🚀**

*Ultimo aggiornamento: Dicembre 2025*
