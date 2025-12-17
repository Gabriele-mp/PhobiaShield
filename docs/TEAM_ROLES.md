# Team Roles - PhobiaShield

**Project Duration:** 14 days (December 2025)

**Team Size:** 3 members

**Strategy:** "ALL-IN" - Shared tasks (dataset, report) + Individual specialization

---

## Team Structure

```
PhobiaShield Team
├── Gabriele (The Architect) - Model Architecture & Loss
├── Gianluca (Training Specialist) - Dataset & Report
└── Marco (Deployment & Demo Engineer) - Inference & Demo
```

---

## 👤 Gabriele - The Architect (Model Architect)

### Primary Responsibilities

**Architecture Design (Days 0-6)**
- Design neural network architecture
- Implement multi-scale FPN (Feature Pyramid Network)
- ResidualBlock with skip connections
- CBAM attention modules
- Detection heads for 3 scales (P3, P4, P5)

**Loss Function Implementation (Days 3-6)**
- Implement custom loss function from scratch
- Focal Loss for class imbalance (1:2,365 ratio)
- MSE Loss for localization (bounding boxes)
- CrossEntropy for classification
- Multi-scale loss aggregation
- Class weighting system

**Training Optimization (Days 7-10)**
- Hyperparameter tuning
- Learning rate scheduling (warmup + cosine annealing)
- Gradient clipping
- Early stopping implementation
- Mixed precision training (AMP)

**NMS Implementation (Days 11-13)**
- Non-Maximum Suppression algorithm
- Multi-class NMS
- IoU calculation
- Confidence thresholding
- Parameter tuning (conf=0.3, iou=0.3)

### Code Ownership

**Primary Files:**
- `src/models/phobia_net.py` (base model)
- `src/models/phobia_net_fpn.py` ⭐ (FPN implementation)
- `src/models/loss.py` (base loss)
- `src/models/loss_fpn.py` ⭐ (FPN + Focal Loss)
- `src/inference/nms.py` ⭐ (NMS algorithm)
- `scripts/train_clean.py` ⭐ (training script)

**Configuration:**
- `cfg/model/tiny_yolo_5class.yaml`
- `cfg/model/phobia_net_fpn.yaml`

### Key Deliverables

✅ **Architecture**
- PhobiaNetFPN class (400 lines)
- 5.4M parameters (21.6 MB)
- Multi-scale detection at 52×52, 26×26, 13×13

✅ **Loss Function**
- FPNLoss class with Focal Loss (γ=2.0, α=0.25)
- Handles 1:2,365 positive/negative imbalance
- Class weights: [4.76, 1.28, 3.70, 1.01, 1.39]

✅ **Training**
- Best model: epoch 22, val_loss 4.5031
- Training time: ~4 hours (Tesla T4)
- Early stopping prevented overfitting

✅ **NMS**
- Reduced predictions from 130k → 11k (92% reduction)
- Optimal parameters: conf=0.3, iou=0.3

### Expertise Gained

- Multi-scale object detection (FPN)
- Focal Loss implementation
- PyTorch optimization techniques
- Debugging neural networks
- Hyperparameter tuning

### Timeline

| Days | Task | Status |
|------|------|--------|
| 0-2 | Design FPN architecture | ✅ Complete |
| 3-6 | Implement loss function | ✅ Complete |
| 7-10 | Training & optimization | ✅ Complete |
| 11-13 | NMS implementation | ✅ Complete |
| 14 | Final integration | ✅ Complete |

---

## 👤 Gianluca - Training Specialist

### Primary Responsibilities

**Dataset Acquisition (Days 0-2)**
- Download datasets from multiple sources:
  - COCO subsets
  - Kaggle datasets
  - Roboflow (Clown dataset)
  - Open Images (Shark, Spider)
- Format annotations to YOLO format
- Total: 11,425 images across 5 classes

**Data Processing (Days 3-5)**
- Implement `PhobiaDataset` class
- Data augmentation pipeline (Albumentations)
- Train/Val/Test split (70/15/15, stratified)
- Class ID remapping
- Quality assurance (coordinate validation)

**Statistical Analysis (Days 6-8)**
- Object size distribution analysis
- Class balance verification
- Dataset bias identification
- 260× size variation documentation

**Report Writing (Days 6-14)**
- Introduction & Related Works
- Dataset section
- Experimental results
- Analysis of errors (confusion matrix)
- Comparative analysis (FPN vs YOLOv8)

### Code Ownership

**Primary Files:**
- `src/data/phobia_dataset.py` ⭐ (Dataset class)
- `scripts/merge_final_dataset.py` (dataset merging)
- `scripts/download_from_roboflow.py` (Roboflow downloader)
- `scripts/visualize_dataset.py` (visualization)

**Documentation:**
- `docs/DATASET_ULTIMATE_README.md` ⭐
- `results/README.md`
- `results/comparison.md`

### Key Deliverables

✅ **Dataset**
- DATASET_ULTIMATE_COMPLETE: 11,425 images
- 5 classes: Clown (3,052), Shark (2,683), Spider (2,206), Blood (1,856), Needle (1,062)
- Stratified split: 7,593 train / 1,624 val / 1,634 test
- YOLO format annotations

✅ **PhobiaDataset Class**
- Supports augmentation (flip, brightness, HSV)
- Handles edge cases (coordinates >1, empty files)
- Efficient loading with collate_fn

✅ **Documentation**
- Complete dataset README with statistics
- Size analysis (1.36px to 354px range)
- Class distribution tables
- Source attribution

✅ **Report**
- 2-page LaTeX report (template: MNIST-FDS)
- Results analysis
- Error analysis with confusion matrices

### Expertise Gained

- Large-scale dataset management
- Data augmentation techniques
- Statistical analysis
- Technical writing
- LaTeX formatting

### Timeline

| Days | Task | Status |
|------|------|--------|
| 0-2 | Dataset acquisition | ✅ Complete |
| 3-5 | Dataset class & augmentation | ✅ Complete |
| 6-8 | Statistical analysis | ✅ Complete |
| 9-14 | Report writing | ✅ Complete |

---

## 👤 Marco - Deployment & Demo Engineer

### Primary Responsibilities

**Inference Pipeline (Days 0-3)**
- Implement inference wrapper
- NMS post-processing integration
- Confidence filtering
- Multi-scale prediction aggregation

**Video Processing (Days 4-7)**
- Frame-by-frame video processing (OpenCV)
- Gaussian blur on detected regions (ROI)
- Real-time performance optimization
- Video saving with processed frames

**YOLOv8 Baseline (Days 5-8)**
- Train YOLOv8s for comparison
- Transfer learning from COCO
- Hyperparameter tuning
- Results documentation

**Demo Interface (Days 8-14)**
- Streamlit/Gradio interface
- Upload video functionality
- Model selection (FPN vs YOLOv8)
- Confidence threshold slider
- Download processed video

**Presentation Demo (Day 14)**
- Prepare video demo (trailer)
- Side-by-side comparison (original vs blurred)
- Live webcam demo (bonus)

### Code Ownership

**Primary Files:**
- `src/inference/predictor.py` (inference wrapper)
- `src/demo/video_processor.py` (video processing)
- `demo_app.py` (Streamlit interface)
- `scripts/train_yolov8.py` ⭐ (YOLOv8 training)

**Notebooks:**
- `notebooks/02_YOLOv8_Training.ipynb` ⭐
- `notebooks/04_Inference_Demo.ipynb`

### Key Deliverables

✅ **Inference Pipeline**
- Predictor class with NMS integration
- Real-time processing (~40ms FPN, ~10ms YOLO)
- Batch prediction support

✅ **Video Processor**
- OpenCV-based frame processing
- Gaussian blur on ROI (configurable kernel)
- 20 FPS processing capability

✅ **YOLOv8 Baseline**
- mAP50: 70.0% (+152% vs FPN)
- Training time: ~1.5-2 hours
- Production-ready performance

✅ **Demo Interface**
- Streamlit web app
- Upload/process/download workflow
- Model comparison feature
- Interactive parameter tuning

✅ **Presentation Demo**
- Trailer video (Harry Potter / Indiana Jones)
- Side-by-side comparison
- Impressive visual impact

### Expertise Gained

- Real-time video processing
- Model deployment
- User interface design
- Transfer learning (YOLOv8)
- Performance optimization

### Timeline

| Days | Task | Status |
|------|------|--------|
| 0-3 | Inference pipeline | ✅ Complete |
| 4-7 | Video processing | ✅ Complete |
| 5-8 | YOLOv8 baseline | ✅ Complete |
| 8-12 | Demo interface | ✅ Complete |
| 13-14 | Presentation demo | ✅ Complete |

---

## Shared Responsibilities

### Dataset Creation (ALL members)

**Collaborative tasks:**
- Downloading images from sources
- Manual annotation (if needed)
- Quality checking
- Format standardization

**Contribution:**
- Gabriele: Shark and Spider datasets
- Member A: Clown and Blood datasets
- Member C: Needle dataset + background images

### Report Writing (ALL members)

**Section ownership:**
- Introduction: Member A
- Architecture: Gabriele
- Dataset: Member A
- Training: Gabriele
- Results: ALL (collaborative)
- Demo: Member C
- Conclusion: Member A

**Review process:**
- First draft: Individual sections
- Review: Cross-review by others
- Final edit: Member A (coordinator)

### Presentation (ALL members)

**Slide ownership:**
- Slide 1-2 (Intro, Dataset): Member A
- Slide 3-4 (Architecture, NMS): Gabriele
- Slide 5-6 (Results, YOLOv8): Gabriele + Member C
- Slide 7-8 (Comparison, Conclusion): Member A
- Demo: Member C

**Rehearsal:**
- Full team rehearsal 2 days before
- Timing practice (5 min + 1 min demo)
- Q&A preparation

---

## Communication Channels

### Primary

- **GitHub Issues**: Task tracking and bugs
- **GitHub Projects**: Kanban board for tasks
- **Pull Requests**: Code review

### Secondary

- **WhatsApp/Telegram**: Quick questions
- **Google Meet**: Weekly sync (30 min)
- **Google Drive**: Shared documents and datasets

---

## Decision Making

### Technical Decisions

**Architecture choices:**
- Gabriele proposes → Team reviews → Consensus

**Dataset choices:**
- Member A proposes → Team reviews → Consensus

**Demo choices:**
- Member C proposes → Team reviews → Consensus

### Conflict Resolution

1. Discuss on GitHub Issue
2. If no consensus → Majority vote
3. If tied → Gabriele has tie-breaking vote (project lead)

---

## Meeting Schedule

### Weekly Sync (30 min)

**Agenda:**
1. Progress updates (5 min each)
2. Blockers discussion (10 min)
3. Next week planning (5 min)

**Time:** Tuesdays, 18:00 CET

### Daily Standup (Optional)

**Format:** Async on WhatsApp
- What I did yesterday
- What I'll do today
- Any blockers

---

## Code Review Process

### Pull Request Guidelines

**Required reviews:**
- Architecture changes: Gabriele must review
- Dataset changes: Member A must review
- Demo changes: Member C must review

**Review checklist:**
- [ ] Code follows style guide
- [ ] Tests pass (if applicable)
- [ ] Documentation updated
- [ ] No conflicts with main

### Approval

- 1 approval required for merge
- Self-merge allowed for hotfixes (inform team)

---

## Recognition & Credits

### Individual Contributions

**In report and presentation:**
- Explicitly mention who implemented what
- Example: "The FPN architecture was designed and implemented by Gabriele"

### Shared Credit

**Team effort sections:**
- Dataset creation
- Report writing
- Final results analysis

### External Credits

**Sources to acknowledge:**
- COCO dataset
- Roboflow datasets
- Open Images
- YOLO papers (Redmon et al.)
- Focal Loss paper (Lin et al.)
- FPN paper (Lin et al.)

---

## Contact Information

### GitHub

- **Gabriele**: [@Gabriele-mp](https://github.com/Gabriele-mp)
- **Member A**: [Insert GitHub handle]
- **Member C**: [Insert GitHub handle]

### Repository

- **Main**: https://github.com/Gabriele-mp/PhobiaShield
- **Issues**: https://github.com/Gabriele-mp/PhobiaShield/issues
- **Projects**: https://github.com/Gabriele-mp/PhobiaShield/projects

---

## Success Metrics

### Individual

**Gabriele:**
- ✅ FPN implemented and working
- ✅ Loss function converges
- ✅ Training completes successfully
- ✅ NMS reduces predictions effectively

**Member A:**
- ✅ Dataset ready (11k+ images)
- ✅ PhobiaDataset class working
- ✅ Report submitted (2 pages)
- ✅ Statistical analysis complete

**Member C:**
- ✅ YOLOv8 baseline trained
- ✅ Demo interface working
- ✅ Video processing pipeline ready
- ✅ Presentation demo impressive

### Team

- ✅ Final presentation delivered (5 min + demo)
- ✅ Report submitted on time
- ✅ GitHub repo organized and documented
- ✅ Models achieve >80% recall
- ✅ Demo works reliably

---

## Lessons Learned (Post-Project)

### What Worked Well

1. **Role specialization**: Clear ownership prevented conflicts
2. **ALL-IN strategy**: Shared tasks built team cohesion
3. **GitHub workflow**: Clean history, easy collaboration
4. **Early milestones**: Prevented last-minute rush

### What Could Improve

1. **Earlier YOLOv8 baseline**: Would have guided FPN design
2. **More frequent syncs**: Weekly was sometimes too sparse
3. **Earlier NMS implementation**: Debugging took longer than expected
4. **Better time estimation**: Some tasks took 2× longer than planned

### Key Takeaways

1. **Transfer learning is powerful**: YOLOv8 crushed FPN (+152% mAP)
2. **Multi-scale is essential**: Single-scale completely failed (16.7% recall)
3. **Focal Loss works**: Handled 1:2,365 imbalance perfectly
4. **Team coordination matters**: Clear roles = less friction

---

## Acknowledgments

**Course:** Fundamentals of Data Science

**University:** Sapienza University of Rome

**Instructor:** Prof. Indro Spinelli

**Year:** 2025

---

**This document reflects the actual team structure and contributions for the PhobiaShield project.**
