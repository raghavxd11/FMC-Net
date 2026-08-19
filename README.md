# FMC-Net: Feature-level Multi-scale Cross-attention Network with GCN on Kvasir v1

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Colab-Ready-orange.svg?logo=googlecolab)](https://colab.research.google.com/)
[![Dataset](https://img.shields.io/badge/Dataset-Kvasir%20v1-blue.svg)](https://datasets.simula.no/kvasir/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A paper-inspired, end-to-end PyTorch prototype of **FMC-Net** tailored for automated gastrointestinal tract disease classification on the **Kvasir v1** dataset (8 classes, 4,000 endoscopic images).

---

## 📌 Overview & Architecture

FMC-Net combines hierarchical vision transformers with multi-scale feature pyramids, spatial/channel attention, weakly-supervised region selection, cross-attention, and graph convolutional networks (GCN) to capture fine-grained endoscopic lesions and broad anatomical landmarks.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 Input Image (224 x 224)                │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │    Swin Transformer Backbone  │
                             └───────────────┬───────────────┘
                                             │ Multi-scale stages
                                             ▼
                             ┌───────────────────────────────┐
                             │ Feature Pyramid Network (FPN) │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │   CBAM (Channel + Spatial)    │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │ Weakly Supervised Selector    │
                             │  (Global & Local Extraction)  │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │ Global-Local Cross Attention  │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │       GCN Feature Fusion      │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │    8-Class Classification     │
                             └───────────────────────────────┘
```

---

## 🗂️ Dataset: Kvasir v1

The dataset contains 4,000 labeled endoscopic images across 8 balanced anatomical and pathological classes (500 images per class):

1. **`dyed-lifted-polyps`** (Pathological finding / procedure)
2. **`dyed-resection-margins`** (Pathological finding / procedure)
3. **`esophagitis`** (Inflammatory disease)
4. **`normal-cecum`** (Anatomical landmark)
5. **`normal-pylorus`** (Anatomical landmark)
6. **`normal-z-line`** (Anatomical landmark)
7. **`polyps`** (Lesion / adenoma)
8. **`ulcerative-colitis`** (Inflammatory bowel disease)

---

## 📁 Repository Structure

```text
├── FMCNet_Kvasir_Prototype.ipynb         # Main Colab notebook (Trained on Kvasir v1 with full outputs)
└── README.md                             # Project documentation
```

---

## ⚡ Key Optimizations in `FMCNet_Kvasir_Prototype.ipynb`

- **Colab Local SSD Caching**: Automatically copies the dataset from Google Drive to Colab's high-speed local NVMe storage (`/content/kvasir_local`) once before training, eliminating Drive network I/O bottlenecks and boosting training speed by **5x–10x**.
- **Worker Crash Protection**: Safe DataLoader configuration (`num_workers=0`, `pin_memory=True`) preventing broken pipe / CUDA worker crashes when interrupting or resuming Colab execution.
- **Automated Ablation Study**: Runs true end-to-end evaluation across all ablation variants (Swin, +FPN, +CBAM, +WSS, +Cross-Attention, Full FMC-Net) and dynamically constructs real comparison tables.

---

## 🚀 Quickstart Guide (Google Colab)

1. Open **`FMCNet_Kvasir_Prototype.ipynb`** in Google Colab with GPU runtime enabled (`Runtime` → `Change runtime type` → `T4 GPU` or better).
2. Mount your Google Drive where the Kvasir dataset is stored:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Set your dataset path in `CFG.DATASET_ROOT`:
   ```python
   class CFG:
       DATASET_ROOT = "/content/drive/MyDrive/kvasir-dataset"
       IMAGE_SIZE = 224
       BATCH_SIZE = 16
       EPOCHS = 30
   ```
4. Run all cells to execute data loading, mixed-precision training, validation checkpointing, test evaluation, and visualizations.

---

## 📊 Evaluation & Visualizations

The notebook includes full post-training evaluation and interpretability tools:

- **Metrics**: Macro Precision, Macro Recall, Macro F1-Score, and Accuracy.
- **Classification Report**: Full per-class precision, recall, f1-score, and support metrics.
- **Confusion Matrix**: Annotated Seaborn heatmap showing true vs. predicted classifications.
- **t-SNE Embeddings**: 2D feature representation scatter plot to inspect class clustering.
- **WSS Attention Maps**: Weakly-supervised feature localization overlays showing focused regions on endoscopic frames.

---

## 🔬 Ablation Experiments

The notebook benchmarks modular component contributions:

| Model Variant | Key Components |
| :--- | :--- |
| **Swin Backbone** | Baseline Swin Transformer |
| **Swin + FPN** | Multi-scale feature pyramid integration |
| **Swin + FPN + CBAM** | Dual-attention (Channel + Spatial) refinement |
| **Swin + FPN + CBAM + WSS** | Weakly-supervised regional selection |
| **Full FMC-Net** | Swin + FPN + CBAM + WSS + Cross-Attention + GCN |

---

## 📜 Disclaimer
This implementation is a **paper-inspired research prototype** for educational and research comparison on the Kvasir v1 dataset.