# Comprehensive Documentation: Distribution Shift in Medical Imaging (v17)

## Executive Summary

**Notebook:** `M3_Presentation_Distribution_Shift_v17.ipynb`
**Project:** CAS AML Module 3, University of Bern
**Authors:** Sharad George, Moataz Mansour
**Purpose:** Investigate whether autoencoder reconstruction error can predict ML model performance degradation across different medical imaging populations, focusing on distribution shift due to institutional, demographic, and pathological factors.

---

## Research Question

Can autoencoder reconstruction error predict ML model performance degradation across different medical imaging populations?

---

## Datasets

- **NIH ChestX-ray14**: US adult population, 112,120 images (53.8% normal, 46.2% abnormal)
- **Pediatric Pneumonia**: Chinese pediatric population, 5,856 images (27.0% normal, 73.0% abnormal)
- **CheXpert**: US Stanford Hospital, 29,031 images (3.9% normal, 96.1% abnormal)

**Key Characteristics:**
- NIH: Balanced, general screening
- Pediatric: Disease-enriched, symptomatic children
- CheXpert: Highly disease-enriched, ICU/critical care

---

## Experimental Design

- **Phase 1a:** All data (mixed institutions/demographics)
- **Phase 1b:** NIH_Normal control (same pathology distribution)
- **Phases 2-6:** Autoencoder-based analysis pipeline

**Controls for institutional vs demographic factors in distribution shift.**

---

## Data Preparation & Storage

- **Raw Datasets:** Local hard drive (`../data/`)
- **Project Files:** Google Drive (`/content/drive/MyDrive/chest_xray_distribution_shift/`)
- **Processed Data:** HDF5 files in `data/processed/` (train/val/test splits)
- **Scripts:** In `scripts/` folder (Google Drive)
- **Results:** In `results/` folder

**Processing Pipeline:**
- Raw images → Python scripts → HDF5 files → Analysis
- All images resized to 224×224, grayscale, normalized

---

## Setup & Environment

- Google Colab notebook with Drive integration
- Automated setup: mounts Drive, copies scripts/labels/data locally for fast access
- GPU memory optimization enabled (TensorFlow memory growth)
- Python path updated to use local scripts if available
- Data/labels verification and summary printed

---

## Data Filtering & Preprocessing

- **Normals Filtering:**
  - For each dataset, HDF5 files are filtered to create `*_normals.h5` (only normal images)
  - NIH: Uses H5 label columns (column 0 = 'No Finding')
  - CheXpert/Pediatric: Similar logic, using available labels
- **Project Data Structure:**
  - `data/labels/`: CSV label files
  - `data/processed/nih/`, `chexpert/`, `pediatric/`: H5 files for each split
  - `results/phase2a/`, `phase2b/`, ...: Results for each phase

---

## Phase Overview

### Phase 1: Statistical Analysis
- Baseline distribution differences using summary statistics and JS divergence
- Compare all data and normals-only subsets

### Phase 2: Autoencoder Training
- Train convolutional autoencoders on NIH data (all and normals-only)
- Architecture: Encoder (Conv2D+ReLU+Pooling), Latent (Dense), Decoder (Conv2DTranspose+ReLU, final sigmoid)
- Loss: MSE, Optimizer: Adam

### Phase 3: Reconstruction Error Analysis
- Use trained autoencoders to reconstruct images from all datasets
- Compute per-image and per-dataset reconstruction error (MSE)
- Compare error distributions to detect distribution shift

### Phase 4: Classifier Training & Evaluation
- Train binary classifier (DenseNet121 + custom head) on NIH, test on all datasets
- Metrics: AUC, balanced accuracy, sensitivity, specificity
- Analyze performance degradation on out-of-distribution data

### Phase 5: Correlation Analysis
- Correlate reconstruction error with classifier performance (AUC, accuracy)
- Quantify predictive power of autoencoder error for model degradation

---

## Technical Details

- **Image Normalization:** All images scaled to [0,1] (divide by 255)
- **Batch Sizes:** Large for autoencoder (e.g., 512), smaller for classifier
- **Mixed Precision:** Enabled for speed/memory efficiency
- **Model Saving:** All models and results saved to Drive for persistence
- **Reproducibility:** Random seeds set, package versions controlled

---

## Project Data Structure Example

```
chest_xray_distribution_shift/
│
├── data/
│   ├── labels/
│   └── processed/
│       ├── chexpert/
│       ├── nih/
│       └── pediatric/
├── scripts/
├── results/
│   ├── phase2a/
│   ├── phase2b/
│   ├── phase3a/
│   ├── phase3b/
│   ├── phase4/
│   └── phase5/
```

---

## Key Findings & Insights

- **Institutional/technical factors** are the dominant source of distribution shift (more than demographics or pathology)
- **Autoencoder reconstruction error** is a strong indicator of out-of-distribution data and correlates with classifier performance drop
- **Class imbalance** is severe in test sets (especially CheXpert); balanced accuracy and AUC are critical metrics
- **Multi-institutional validation** is essential for clinical ML deployment

---

## How to Use This Notebook

1. Run setup cells to mount Drive and copy data/scripts locally
2. Prepare filtered H5 files for normals (if not already present)
3. Train autoencoders (Phase 2a/2b) or load pre-trained models
4. Run reconstruction error analysis (Phase 3)
5. Train and evaluate classifier (Phase 4)
6. Run correlation analysis (Phase 5)

**Tips:**
- Use local data for speed, Drive for persistence
- Monitor GPU memory and batch sizes
- Check all paths and file existence before running heavy jobs

---

## References
- Wang et al. (2017): ChestX-ray14 dataset
- Irvin et al. (2019): CheXpert dataset
- Kermany et al. (2018): Pediatric pneumonia dataset
- Koh et al. (2021): WILDS benchmark for distribution shift
- Rabanser et al. (2019): Detecting distribution shifts

---

**Last updated:** November 24, 2025
