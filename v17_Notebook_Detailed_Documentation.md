# Ultra-Detailed Documentation: Distribution Shift in Medical Imaging (v17)

## Executive Summary

**Notebook:** `M3_Presentation_Distribution_Shift_v17.ipynb`
**Project:** CAS AML Module 3, University of Bern
**Authors:** Sharad George, Moataz Mansour
**Purpose:** To rigorously investigate whether autoencoder reconstruction error can predict ML model performance degradation across different medical imaging populations, with a focus on understanding and quantifying distribution shift due to institutional, demographic, and pathological factors.

---

## 1. Research Question & Motivation

**Main Question:**
> Can autoencoder reconstruction error predict ML model performance degradation across different medical imaging populations?

**Why is this important?**
- Medical imaging models often fail when deployed to new hospitals or populations due to distribution shift.
- Detecting this shift before catastrophic failure is critical for safe clinical AI deployment.
- Autoencoders offer a label-free, unsupervised way to detect such shifts.

---

## 2. Datasets

### NIH ChestX-ray14
- **Source:** US adult population
- **Size:** 112,120 images (53.8% normal, 46.2% abnormal)
- **Role:** Main training dataset for both autoencoder and classifier

### Pediatric Pneumonia
- **Source:** Chinese pediatric population
- **Size:** 5,856 images (27.0% normal, 73.0% abnormal)
- **Role:** Out-of-distribution test set (demographic + institutional shift)

### CheXpert
- **Source:** US Stanford Hospital
- **Size:** 29,031 images (3.9% normal, 96.1% abnormal)
- **Role:** Out-of-distribution test set (institutional shift)

**All datasets are preprocessed to 224x224 grayscale, normalized to [0,1], and stored as HDF5 files.**

---

## 3. Data Preparation & Filtering

### HDF5 Structure
- Each dataset split (train/val/test) is stored as an H5 file with `images` and `labels` datasets.
- For each dataset, additional `*_normals.h5` files are created containing only normal images (for pathology-controlled experiments).

### Filtering Normals
- **NIH:** Normals are images where the first label column ('No Finding') is 1 and all other columns are 0.
- **CheXpert:** Normals are images where all 14 pathology columns are 0 (most conservative filter).
- **Pediatric:** Normals are images where the label is 0 (binary: 0=normal, 1=pneumonia).

**Scripts are provided to automate this filtering and save new H5 files.**

---

## 4. Phase 1: Baseline Statistical Analysis

### Phase 1a: All Data Comparison
- **Goal:** Quantify baseline distribution shift using all images (mixed pathologies).
- **Method:**
  - Load test images from all three datasets.
  - Compute pixel intensity statistics (mean, std, min, max).
  - Calculate Jensen-Shannon (JS) divergence between pixel intensity histograms.
  - Visualize sample images, histograms, boxplots, and JS divergence.
- **Key Results:**
  - NIH vs Pediatric: JS = 0.18
  - NIH vs CheXpert: JS = 0.28
  - Pediatric vs CheXpert: JS = 0.22
  - **Interpretation:** Institutional factors (NIH vs CheXpert) cause more shift than demographics (NIH vs Pediatric).

### Phase 1b: Normals-Only Comparison
- **Goal:** Isolate institutional/demographic shift by controlling for pathology.
- **Method:**
  - Load only normal images from each dataset.
  - Repeat statistics and JS divergence calculations.
  - Visualize and compare to Phase 1a.
- **Key Results:**
  - NIH vs Pediatric: JS = 0.11 (down from 0.18)
  - NIH vs CheXpert: JS = 0.21 (down from 0.28)
  - Pediatric vs CheXpert: JS = 0.16 (down from 0.22)
  - **Interpretation:** Pathology accounts for ~25-30% of shift; institutional/technical factors dominate.

---

## 5. Phase 2: Autoencoder Training

### Phase 2a: NIH_Full Autoencoder
- **Data:** All NIH images (normals + abnormals)
- **Architecture:**
  - Encoder: 4 Conv2D+ReLU+MaxPool layers → Flatten → Dense(256)
  - Decoder: Dense → Reshape → 4 Conv2DTranspose+ReLU → Conv2D(1, tanh)
  - **Latent space:** 256 dimensions
- **Training:**
  - Loss: Mean Squared Error (MSE)
  - Optimizer: Adam (lr=0.001)
  - Batch size: 256 (uses 160GB GPU)
  - Epochs: 100 (early stopping, reduce LR on plateau)
  - Mixed precision enabled for speed/memory
  - Data loaded fully into GPU memory for ultra-fast training
- **Outputs:**
  - Best model checkpoint, final model, encoder, decoder, training history, metadata
  - All files saved and verified for integrity
- **Visualizations:**
  - Architecture diagram
  - Training/validation loss curves
  - Sample reconstructions (original vs reconstructed)
  - Reconstruction error distribution (histogram, summary stats)

### Phase 2b: NIH_Normal Autoencoder
- **Data:** Only normal NIH images (from pre-filtered H5)
- **Same architecture and training procedure as Phase 2a**
- **Purpose:** To learn "normal" chest X-ray appearance and test if error is more sensitive to institutional shift
- **Outputs/Visualizations:** As in Phase 2a, but for normals-only data

---

## 6. Phase 3: Reconstruction Error Analysis

### Phase 3a: NIH_Full Autoencoder on All Test Images
- **Process:**
  - Use trained NIH_Full autoencoder to reconstruct test images from all datasets
  - Compute per-image MSE (original - reconstructed)
  - Aggregate mean, std, min, max, median for each dataset
  - Visualize error distributions (histograms, boxplots, barplots)
- **Key Results:**
  - NIH: Mean error = 0.00537 (baseline)
  - Pediatric: 0.01151 (+114% shift)
  - CheXpert: 0.01705 (+218% shift)
  - **All differences highly significant (p < 0.001)**

### Phase 3b: NIH_Normal Autoencoder on Normals-Only Test Images
- **Process:**
  - Use NIH_Normal autoencoder to reconstruct only normal images from each dataset
  - Compute and visualize error as above
- **Key Results:**
  - NIH: 0.00597 (baseline)
  - Pediatric: 0.01602 (+168% shift)
  - CheXpert: 0.01868 (+213% shift)
  - **All differences highly significant (p < 0.001)**

### Statistical Testing
- Pairwise t-tests and Mann-Whitney U tests confirm all differences are highly significant
- Cohen's d effect sizes are large for all comparisons
- Bonferroni correction applied for multiple comparisons

---

## 7. Phase 4: Classifier Training & Evaluation

### Model Architecture
- **Base:** DenseNet121 (ImageNet pre-trained, first 2 dense blocks frozen, last 2 trainable)
- **Input:** 224x224 grayscale images (converted to 3 channels)
- **Data Augmentation:** Random flip, rotation, zoom
- **Head:** Dense(256, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
- **Loss:** Binary cross-entropy
- **Optimizer:** Adam (lr=0.0001)
- **Batch size:** 32
- **Epochs:** 50 (early stopping, reduce LR on plateau)
- **Class Weights:** Computed to balance normal/abnormal classes

### Training Process
- Load NIH train/val/test splits
- Convert multi-label to binary: 0 = Normal, 1 = Abnormal
- Train with data augmentation and class weights
- Save best model by validation AUC
- Training/validation loss and metrics plotted

### Evaluation
- Evaluate on NIH, Pediatric, and CheXpert test sets
- Compute AUC, balanced accuracy, sensitivity, specificity, confusion matrices
- Visualize ROC curves, prevalence analysis, and summary dashboards
- **Key Finding:** AUC can be misleading in imbalanced datasets; balanced accuracy is the correct metric

---

## 8. Phase 5: Correlation Analysis

### Objective
- Test whether autoencoder reconstruction error (from Phase 3) predicts classifier performance degradation (from Phase 4)

### Process
- For each dataset, collect:
  - Mean reconstruction error (Phase 3a/3b)
  - Classifier AUC, balanced accuracy, sensitivity, specificity (Phase 4)
- Compute Pearson and Spearman correlations between error and performance metrics
- Visualize scatter plots and regression lines
- Save all results and dataframes

### Key Results
- **Reconstruction error vs Balanced Accuracy:** Strong negative correlation (higher error → lower performance)
- **AUC correlation is misleading** due to prevalence bias
- **Conclusion:** Autoencoder error is a valid, label-free detector of distribution shift and performance risk

---

## 9. Project Data Structure

```
chest_xray_distribution_shift/
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
- **H5 File Contents:**
  - `*_all.h5`: All images (normal + abnormal)
  - `*_normal.h5`: Normal images only (for pathology-controlled experiments)

---

## 10. Key Takeaways & Clinical Implications

- **Institutional/technical factors** are the dominant source of distribution shift (more than demographics or pathology)
- **Autoencoder reconstruction error** is a strong, label-free indicator of out-of-distribution data and correlates with classifier performance drop
- **Class imbalance** is severe in test sets (especially CheXpert); balanced accuracy and AUC are critical metrics
- **Multi-institutional validation** is essential for clinical ML deployment
- **Metric selection is critical:** AUC can hide catastrophic failures in imbalanced data; always use balanced accuracy and class-specific metrics

---

## 11. Step-by-Step: What Each Code Block Does

### Data Preparation
- Mounts Google Drive, copies scripts and data locally for speed
- Verifies all required files and directories exist
- Filters and saves normals-only H5 files for each dataset

### Phase 1
- Loads test images from H5 files
- Computes and prints pixel statistics
- Calculates JS divergence between datasets
- Visualizes sample images, histograms, boxplots, and JS divergence barplots
- Saves all results and figures to Drive

### Phase 2
- Loads training/validation images from H5 files
- Normalizes images to [0,1]
- Builds convolutional autoencoder (encoder, decoder, full model)
- Compiles with MSE loss and Adam optimizer
- Sets up callbacks: checkpoint, early stopping, reduce LR, save verification
- Trains model with full data in GPU memory (ultra-fast)
- Saves all models, training history, and metadata
- Visualizes architecture, training curves, sample reconstructions, error distributions

### Phase 3
- Loads trained autoencoders
- Reconstructs test images from all datasets
- Computes per-image and per-dataset reconstruction error
- Visualizes error distributions, boxplots, and summary tables
- Performs statistical significance testing (t-test, Mann-Whitney, Cohen's d)
- Saves all results and figures

### Phase 4
- Loads NIH train/val/test splits
- Builds DenseNet121-based classifier with custom head
- Applies data augmentation and class weights
- Trains with early stopping and checkpointing
- Evaluates on all test sets, computes all metrics
- Visualizes ROC curves, confusion matrices, prevalence analysis, and summary dashboards
- Saves all results and figures

### Phase 5
- Loads Phase 3 and Phase 4 results
- Computes correlations between reconstruction error and classifier performance
- Visualizes scatter plots and regression lines
- Saves all results and dataframes
- Prints key findings and summary tables

---

## 12. Advanced Details

- **Mixed Precision Training:** Reduces memory usage and speeds up training on supported GPUs
- **Ultra-Fast Training:** Loads entire dataset into GPU memory for maximum speed (requires 160GB GPU)
- **Save Verification:** Custom callback checks that all model files are saved correctly after each epoch
- **Statistical Testing:** All pairwise comparisons are Bonferroni-corrected for multiple testing
- **Reproducibility:** Random seeds set, package versions controlled, all outputs saved with metadata

---

## 13. References
- Wang et al. (2017): ChestX-ray14 dataset
- Irvin et al. (2019): CheXpert dataset
- Kermany et al. (2018): Pediatric pneumonia dataset
- Koh et al. (2021): WILDS benchmark for distribution shift
- Rabanser et al. (2019): Detecting distribution shifts

---

**Last updated:** November 24, 2025
