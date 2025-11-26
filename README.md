# Distribution Shift in Medical Imaging  
### Using Autoencoder Reconstruction Error to Predict Model Performance on New Data

> CAS AML Project — University of Bern  
> Authors: Sharad George & Moataz Mansour

---

## 🔍 Project Overview

Deep learning models in medical imaging are typically trained and validated on a **single curated dataset**.  
Once deployed in a **different hospital**, with different scanners, demographics, and pathology prevalence, their performance can **silently degrade** — this is the classic **distribution shift** problem.

In this project we:

- Train a **convolutional autoencoder (CAE)** on NIH ChestX-ray14 images.
- Use its **reconstruction error** and **latent space** as **unsupervised signals** of domain shift.
- Train a **DenseNet-121 classifier** on NIH and evaluate it on:
  - NIH (in-distribution),
  - a **Chinese pediatric** chest X-ray dataset,
  - **CheXpert** (Stanford).
- Study how **reconstruction error correlates** with **classifier performance** (especially balanced accuracy) across these datasets.

The goal:  
👉 Show that **autoencoder reconstruction error** can act as an **early-warning signal** for performance degradation on out-of-distribution medical images.

---

## 🎯 Motivation

- Training and deployment data are rarely i.i.d. in real hospitals.
- Differences in:
  - scanners and acquisition protocols,
  - image post-processing,
  - demographics (adults vs children),
  - label prevalence,
  all contribute to distribution shift.
- Labels in new domains are **expensive and delayed** (radiologists).
- We need **unsupervised tools** that can say:
  > “This new data no longer looks like the training data – be careful with this model.”

This project investigates whether a CAE’s **reconstruction error** can provide such a warning.

---

## 📊 Datasets

We consider three chest X-ray datasets:

1. **NIH ChestX-ray14**
   https://www.kaggle.com/datasets/nih-chest-xrays/data 
   - Large adult dataset from a US institution.  
   - Used as the **training domain** for both autoencoder and classifier.  
   - Binary target: **No Finding vs Abnormal** (e.g. Pneumonia vs Normal).

3. **Pediatric CXR Dataset** (China)  
   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 
   - Pediatric patients (children) from a different hospital.  
   - Binary labels: **Pneumonia vs Normal**.  
   - Introduces **demographic** + **institutional** shift.

5. **CheXpert** (Stanford)
   https://www.kaggle.com/datasets/ashery/chexpert 
   - Large adult dataset from another US institution.  
   - Different acquisition & labeling pipelines, **heavily imbalanced** towards abnormal cases.  
   - Introduces **institutional** + **prevalence** shift.

> **Important:**  
> All models in this project are **trained only on NIH**.  
> Pediatric and CheXpert are **never seen during training** and are used strictly for evaluation.

---

## 🧱 Method Overview

The code, experiments, and figures are primarily organized in:

- `notebooks/M3_Presentation_Distribution_Shift_v18_with_tSNE.ipynb`  
  (main analysis + plots used in the presentation)

The pipeline is structured in **five phases**:

### Phase 1 – Baseline Distribution Shift (Pixels)

- Compare **pixel intensity histograms** across NIH, Pediatric, CheXpert.
- Compute **Jensen–Shannon divergence** between distributions.
- Repeat analysis on **normals-only subsets**.
- **Finding:** clear shift even for normals → **institutional factors matter**, not just pathology.

---

### Phase 2 – Convolutional Autoencoder (NIH Only)

- CAE Architecture:
  - Input: 1×224×224 chest X-ray.
  - Encoder: Conv + pooling → **256-dim latent vector**.
  - Decoder: Transposed conv → reconstruct 1×224×224.
- Loss: **Mean Squared Error (MSE)**.
- Two variants:
  1. **NIH_Full AE** – trained on all NIH images.
  2. **NIH_Normal AE** – trained on NIH normals only.

**Key observations:**

- CAE converges with low reconstruction error on **NIH validation**.
- NIH_Normal AE shows only a **small increase** in error (~few %) for **NIH abnormals**.  
  → This is our **baseline “pathology effect”** inside one domain.

---

### Phase 3 – Cross-Dataset Reconstruction Error

- Freeze AE weights and feed images from:
  - NIH,
  - Pediatric,
  - CheXpert.
- Compute reconstruction error for each image; compare **error distributions**.

**Results (qualitative pattern):**

- **NIH**: lowest reconstruction error (in-distribution).
- **Pediatric**: error roughly **~2× NIH**.
- **CheXpert**: error often **>2× NIH**.
- Same pattern holds when evaluating **normals only** with NIH_Normal AE.
- These cross-dataset shifts are **much larger** than the internal NIH normal vs abnormal gap.

**Interpretation:**

- From the autoencoder’s perspective, Pediatric and CheXpert are **far more unusual** than diseased NIH lungs.
- Shift is dominated by **institutional/technical + demographic** factors.

---

### Phase 3c – Latent Space & t-SNE

- Extract **256-dim latent vectors** from NIH_Normal AE.
- Apply **t-SNE** to embed into 2D.
- Color points by dataset (NIH, Pediatric, CheXpert).

**Observation:**

- Datasets form **distinct clusters** in latent space.
- NIH cluster = training domain region.  
- Pediatric & CheXpert clusters = **out-of-domain regions**.

This gives geometric intuition:  
points far from the NIH cluster (in latent space) correspond to higher **reconstruction error** and, as we will see, worse **classifier performance**.

---

### Phase 4 – DenseNet-121 Classifier (NIH Only)

- Architecture:
  - **DenseNet-121** pre-trained on ImageNet.
  - Adapted for 1-channel 224×224 inputs.
  - Early blocks **frozen**, later blocks + new head **fine-tuned**.
- Task: binary **Abnormal vs No Finding** on NIH.
- Training:
  - Data augmentation (flips, small rotations, zoom).
  - Class weights / balanced sampling.
  - Validation on held-out NIH subset.

**Evaluation metrics:**

- **AUC** (Area under ROC).
- **Accuracy** (but used with care).
- **Balanced Accuracy** = (Sensitivity + Specificity) / 2  
  → gives equal weight to minority/majority classes.

**Cross-dataset performance:**

- NIH → **best** balanced accuracy (in-distribution).
- Pediatric → drop in balanced accuracy.
- CheXpert → AUC and accuracy can look okay, but **balanced accuracy** and **specificity on normals** are clearly worse.

---

### Phase 5 – Correlation Analysis

Finally, we relate **reconstruction error** to **classifier performance**.

- At **dataset level**:
  - NIH: lowest error, highest balanced accuracy.
  - Pediatric: medium error, medium balanced accuracy.
  - CheXpert: highest error, lowest balanced accuracy.
- In more fine-grained analyses:
  - Error **quantiles**: high-error bins tend to have worse balanced accuracy.

**Takeaway:**  
There is a **negative correlation** between reconstruction error and classifier performance:  
> Higher AE error → Lower balanced accuracy  
This supports using the autoencoder as an **unsupervised domain-shift monitor**.

---
