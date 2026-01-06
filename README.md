# Amazon ML Challenge 2025 – Multimodal Product Price Prediction

## 📌 Overview

The **Amazon ML Challenge 2025** focuses on predicting optimal product prices in an e-commerce setting using **multimodal data**—structured text metadata and product images. Product pricing depends on subtle interactions between brand, quantity, specifications, packaging, and visual cues, making it a challenging real-world regression problem.

This repository presents a **fully end-to-end, research-driven multimodal pipeline**, progressing from strong pretrained embeddings to advanced cross-modal fusion architectures, strict cross-validation, and ensemble stacking—while rigorously avoiding data leakage.

---

## 🧠 Problem Statement

Given:

* Product metadata (title, description, item pack quantity, etc.)
* Product images

Predict the **final product price** as accurately as possible.

---

## 📂 Dataset Description

Each sample contains:

* **sample_id** – Unique product identifier
* **catalog_content** – Concatenated text containing:

  * Product title
  * Product description
  * Item Pack Quantity (IPQ)
* **image_link** – Public URL for the product image
* **price** – Target variable (train only)

---

## 🏗️ Methodology & System Design

### 1️⃣ Text Representation

* Cleaned and normalized `catalog_content`
* Embedded using **Qwen3 transformer-based text encoder**
* Captures:

  * Brand semantics
  * Quantity & packaging cues
  * Functional and descriptive language

**Output:** Dense text embeddings (~2048 dimensions)

---

### 2️⃣ Image Representation

Two independent, high-capacity vision encoders were used:

* **SigLIP (Vision Encoder Only)**

  * Strong global semantic alignment
  * Robust to diverse product imagery

* **DINOv3**

  * Self-supervised visual representation
  * Strong structural and texture awareness

Embeddings were:

* Extracted in batches with mixed precision
* L2-normalized
* Robust to corrupted or missing images

**Final image embedding:**
Average of SigLIP + DINOv3 representations (~2304 dimensions)

---

### 3️⃣ Multimodal Fusion Strategies (Progressive)

The project evolved through **multiple increasingly advanced fusion designs**:

#### 🔹 Baseline Fusion

* Concatenation of text + image embeddings
* Multi-layer MLP regressor
* Log-price prediction (`log1p(price)`)

#### 🔹 MSGCA (Multi-Stage Gated Cross Attention)

* Separate encoders for text and image streams
* Gated cross-attention layers
* Residual fusion to control modality dominance
* Quantile-aware loss for price stability

#### 🔹 MSGCA (Proper CV Fine-Tuning)

* 5-Fold K-Fold Cross-Validation
* No validation leakage
* Mixup augmentation
* Quantile + MSE hybrid loss
* Early stopping and LR scheduling

#### 🔹 Ultimate MSGCA-TFT Hybrid (Experimental)

* MSGCA backbone
* Temporal Fusion Transformer (TFT)-inspired blocks
* Gated residual connections
* SWA (Stochastic Weight Averaging)
* Multi-variant training for ensemble diversity

> **Key Insight:** Despite architectural sophistication, performance saturated due to dataset information limits rather than modeling capacity.

---

### 4️⃣ Training Strategy

* Target transformed using `log1p(price)`
* Optimizers: **AdamW**
* Gradient clipping for stability
* Early stopping per fold
* Mixed precision training on GPU
* Strict separation of train / validation / test data

---

### 5️⃣ Evaluation Metric

**SMAPE (Symmetric Mean Absolute Percentage Error)**
Chosen due to:

* Scale invariance
* Robustness to wide price ranges
* Official competition metric

---

## 📊 Results & Performance

| Model Stage                   | SMAPE       |
| ----------------------------- | ----------- |
| Baseline Multimodal MLP       | ~58%        |
| Hierarchical Fusion           | ~57%        |
| MSGCA (Fine-Tuned, Proper CV) | ~53.3%      |
| Ultimate MSGCA-TFT            | **~52.1%**  |
| Ensemble + Stacking           | **~51–52%** |

**Final Leaderboard Performance**

* **Best SMAPE:** ~50–52% (varies by submission)
* **Leaderboard Rank:** ~1200
* **Competition Scale:** National-level (Amazon ML Challenge 2025)

> Extensive experimentation demonstrated a **hard performance ceiling** using embeddings alone, highlighting realistic modeling limits.

---

## 🧪 Key Technical Contributions

* Vision-only SigLIP extraction (no text leakage)
* Robust DINOv3 fallback handling
* Proper K-Fold CV with aligned OOF & validation scores
* Quantile-aware loss design for price stability
* Controlled gating to prevent modality dominance
* Demonstrated limits of over-engineering without new signal

---

## 🚀 Key Learnings

* Multimodal learning significantly outperforms unimodal baselines
* Vision embeddings provide strong complementary pricing signals
* Cross-modal gating is critical for stable fusion
* More complex architectures **cannot overcome missing data signal**
* Proper validation is more important than architectural complexity

---

## 👨‍💻 Author

**Kartik Garg**
Amazon ML Challenge 2025 – Participant
