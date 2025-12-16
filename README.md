# Amazon ML Challenge 2025 – Smart Product Pricing

## 📌 Overview

The **Amazon ML Challenge 2025** focuses on predicting optimal product prices in an e-commerce setting using **multimodal data**—textual product information and product images. Pricing in online marketplaces is influenced by complex interactions between brand, specifications, quantity, and visual cues.
This repository presents our end-to-end machine learning pipeline designed to capture these interactions and predict competitive product prices effectively.

---

## 🧠 Problem Statement

Given product metadata and images, the task is to **predict the product price** as accurately as possible.

### Dataset Components

* **sample_id** – Unique identifier for each product
* **catalog_content** – Concatenated text containing:

  * Product title
  * Product description
  * Item Pack Quantity (IPQ)
* **image_link** – Public URL to download the product image

---

## 🏗️ Our Approach

### 1. Text Processing

* Cleaned and normalized `catalog_content`
* Tokenization and embedding using transformer-based text encoders
* Captured semantic information such as brand, quantity, and specifications

### 2. Image Processing

* Downloaded product images from URLs
* Used pretrained CNN-based vision encoders to extract visual embeddings
* Handled missing or corrupted images robustly

### 3. Multimodal Fusion

* Combined **text embeddings + image embeddings**
* Feature concatenation followed by dense regression layers
* Learned cross-modal interactions impacting pricing

### 4. Model Training

* Regression-based learning objective
* Optimized directly for **SMAPE (Symmetric Mean Absolute Percentage Error)**
* Regularization and careful validation to prevent overfitting

---

## 📊 Evaluation Metric

**SMAPE (Symmetric Mean Absolute Percentage Error)**
Chosen due to its robustness for price prediction tasks with varying scales.

---

## 🏆 Results

* **Final SMAPE:** **50.49**
* **Leaderboard Position:** ~**1200**
* **Competition Level:** National (Amazon ML Challenge 2025)

This result validates the effectiveness of our multimodal learning strategy under strict evaluation constraints.

---

## 📁 Repository Structure

```
├── aml-best.ipynb        # Final best-performing pipeline
├── aml-hirerac.ipynb     # Experimental & alternative approaches
├── README.md             # Project documentation
```

---

## 🚀 Key Learnings

* Multimodal learning significantly improves price prediction accuracy
* Image features provide strong complementary signals to text
* Robust preprocessing is critical for real-world e-commerce data

---

## 👨‍💻 Authors

**Kartik Garg**
Amazon ML Challenge 2025 Participant

---
