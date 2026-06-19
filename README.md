# Biomedical-Signal-Processing-and-Control
# Two-Stream CNN-LSTM for Surgical Phase Recognition

This repository contains a Python/TensorFlow implementation based on the methodology described in the paper: **"Adaptive undersampling and short clip-based two-stream CNN-LSTM model for surgical phase recognition on cholecystectomy videos" (2024)**.

---

## 📌 Project Overview
Automated surgical phase recognition is highly challenging due to severe class imbalances among different phases of surgery, which often leads to biased learning or overfitting. 

This project tackles the problem using a two-fold approach:
1. **Adaptive Temporal Subsampling (Undersampling):** Dynamically calculates specific strides and frame rates (FPS) for each individual surgical phase to balance the training data distribution.
2. **Two-Stream CNN-LSTM Architecture:** Uses a pre-trained **GoogLeNet** backbone to extract spatial visual features, which are then processed in parallel by two distinct LSTM models:
   - **Sequence-to-Sequence LSTM:** Captures short-range temporal context and behavioral changes within frame neighbors.
   - **Sequence-to-Vector LSTM:** Captures long-term video context and structural flow across the entire clip.

---

## 🛠️ Features & Architecture
- **Data Balancing:** Re-samples imbalanced data into balanced, short 10-frame clips.
- **Parallel Learning:** Merges the prediction scores of both LSTM streams using an additive fusion method to minimize lost temporal information.
- **End-to-End Style:** Formulated using `TimeDistributed` layers in TensorFlow for efficient clip processing.

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python 3.x and the required dependencies installed:
```bash
pip install tensorflow numpy
