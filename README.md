# Deepfake Detection with Vision Transformer & Advanced CNNs

## 📌 Overview
This project is a **deepfake detection system** built using **Vision Transformer (ViT)** and advanced **Convolutional Neural Networks (CNNs)**.  
The model takes in video frames or images, processes them into patches, and uses attention mechanisms to identify whether the content is manipulated or authentic.

## 🚀 Features
- **Vision Transformer Backbone** – Splits the image into 16×16 patches and processes them as sequences.
- **Advanced CNN Layers** – Pretrained layers for strong feature extraction.
- **Gradient Accumulation** – Train with large effective batch sizes without running out of GPU memory.
- **Mixed Precision Training** – Faster training with less GPU memory usage.
- **Learning Rate Scheduling** – OneCycleLR for optimal convergence.
- **Custom Dataset Support** – Easily plug in your own dataset.
- **Robust Logging** – Track training progress with Python logging.


## ⚙️ Key Parameters
- **Patch Size:** `16` → splits a 224×224 image into **196 small images** (patches).
- **Gradient Accumulation Steps:** `2` → bigger effective batch size without more GPU memory.
- **Attention Heads:** `16` → multiple perspectives for better pattern recognition.
- **Model Path:** `'deepfake_vit_model.pt'` → saved trained model weights.
- **Weight Decay:** `1e-4` → prevents overfitting by shrinking large weights.

## 🖼 How It Works
1. **Frame Extraction:** Convert videos into image frames.
2. **Preprocessing:** Resize, normalize, and split images into patches.
3. **Feature Extraction:** CNN layers capture low-level patterns.
4. **Transformer Encoding:** ViT processes patch sequences to learn relationships.
5. **Classification:** Output is either *Real* or *Deepfake*.

## 🛠 Installation
```bash
# Clone repo
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection

## 🔄 Pipeline Diagram

```mermaid
flowchart LR
    A[Video Input] --> B[Frame Extraction]
    B --> C[Preprocessing & Patch Splitting]
    C --> D[Advanced CNN Feature Extraction]
    D --> E[Vision Transformer Encoding]
    E --> F[Classification Layer]
    F --> G[Real or Deepfake Prediction]

# Install dependencies
pip install -r requirements.txt
