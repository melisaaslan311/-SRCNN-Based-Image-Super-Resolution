# Deep Learning Based Super-Resolution: SRCNN Model Comparison

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)


**Author:** Melisa Aslan & Ömer Aysal

A comprehensive comparison study of Super-Resolution Convolutional Neural Network (SRCNN) models using different input types (RGB vs Y-channel) and deep learning frameworks (TensorFlow vs PyTorch) on custom dataset.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Methodology](#-methodology)
- [Dataset](#-dataset)
- [Results](#-results)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Discussion](#-discussion)

## 🎯 Project Overview

This project aims to evaluate the performance of three different SRCNN (Super-Resolution Convolutional Neural Network) models to understand how input type and deep learning framework affect super-resolution performance.

### Key Objectives
- 🔬 Compare SRCNN performance across different input channels (RGB vs Y-channel)
- ⚖️ Evaluate TensorFlow vs PyTorch implementation differences
- 📊 Analyze reconstruction quality using PSNR metrics
- 🧪 Provide insights for optimal super-resolution model selection

## 🔬 Methodology

### Model Architecture
The SRCNN model consists of three convolutional layers:
- **Layer 1**: 9×9 kernel - Feature extraction
- **Layer 2**: 1×1 kernel - Non-linear mapping  
- **Layer 3**: 5×5 kernel - Reconstruction

### Training Configuration
```python
# Training Parameters
epochs = 10
batch_size = 16
optimizer = Adam
loss_function = MSE (Mean Squared Error)
training_images = 684
```

### Normalization Strategies
- **RGB Models**: Channel-wise normalization using mean and standard deviation
- **Y-channel Models**: Pixel value scaling to [0, 1] range

## 📊 Dataset

The dataset consists of:
- **High-Resolution (HR) Images**: Ground truth images
- **Low-Resolution (LR) Images**: Generated using bicubic interpolation downsampling
- **Training Set**: 684 image pairs
- **Data Augmentation**: Bicubic downsampling for LR generation

### Data Preprocessing Pipeline
1. Load high-resolution images
2. Generate low-resolution counterparts via bicubic downsampling
3. Apply appropriate normalization based on input type
4. Create training batches

## 📈 Results

### Performance Metrics

| Metric | Model-1 (RGB) | Model-1 (Y) | Model-2 (Y) |
|--------|---------------|-------------|-------------|
| **Avg PSNR (dB)** | **30.37** | 27.99 | 24.48 |
| **Framework** | TensorFlow | TensorFlow | PyTorch |
| **Input Type** | RGB | Y-channel | Y-channel |

<img src="https://github.com/user-attachments/assets/3b98e3a0-94f3-4159-9ece-013d6737ee35" width="330" />
<img src="https://github.com/user-attachments/assets/c820254f-ce5a-4b27-a5f0-3ccaab77c6c6" width="330" />
<img src="https://github.com/user-attachments/assets/09054eb7-bab6-4fb6-a37b-47cf2ea9c0bc" width="330" />

### Key Findings

#### 🎨 Input Type Impact
- **RGB input** significantly outperformed Y-channel input (30.37 dB vs 27.99 dB)
- RGB channels provide both structural and chromatic information for better reconstruction

#### 🔧 Framework Comparison
- **TensorFlow** model achieved higher PSNR than PyTorch (27.99 dB vs 24.48 dB) for Y-channel input
- Differences may stem from initialization strategies or optimization approaches

### Visualization
```
PSNR Comparison
│
30.37 ████████████████████████████████ Model-1 (RGB)
27.99 ███████████████████████████      Model-1 (Y)
24.48 ████████████████████████         Model-2 (Y)
│
└── Higher is Better →
```

## 🚀 Installation

### Prerequisites
```bash
Python 3.7+
CUDA-compatible GPU (recommended)
```

### Dependencies
```bash
# Clone repository
git clone https://github.com/melisaaslan311/srcnn-comparison.git
cd srcnn-comparison

# Install requirements
pip install -r requirements.txt
```

### Required Packages
```txt
tensorflow>=2.8.0
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.5.0
scikit-image>=0.19.0
pillow>=8.3.0
```

## 📂 Project Structure

```
SRCNN-master/
├── model_original.py
├── run.ipynb       #.py dosyaları yerine .ipynb dosyaları çalıştırılacak 
├── run.py
├── .ipynb_checkpoints/
├── data.ipynb
├── data_original.py
├── main.py
├── main.py.ipynb
├── .DS_Store

Super-Resolution-master/
├── SRCNN.ipynb
├── SRCNN.py
├── SRCNN_Train_OS.ipynb
├── SRCNN_Train_OS.py
├── trained_SRCNN_model.keras
├── trained_SRCNN_model_y.weights.h5
├── y_channel.ipynb
├── Predictions/
├── Predictions_Y/
├── __pycache__/
├── .ipynb_checkpoints/
├── .DS_Store
├── .gitignore

Dataset/
├── Test
├── Train       #dosyalara ulaşmak için iletişime geçebilirsiniz 
├── Val

```

## 🤔 Discussion

### Key Insights
- **Input Channel Selection**: RGB input provides richer information leading to better reconstruction quality
- **Framework Differences**: Implementation variations between TensorFlow and PyTorch affect final performance
- **Training Efficiency**: All models converged within 10 epochs, suggesting efficient architecture design

### Limitations
- ⏱️ **Limited Training**: Only 10 epochs used for training
- 📏 **Single Metric**: Evaluation based solely on PSNR
- 📦 **Small Dataset**: 684 images may limit generalization capability
- 🔍 **Architecture Scope**: Only basic SRCNN tested
