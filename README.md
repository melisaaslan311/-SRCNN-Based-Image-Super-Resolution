# Deep Learning Based Super-Resolution: SRCNN Model Comparison

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)


**Author:** Melisa Aslan & Ömer Aysal

A comprehensive comparison study of Super-Resolution Convolutional Neural Network (SRCNN) models using different input types (RGB vs Y-channel) and deep learning frameworks (TensorFlow vs PyTorch) on custom dataset.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Model Comparison](#-model-comparison)
- [Methodology](#-methodology)
- [Dataset](#-dataset)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Discussion](#-discussion)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Project Overview

This project aims to evaluate the performance of three different SRCNN (Super-Resolution Convolutional Neural Network) models to understand how input type and deep learning framework affect super-resolution performance.

### Key Objectives
- 🔬 Compare SRCNN performance across different input channels (RGB vs Y-channel)
- ⚖️ Evaluate TensorFlow vs PyTorch implementation differences
- 📊 Analyze reconstruction quality using PSNR metrics
- 🧪 Provide insights for optimal super-resolution model selection

## 🏗️ Model Comparison

| Model | Framework | Input Type | Architecture | PSNR (dB) |
|-------|-----------|------------|--------------|-----------|
| **Model-1** | TensorFlow | RGB Channel | 3-layer SRCNN | **30.37** |
| **Model-1** | TensorFlow | Y Channel | 3-layer SRCNN | 27.99 |
| **Model-2** | PyTorch | Y Channel | 3-layer SRCNN | 24.48 |

### 🏆 Best Performance
**TensorFlow-based Model-1 with RGB input** achieved the highest PSNR value (30.37 dB), demonstrating superior reconstruction quality.

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

## 💻 Usage

### Training Models
```bash
# Train TensorFlow RGB model
python train_tensorflow_rgb.py

# Train TensorFlow Y-channel model
python train_tensorflow_y.py

# Train PyTorch Y-channel model
python train_pytorch_y.py
```

### Evaluation
```bash
# Evaluate all models
python evaluate_models.py

# Generate comparison plots
python plot_results.py
```

### Single Image Super-Resolution
```python
from models.srcnn_tensorflow import SRCNN_TF
from utils.image_utils import load_image, save_image

# Load model
model = SRCNN_TF()
model.load_weights('checkpoints/srcnn_rgb_best.h5')

# Process image
lr_image = load_image('input/low_res.jpg')
sr_image = model.predict(lr_image)
save_image(sr_image, 'output/super_res.jpg')
```

## 📂 Project Structure

```
srcnn-comparison/
├── 📁 data/
│   ├── 📁 train/
│   │   ├── 📁 HR/          # High-resolution images
│   │   └── 📁 LR/          # Low-resolution images
│   └── 📁 test/
├── 📁 models/
│   ├── 📄 srcnn_tensorflow.py
│   ├── 📄 srcnn_pytorch.py
│   └── 📄 __init__.py
├── 📁 utils/
│   ├── 📄 image_utils.py
│   ├── 📄 metrics.py
│   └── 📄 data_loader.py
├── 📁 checkpoints/         # Trained model weights
├── 📁 results/            # Output images and plots
├── 📁 notebooks/          # Jupyter analysis notebooks
├── 📄 train_tensorflow_rgb.py
├── 📄 train_tensorflow_y.py
├── 📄 train_pytorch_y.py
├── 📄 evaluate_models.py
├── 📄 requirements.txt
└── 📄 README.md
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
