<div align="center">

# SGL-Net: Decoupling Multi-Granularity Perception and Harmonizing Inductive Biases for Robust Medical Image Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)

</div>

---

## 📋 Abstract

**SGL-Net** (Synergistic Global-Local Network) is a hybrid CNN-Transformer encoder-decoder architecture engineered for robust medical image segmentation. It is purpose-built to simultaneously resolve three fundamental pathologies that limit existing U-shaped segmentation networks: **(1) the Multi-Scale Aggregation Paradox**, wherein a single attention window scale fails to capture both fine-grained structures and broader contextual regions; **(2) the Heterogeneous Representation Gap**, which arises from the conflicting inductive biases of CNN (local, translation-equivariant) and Transformer (global, permutation-equivariant) representations when coupled naively; and **(3) Cross-Level Semantic Fusion Misalignment**, in which undiscriminating direct skip connections propagate low-level noise into the decoder, corrupting semantic feature fusion. SGL-Net addresses all three issues through a family of dedicated, theoretically-motivated modules described below.

---

## 🏗️ Architecture

![SGL-Net Architecture](docs/architecture.png)

*Figure: Overall encoder-decoder architecture of SGL-Net. The encoder and decoder stages are built from HSAB blocks featuring Parallel Granularity Attention (PGA). Downsampling uses Dual-Path Reduction (DPR) and upsampling uses Dual-Path Restoration (DPRestore). Skip connections are semantically filtered by the Semantic-Guided Aligner (SGA) before cross-level fusion.*

---

## ✨ Core Innovations

- **🔀 Decoupled Multi-Granularity Perception** — via the **Heterogeneous Synergistic Attention Block (HSAB)** and its intelligent head-partitioning strategy, **Parallel Granularity Attention (PGA)**. Rather than applying a single window size to all attention heads, PGA partitions the head budget across multiple independent window scales (e.g., 3×3, 7×7, 11×11) that operate concurrently and in parallel. This decoupled multi-granularity strategy allows the block to simultaneously perceive fine cell boundaries, intermediate tissue structures, and coarse organ-level context—without any sequential overhead or scale-selection bias.

- **🔧 Heterogeneous Structural Harmonization** — via the **Dual-Path Reduction (DPR)** and **Dual-Path Restoration (DPRestore)** modules. Each module fuses a Transformer patch-merging/sub-pixel-expansion path with a parallel strided/transposed-convolution CNN path. Critically, both paths inject locally-inductive geometric structure through the **Local Geometry Injector (LGI)**, a depthwise-convolutional + Squeeze-and-Excitation module appended after the FFN in every HSAB block. This harmonization bridges the representation gap between CNN and Transformer features throughout the hierarchy, preventing catastrophic representation conflicts at stage boundaries.

- **🎯 Intelligent Cross-Level Feature Routing** — via the **Semantic-Guided Aligner (SGA)**, a cross-attention gating mechanism that operates on every encoder-to-decoder skip connection. The decoder's high-level semantic features act as query vectors to selectively attend to and amplify semantically relevant tokens in the encoder skip features—effectively acting as a smart noise filter that suppresses irrelevant low-level texture information while preserving structurally discriminative features before fusion.

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
conda create -n sglnet python=3.8 -y
conda activate sglnet

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install dependencies
pip install timm einops thop
```

Or equivalently with a requirements file:

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

Organize datasets under `./data/` using the following structure:

```
data/
├── DSB2018/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── ISIC-Melanoma/
│   ├── train/
│   └── test/
├── ISIC-SK/
│   ├── train/
│   └── test/
├── SkinLesion/
│   ├── train/
│   └── test/
└── BrainMRI/
    ├── train/
    └── test/
```

### 3. Training

```bash
python train.py \
    --dataset ISIC-Melanoma \
    --data_path ./data/ISIC-Melanoma \
    --img_size 224 \
    --batch_size 16 \
    --num_epochs 300 \
    --lr 0.001 \
    --output_dir ./output/
```

### 4. Testing / Evaluation

```bash
python test.py \
    --dataset ISIC-Melanoma \
    --data_path ./data/ISIC-Melanoma \
    --img_size 224 \
    --checkpoint ./output/best_model.pth \
    --output_dir ./results/
```

---

## 📦 Pre-trained Models

| Dataset | IoU | Dice | Checkpoint |
|---|---|---|---|
| DSB2018 | — | — | Coming soon |
| ISIC-Melanoma | — | — | Coming soon |
| ISIC-SK | — | — | Coming soon |
| Skin Lesion | — | — | Coming soon |
| Brain MRI | — | — | Coming soon |

---

## 🔬 Ablation Study Models

The `Ablation Study/` directory contains the following variant models for controlled comparison:

| File | Description |
|---|---|
| `SGL_Net_NoSGA.py` | SGL-Net without Semantic-Guided Aligner (SGA) |
| `SGL_Net_NoDPR.py` | SGL-Net without Dual-Path Reduction (DPR) |
| `SGL_Net_NoDPRestore.py` | SGL-Net without Dual-Path Restoration (DPRestore) |
| `SGL_Net_NoHSAB.py` | SGL-Net without HSAB (uses standard Swin Transformer blocks) |
| `Swin_UNet_SGA.py` | Pure Swin-UNet baseline + SGA skip routing |
| `Swin_UNet_DPR.py` | Pure Swin-UNet baseline + DPR downsampling |
| `Swin_UNet_DPRestore.py` | Pure Swin-UNet baseline + DPRestore upsampling |
| `Swin_UNet_HSAB.py` | Pure Swin-UNet baseline + HSAB blocks |

---

## 📁 Repository Structure

```
SGL-Net/
├── SGL_Net.py              # Main model: SwinTransformerSys (SGL-Net full model)
├── Modules/
│   ├── HSAB.py             # Heterogeneous Synergistic Attention Block (PGA/S-PGA)
│   ├── SGA.py              # Semantic-Guided Aligner (skip connection routing)
│   ├── DPR.py              # Dual-Path Reduction (downsampling)
│   ├── DPRestore.py        # Dual-Path Restoration (upsampling)
│   └── LGI.py              # Local Geometry Injector (structural harmonization)
├── Ablation Study/         # Eight ablation variant models
├── docs/
│   └── architecture.png    # Architecture diagram (placeholder)
├── requirements.txt
└── README.md
```

---

## 📝 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{sglnet2025,
  title   = {SGL-Net: Decoupling Multi-Granularity Perception and Harmonizing Inductive
             Biases for Robust Medical Image Segmentation},
  author  = {[Author Names]},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2025},
  volume  = {},
  pages   = {},
  doi     = {}
}
```

---

## 🙏 Acknowledgments

We gratefully acknowledge the open-source community for foundational work and implementations, in particular the authors of [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet), and [TransUNet](https://github.com/Beckschen/TransUNet). This research was supported by [Funding Source Placeholder].

---

<div align="center">
<sub>Built with ❤️ for the medical imaging community</sub>
</div>
