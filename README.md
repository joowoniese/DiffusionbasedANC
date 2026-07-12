# Removing Occlusion Effect in ANC based on Generative Model (DiffWave) 🎧

[![Paper](https://img.shields.io/badge/Journal-Measurement%20(2026)-E91E63.svg)](https://doi.org/10.1016/j.measurement.2025.119599)
[![Python](https://img.shields.io/badge/Python-3.8-blue.svg?style=flat-flat&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C.svg?style=flat-flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official implementation of the paper: **"Mitigating ANC pressure effect in Active Noise Cancellation using diffusion-based generative models"** (*Measurement*, 2026). 

This project introduces a post-processing generative module based on **Conditional DiffWave** combined with a **novel frequency-weighted loss function** to suppress the perceptual "ANC pressure effect" (auditory occlusion/muffled sensation) without paradoxically reintroducing external low-frequency noise.

---

## 📌 Problem Statement: The ANC Pressure Effect
While Active Noise Cancellation (ANC) effectively suppresses ambient noise, temporal misalignments and low-frequency resonance within the sealed ear canal often induce a distressing **perceptual pressure effect** (experienced as auditory fatigue, muffledness, or headaches). 

Traditional solutions reinject low-frequency noise to buffer this sensation—a paradoxical approach that undermines the core goal of ANC. This framework addresses the phenomenon **purely at the audio level downstream**, training a diffusion model to transform realistic ANC-processed signals into idealized, occlusion-free anti-noise.

---

## 🛠️ System Overview

### 1. Data Preprocessing & Conditioning Pipeline
The framework overlays real-world environmental noise (from AI Hub) with source music. Input audio consists of ANC-generated anti-noise suffering from pressure artifacts, while the target condition maps to an idealized anti-noise created via exact phase inversion.

<p align="center">
  <img src="https://github.com/joowoniese/DiffusionbasedANC/blob/master/AboutModel/overview.png" width="60%" alt="Data Preprocessing Overview" />
</p>

### 2. Conditional DiffWave Architecture
The core model utilizes a non-autoregressive, parallel waveform generation setup. The target audio configuration is transformed into localized feature maps injected as a bias step into the multi-layer bidirectional dilated convolutional architecture ($Bi-DilConv-2^i \bmod n$).

<p align="center">
  <img src="https://github.com/joowoniese/DiffusionbasedANC/blob/master/AboutModel/ModelOverview.png" width="60%" alt="Model Architecture" />
</p>
*(Note: Please ensure the correct image path is mapped for your architecture detailed view)*

---

## 🔬 Novel Multi-Task Loss Formulation
To focus the network precisely on the spectral regions responsible for the occlusion effect, we implement a custom frequency-weighted loss that prioritizes low and mid bands:

$$\mathcal{L}_{weighted} = \alpha \cdot \frac{1}{N}\sum_{i \in \text{Low}}\sum_{j=1}^{T}(P_{ij} - \hat{P}_{ij})^2 + (1-\alpha) \cdot \frac{1}{N}\sum_{i \in \text{Mid}}\sum_{j=1}^{T}(P_{ij} - \hat{P}_{ij})^2$$

*   **Low Frequency Range ($\alpha = 0.9$):** $20\text{ Hz} - 250\text{ Hz}$ (accounts for $91.52\%$ of the spectral variance).
*   **Mid Frequency Range ($1-\alpha = 0.1$):** $250\text{ Hz} - 2000\text{ Hz}$ (accounts for $7.90\%$ of the variance).
*   **Total Loss Optimization:** Dynamically balanced via uncertainty-based task weights ($\alpha=0.495$, $\beta=0.505$ at convergence):
    $$\mathcal{L}_{\text{total}} = (1-\beta)\mathcal{L}_{\text{weighted}} + \beta\mathcal{L}_{\text{DiffWave}}$$

---

## 📊 Experimental Results

The proposed framework demonstrates substantial improvements over conventional Feedforward ANC (FFANC) and standard Deep Learning (DL-based) ANC systems:

| Metrics | FFANC Baseline [39] | DL-based ANC [43] | **Proposed Model** | Ideal Target |
| :--- | :---: | :---: | :---: | :---: |
| **PESQwb** (Perceptual Quality) | 1.912 | 1.770 | **2.114** | 4.5 (Max) |
| **SI-SDR** (Reconstruction Accuracy) | -21.409 dB | -11.010 dB | **+5.195 dB** | $\infty$ (Max) |
| **NRP** (Low-Freq Suppression) | - | -6.661 dB | **-10.122 dB** | $\infty$ (Max) |

*   **Inference Latency:** Achieves an average processing time of **$0.0148\text{ s}$ per 10-second slice**, falling well within the standard real-world human latency perception threshold ($0.1\text{s} - 0.15\text{s}$) for modern Bluetooth hardware.
*   **Subjective Auditory Validation:** In double-blind A/B listening tests, **$90\%$ of participants ($18/20$)** perceived clear mitigation or complete elimination of the pressure effect.

---

## 🚀 Getting Started

### Environment & Dependencies
The experiment environment was configured under Ubuntu 20.04.6 LTS utilizing PyTorch 2.4.1 and CUDA 12.2:
```bash
# Core requirements installation
pip install torch==2.4.1 torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install numpy pandas librosa matplotlib pydub
