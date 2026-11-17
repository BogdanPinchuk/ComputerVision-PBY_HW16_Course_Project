# Intelligent Image Denoising: Classical Spatial Filters vs. Convolutional Autoencoder vs. Novel Fourier KAN Autoencoder

<p align="center">
  <b><a href="README.md">🇬🇧 English</a></b> • <b><a href="README_UA.md">🇺🇦 Українська</a></b>
</p>

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hardware Acceleration](https://img.shields.io/badge/Acceleration-Apple%20Silicon%20MPS-black?logo=apple&logoColor=white)](https://developer.apple.com/metal/pytorch/)
[![Training](https://img.shields.io/badge/Training-BSD432-informational)](https://github.com/liutianlin0121/BSD-432-68)
[![Benchmark](https://img.shields.io/badge/Benchmark-BSD68-success)](https://github.com/liutianlin0121/BSD-432-68)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Computer Vision Course Project (Topic 2: *«Intelligent Image Denoising based on Autoencoders»*).  
Author: **Bohdan Pinchuk**

---

## 📌 Project Overview & Motivation

Digital image restoration (*Image Denoising*) is a foundational problem in low-level computer vision. Physical sensors inevitably introduce corruptions during acquisition under low-light conditions, high ISO settings, sensor thermal noise, and transmission channel degradation.

The primary objective of this project is to conduct a systematic, end-to-end comparative study across three conceptually distinct denoising paradigms:
1. **Classical Spatial Filtering (Zero-Learning Baselines):** Gaussian filter (mandatory baseline), Median rank filter, and Bilateral edge-preserving filter.
2. **Deep Learning Convolutional Baseline (CAE):** A 5-stage symmetric Convolutional Autoencoder with reflection padding, skip connections, and a scale-balanced composite loss ($0.85 L_1 + 0.15 (1 - \text{SSIM})$).
3. **Novel Spectral Model (Fourier KAN Autoencoder):** A Kolmogorov-Arnold Network autoencoder employing learnable complex-exponential trigonometric Fourier harmonics along connection edges, stabilized via Sobolev spectral decay ($\sigma_k \propto 1/k^2$), adaptive phase offsets $\phi_k$, frequency scales $\omega_k$, and a residual SiLU highway.

All models undergo rigorous evaluation on the benchmark **BSD68** dataset, assessing restoration fidelity ($\text{PSNR}$, $\text{SSIM}$, $\text{MAE}$, $\text{MSE}$, $R^2$), hardware inference latency, and memory footprint, while robustness across 6 noise modalities is evaluated in multi-modal experiments.

---

## 🔬 Methodological Pipeline (8 Stages)

```mermaid
flowchart LR
    S1["1. EDA & Data Audit<br/>(BSD432, BSD68, Stats)"] --> S2["2. Noise Engine<br/>(6 Modalities, 2D FFT)"]
    S2 --> S3["3. Classical Baselines<br/>(Gaussian, Median, Bilateral)"]
    S3 --> S4["4. Deep CAE Baseline<br/>(Conv Autoencoder, Optuna)"]
    S4 --> S5["5. Novel Fourier KAN<br/>(Spectral KAN, Sobolev Reg)"]
    S5 --> S6["6. Master Benchmark<br/>(PSNR, SSIM, Latency)"]
    S6 --> S7["7. Error Diagnostics<br/>(Inferno Residuals, RPSD)"]
    S7 --> S8["8. Synthesis & Decisions<br/>(Deployment Matrix)"]
```

1. **EDA & Data Ingestion:** Systematic analysis of BSD432 (training) and BSD68 (evaluation benchmark) datasets, dynamic range $[0, 255]$ verification, luminance profiling, and streaming train/val patch generation ($96 \times 96$).
2. **Multi-Modal Synthetic Noise Engine:** Procedural generation of 6 synthetic corruptions:
   - Additive White Gaussian Noise (AWGN, $\sigma \in [10, 50]$).
   - Impulse Salt & Pepper Noise ($p \in [0.02, 0.15]$).
   - Multiplicative Speckle Noise ($\text{Var} \in [0.04, 0.20]$).
   - Signal-dependent Poisson Noise ($\lambda \in [10, 50]$).
   - Periodic 2D Sinusoidal Striped Noise (simulating sensor grid interference).
   - Hybrid Composite Corruption (AWGN + Impulse + Periodic).
3. **Classical Spatial Baselines:** Parameter grid search and evaluation for Gaussian blur, Median rank filter, and Bilateral edge-preserving filter.
4. **Convolutional Autoencoder (CAE Baseline):** 5-stage CNN with Strided/Transposed Convolutions, Reflection Padding, and Optuna hyperparameter optimization trained with AdamW.
5. **Novel Spectral Model (Fourier KAN Autoencoder):** Complex-exponential 2D KAN convolutions, Sobolev weight initialization, and Bayesian hyperparameter optimization via Optuna.
6. **Master Multi-Model Benchmark:** Direct comparative evaluation across all 5 models under identical hardware conditions (Apple Silicon MPS).
7. **Spatial & Spectral Error Diagnostics:** Local residual error heatmaps (`inferno` colormap) and 2D FFT Radial Power Spectral Density (RPSD) curves to quantify high-frequency detail retention vs. noise floor suppression.
8. **Engineering Synthesis & Deployment Matrix:** Multi-criteria trade-off analysis balancing visual fidelity, hardware latency, and storage footprint.

---

## 📊 Master Benchmark Results (BSD68 Dataset)

Summary of restoration performance evaluated across 68 full-resolution images from the BSD68 benchmark test set:

| Model Architecture | Model Family | PSNR (dB) ↑ | SSIM ↑ | MSE ↓ | MAE ↓ | R² ↑ | Inference Latency ↓ | Parameters (M) ↓ | Storage Footprint ↓ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gaussian Filter** *(Baseline)* | Classical Spatial | 26.23 dB | 0.6815 | 365.62 | 11.52 | 0.8233 | 1.1 ms (CPU) | **0.00 M** | **0.00 MB** |
| **Median Filter** | Classical Spatial | 25.51 dB | 0.7173 | 260.78 | 10.73 | 0.8842 | **0.9 ms** (CPU) | **0.00 M** | **0.00 MB** |
| **Bilateral Filter** | Classical Spatial | 25.49 dB | 0.6950 | 396.95 | 11.57 | 0.8099 | 2.2 ms (CPU) | **0.00 M** | **0.00 MB** |
| **Convolutional Autoencoder (CAE)** | Deep CNN Baseline | 28.09 dB | 0.8602 | 155.90 | 8.85 | 0.9274 | 13.0 ms (MPS) | 1.05 M | 4.01 MB |
| **Fourier KAN Autoencoder** | Novel Spectral Model | **29.91 dB** | **0.8749** | **138.44** | **7.83** | **0.9341** | 107.5 ms (MPS) | 7.78 M | 29.70 MB |

> [!NOTE]
> *Metrics represent evaluation across all 68 full-resolution benchmark images of BSD68 under 5 synthetic noise types (AWGN, Poisson, Speckle, Salt & Pepper, Periodic). Classical spatial filters execute on multi-threaded CPU; deep neural networks run on Apple Silicon MPS.*

---

## 💡 Key Findings & Engineering Insights

1. **Architectural Triad & Denoising Quality Hierarchy:**
   - **Fourier KAN** achieves unequivocal leadership in structural reconstruction fidelity ($\text{PSNR} = \mathbf{29.91\text{ dB}}$, an improvement of **$+1.82\text{ dB}$** over CAE and **$+3.68\text{ dB}$** over Gaussian; $\text{SSIM} = \mathbf{0.8749}$, an improvement of **$+0.0147$** over CAE and **$+0.1934$** over Gaussian). Reconstruction error $\text{MSE} = 138.44$ (**$-11.2\%$** lower than CAE's $155.90$), $\text{MAE} = 7.83$ (**$-11.5\%$** lower than CAE's $8.85$), $R^2 = 0.9341$.
   - **Convolutional Autoencoder (CAE)** establishes a high-quality deep learning baseline ($\text{PSNR} = 28.09\text{ dB}$, $\text{SSIM} = 0.8602$), significantly outperforming spatial filters, though exhibiting slight oversmoothing across high-frequency textures.
   - **Classical spatial filters** define the baseline performance level: Gaussian ($26.23\text{ dB}$, $0.6815$), Median ($25.51\text{ dB}$, $0.7173$, uncontested leader for impulse noise), and Bilateral ($25.49\text{ dB}$, $0.6950$).

2. **Spatial & Spectral Error Diagnostics (Residual Maps & 2D FFT RPSD):**
   - *Residual Error Maps ($|I_{\text{clean}} - I_{\text{denoised}}|$):* Classical filters irreversibly truncate structural scene contours (scene silhouettes emerge clearly on error maps), CAE lowers errors with moderate edge blurring, whereas Fourier KAN produces residuals resembling uncorrelated white noise with zero structural scene geometry.
   - *Radial Energy Decay (2D FFT RPSD):* Fourier KAN uniquely and faithfully preserves natural scene power decay $P(f) \propto 1/f^2$, completely attenuating the high-frequency noise floor without washing out fine textures.

3. **Engineering Trade-Off: Quality vs Latency vs Memory Footprint:**
   - *Spatial Filters:* Zero storage ($0.00\text{ M}$ parameters, $0.00\text{ MB}$) and instantaneous CPU inference ($0.9$–$2.2\text{ ms}$, Median: $0.9\text{ ms}$, Gaussian: $1.1\text{ ms}$, Bilateral: $2.2\text{ ms}$), but limited frequency selectivity.
   - *Convolutional Autoencoder (CAE):* $1.05\text{ M}$ parameters ($4.01\text{ MB}$), latency $13.0\text{ ms}$ on Apple Silicon MPS ($\approx 77\text{ FPS}$) — optimal selection for high-frame-rate real-time video streaming (60+ FPS).
   - *Fourier KAN Autoencoder:* $7.78\text{ M}$ parameters ($29.70\text{ MB}$), latency $107.5\text{ ms}$ ($\approx 9.3\text{ FPS}$) — converts additional computational resources into maximum reconstruction fidelity, reducing Optuna validation loss by $-23.67\%$ ($0.2205$ vs $0.2888$ for CAE).

4. **Mathematical & Physical Advantages of Fourier KAN Basis:**
   - Learnable 1D trigonometric kernels along connection edges operate as continuous harmonic interpolators, overcoming spectral bias (*Spectral Bias*) and escaping the rigid discrete receptive field constraints of standard CNN kernels.
   - Sobolev amplitude decay ($\sigma_k \propto 1/k^2$) concentrates initial energy on low-frequency structural shapes, preventing high-frequency noise memorization (*noise memorization*) and suppressing Gibbs ringing artifacts near sharp boundaries.

5. **Engineering Deployment Guidelines:**
   - *Real-Time Video (60+ FPS):* Quantized CAE (INT8) or hardware-accelerated bilateral filtering with sub-millisecond frame latency.
   - *Embedded / Edge AI (mobile ISPs, drones):* Compact CAE or KAN with a scaled-down harmonic basis ($K=2..3$, $C=24$).
   - *High-Fidelity Scientific & Medical Imaging (X-Ray, MRI, microscopy):* Full-scale Fourier KAN ($K=4$, $C=64$) trained with composite loss $0.85 L_1 + 0.15 (1 - \text{SSIM})$ to ensure phase integrity and micro-pathology preservation.

6. **Limitations & Future Directions:**
   - Hardware acceleration for trigonometric kernel evaluation via TensorRT / CoreML (FP16/INT8).
   - Dynamic adaptive selection of harmonic order $K$ across network depth and empirical validation on real-world raw sensor data (SIDD benchmark).

---

## 🗂️ Project Repository Structure

```
ComputerVision-PBY_HW16_Course_Project/
├── Bohdan_Pinchuk_CV_HW16.ipynb      # Master interactive research notebook (English)
├── Bohdan_Pinchuk_CV_HW16_UA.ipynb   # Master interactive research notebook (Ukrainian)
├── README.md                         # Project documentation and benchmark manifest (English)
├── README_UA.md                      # Project documentation and benchmark manifest (Ukrainian)
├── .gitignore                        # Git exclusion rules for large datasets and weights
├── apps/                             # Modular engineering architecture
│   ├── main.py                       # Device management, streaming datasets, augmentations, losses & metrics
│   ├── models.py                     # Implementations: CAE, Fourier KAN, composite SSIM+L1 loss
│   ├── spatial_filters.py            # Classical spatial baselines: Gaussian, Median, Bilateral
│   └── reporter.py                   # Structured tabular Pandas reporting utilities
└── resources/
    ├── data/                         # BSD (Berkeley Segmentation Dataset) image corpora
    │   ├── BSD432/                   # 432 monochrome training images
    │   ├── CBSD432/                  # 432 color RGB training images
    │   ├── BSD68/                    # 68 monochrome evaluation benchmark images
    │   └── CBSD68/                   # 68 color RGB evaluation benchmark images
    ├── models/                       # Checkpoints, parameter manifests, and Optuna studies
    │   ├── model_gaussian.json       # Optimal Gaussian filter kernel & sigma
    │   ├── model_median.json         # Optimal Median filter kernel size
    │   ├── model_bilateral.json      # Optimal Bilateral filter spatial & range sigmas
    │   ├── model_cae.pth             # Trained PyTorch weights for Convolutional Autoencoder (1.05M)
    │   ├── model_cae_params.json     # Best Optuna hyperparameters for CAE baseline
    │   ├── model_fourier_kan.pth     # Trained PyTorch weights for Fourier KAN Autoencoder (7.78M)
    │   └── model_fourier_kan_params.json # Best Optuna hyperparameters for Fourier KAN model
    └── reports/                      # Visual benchmarks and performance plots across 5 noise modalities
```

---

## 📚 Scientific Foundations & Literature References

This project builds upon peer-reviewed literature and open-source research implementations:

### Academic Papers:
* **Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M.** (2024). *KAN: Kolmogorov-Arnold Networks*. MIT, Harvard, & Caltech. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756).
* **Bodner, A., Tepsich, A., Brandstetter, J., & Vergari, A.** (2024). *Convolutional Kolmogorov-Arnold Networks*. [arXiv:2406.13155](https://arxiv.org/abs/2406.13155).
* **Mehrabian, A., et al.** (2024). *FourierKAN: Fourier Kolmogorov-Arnold Networks*. University of British Columbia. [arXiv:2409.09323](https://arxiv.org/abs/2409.09323).
* **Xu, J., Chen, Z., et al.** (2024). *Fourier KAN-Based Framework for Image Restoration*. University of Hong Kong. [arXiv:2406.01034](https://arxiv.org/abs/2406.01034).
* **Drokin, I.** (2024). *Torch-Conv-KAN: PyTorch Implementation of Convolutional KAN*. [arXiv:2407.01092](https://arxiv.org/abs/2407.01092).
* **Tancik, M., Srinivasan, P., Mildenhall, B., et al.** (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*. NeurIPS 2020. [arXiv:2006.10739](https://arxiv.org/abs/2006.10739).
* **Mao, X.-J., Shen, C., & Yang, Y.-B.** (2016). *Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections*. Advances in Neural Information Processing Systems (NeurIPS 2016). [arXiv:1603.09056](https://arxiv.org/abs/1603.09056).
* **Tomasi, C., & Manduchi, R.** (1998). *Bilateral Filtering for Gray and Color Images*. In Proceedings of the Sixth International Conference on Computer Vision (IEEE ICCV 1998), pp. 839–846.

### Open-Source Reference Repositories:
* [`GistNoesis/FourierKAN`](https://github.com/GistNoesis/FourierKAN) — Original PyTorch implementation of 1D Fourier KAN layers.
* [`AntonioTepsich/Convolutional-KANs`](https://github.com/AntonioTepsich/Convolutional-KANs) — 2D Convolutional KAN foundational implementations.
* [`IvanDrokin/torch-conv-kan`](https://github.com/IvanDrokin/torch-conv-kan) — Efficient tensorized 2D Convolutional KAN operators.
* [`Ali-Meh619/FKAN`](https://github.com/Ali-Meh619/FKAN) — Fourier Kolmogorov-Arnold Networks for PyTorch.

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
Clone the repository and install dependencies within a virtual environment:
```bash
git clone https://github.com/your-username/ComputerVision-PBY_HW16_Course_Project.git
cd ComputerVision-PBY_HW16_Course_Project

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # torch torchvision torchaudio optuna opencv-python matplotlib scipy pandas
```

### 2. Running the Research Notebooks
Launch Jupyter Lab or PyCharm (both English and Ukrainian notebooks are 100% synchronized):
```bash
# English version
jupyter lab Bohdan_Pinchuk_CV_HW16.ipynb

# Ukrainian version
jupyter lab Bohdan_Pinchuk_CV_HW16_UA.ipynb
```

Each stage is organized into `# Input data`, `# Solution`, and `# Report` sections with automatic cache detection (`load_model = True`). Pre-computed weights in `resources/models/` will load instantaneously; setting `load_model = False` triggers full retraining and Optuna hyperparameter tuning.
