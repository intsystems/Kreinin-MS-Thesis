# Medical Flow Matching CT Translation

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/kreininmv/MS-Thesis)

**Master's Thesis** | MIPT, Department of Intelligent Systems  
**Author:** Matvei Kreinin  
**Supervisor:** Andrey Grabovoy, PhD

---

## 📋 Abstract

Medical imaging still lacks reliable methods to translate contrast-enhanced arterial-phase CT scans into their non-contrast (native) form. We introduce **Medical Flow Matching (MFM)**, which combines an efficient image-translation paradigm with a bottleneck attention mechanism designed for medical images.

### Key Results

| Metric | Value |
|--------|-------|
| **MAE** | 5.436 HU |
| **SSIM** | 0.996 |
| **PSNR** | 39.776 |
| **Speed-up** | 400× faster than DDPM |

Training nnUNetv2 on MFM-generated images achieves a **Dice score of 0.926** (vs 0.966 on real images) — 95% of baseline performance.

---

## 🎯 Problem Statement

- **Goal:** Translate CT imaging modalities between contrast-enhanced (arterial phase) and native (non-contrast) series
- **Challenge:** Diffusion models for CT images require enormous computational resources
- **Solution:** Flow Matching with vector field approximation between two distributions π₀ and π₁

---

## 🔬 Method

### Flow Matching

Given source distribution π₀ and target distribution π₁, we find a vector field v_θ such that the ODE solution:

$$\frac{dX_t}{dt} = v_\theta(X_t, t), \quad X_0 \sim \pi_0$$

transports π₀ to π₁ at t=1.

**Linear trajectory:**
$$X_t = (1-t)X_0 + tX_1$$

**Loss function:**
$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1]} \mathbb{E}_{X_0, X_1 \sim \pi_0 \times \pi_1} \|v_\theta(X_t, t) - v^*\|^2$$

### TimeResNet Architecture

We propose **TimeResNet** — a novel architecture with:
- Time embedding injection into each ResNet block
- Self-Convolution Attention at the bottleneck
- Group Normalization for training stability

---

## 📊 Results

### Model Comparison (Hold-out dataset, Contrast → Native)

| Model | MAE↓ | SSIM↑ | PSNR↑ | Time (s)↓ | Params (M) |
|-------|------|-------|-------|-----------|------------|
| **TimeResNet (ours)** | **5.436** | **0.996** | 39.776 | 0.209 | 124.7 |
| SwinUNETR | 6.229 | 0.992 | 38.933 | 0.087 | 120.1 |
| SegResNet | 6.146 | 0.983 | **40.108** | **0.056** | 214.1 |
| DiffusionNet | 18.203 | 0.934 | 30.051 | 44.200 | 108.4 |

---

## 🗂 Repository Structure

```
├── paper/                    # Paper source files (LaTeX)
│   ├── paper.tex            # Main paper
│   ├── paper.sty            # Style file
│   ├── references.bib       # Bibliography
│   └── images/              # Paper figures
│
├── slides/                   # Presentation slides (LaTeX)
│   ├── slides.tex           # Beamer presentation
│   └── images/              # Slide figures
│
├── src/                      # Source code
│   ├── metrics.ipynb        # Metrics computation notebook
│   ├── plot.ipynb           # Visualization notebook
│   └── experiments/
│       ├── configs/         # Training configurations (YAML)
│       │   ├── flow_match_mynet.yaml
│       │   ├── flow_match_segresnet.yaml
│       │   ├── flow_match_swinunetr.yaml
│       │   └── ...
│       ├── logs/            # Training logs (JSON)
│       └── trainers/        # Model implementations
│           ├── flow_matching.py    # Flow Matching trainer
│           ├── flow_model.py       # TimeResNet model
│           ├── diffusion_trainer.py # DDPM trainer
│           ├── dif_model.py        # Diffusion model
│           ├── reg_trainer.py      # Regression trainer
│           ├── dataset.py          # Data loading
│           ├── UMambaBot_2d.py     # UMamba architecture
│           └── utils.py            # Utilities
│
└── README.md
```

---

## 🛠 Installation & Usage

### Requirements

- Python 3.9+
- PyTorch 2.0+
- MONAI
- CUDA-compatible GPU (24GB+ recommended)

---

## 📈 Dataset

- **Total:** 120 abdominal CT studies (52,561 images)
- **Train:** 80 studies (34,992 images)
- **Test:** 20 studies (8,197 images)  
- **Hold-out:** 20 studies (9,372 images)

Each study contains paired native and arterial-phase contrast CT images.

**Preprocessing:**
- Image size: 512×512
- HU clipping: [-1000, 1000] → normalized to [-1, 1]
- Registration: Contrast series aligned to native using ANTs

---

## 📝 Key Contributions

1. **Novel image-to-image translation method** — significantly lower memory requirements than 3D approaches while maintaining axial slice consistency
2. **TimeResNet architecture** — achieves state-of-the-art results compared to existing architectures in MONAI
3. **Bidirectional translation** — single network for both Contrast→Native and Native→Contrast, halving training time

---

## 🏥 Clinical Impact

- **Radiation reduction:** ~50% dose reduction per examination by eliminating need for additional native scan
- **Dataset expansion:** Can double available pathology-image CT datasets via bidirectional conversion
- **Accessibility:** Enables virtual non-contrast imaging without dual-energy CT scanners

---

## 📚 Citation

```bibtex
@mastersthesis{kreinin2025mfm,
  title={Medical Flow Matching CT Translation},
  author={Kreinin, Matvei},
  school={Moscow Institute of Physics and Technology},
  year={2025},
  type={Master's Thesis}
}
```

---

## 📧 Contact

- **Author:** Matvei Kreinin — kreinin.mv@phystech.edu

