# Multitask World Model for Ms. Pac-Man

Project developed for the **Deep Learning & Applied AI (DLAI) 2025/2026** course at Sapienza University of Rome, under the supervision of Prof. Emanuele Rodolà.

---

## Overview

A multitask system based on a shared **Encoder** and two task-specific heads (**Decoder** + **Classifier**), progressively extended with a **VQ-VAE**, a **Temporal Transformer**, and two autoregressive generative models (**Token-level Prior** and **Frame-level Prior**). The system builds a *World Model* capable of encoding, reconstructing, classifying, and generating future frames of the Ms. Pac-Man arcade game.

The labels (SAFE vs DANGER) are obtained via **look-ahead self-supervision**: a frame is marked DANGER if Pac-Man loses a life within the next 15 frames. Labels are therefore **predictive**, not descriptive — this motivates the introduction of temporal models such as the Temporal Transformer Classifier.

---

## Project Structure

````
pacman-world-model/
├── src/                              # Modular source code
│   ├── utils.py                      # set_seed, make_averager, helpers
│   ├── dataset.py                    # Frame collection, PyTorch Datasets, DataLoaders
│   ├── models.py                     # Baseline, VectorQuantizer, VQ-VAE
│   └── transformers.py               # Temporal Classifier, Token Prior, Frame Prior
│
├── conf/                             # Hydra configurations
│   ├── config.yaml                   # Main config
│   ├── model/                        # Per-model hyperparameters
│   │   ├── baseline.yaml
│   │   ├── vqvae.yaml
│   │   ├── transformer_classifier.yaml
│   │   ├── token_prior.yaml
│   │   └── frame_prior.yaml
│   └── dataset/                      # Dataset configs (10k, 50k frames)
│       ├── pacman_10k.yaml
│       └── pacman_50k.yaml
│
├── notebook/
│   └── PacMan_WorldModel.ipynb       # Main interactive notebook (launcher)
│
├── train_baseline.py                 # Baseline training script
├── train_vqvae.py                    # VQ-VAE training script
├── train_transformer.py              # Temporal Transformer training script
├── train_token_prior.py              # Token-level Prior training script
├── train_frame_prior.py              # Frame-level Prior training script
├── test.py                           # Evaluation script
└── requirements.txt                  # Python dependencies
````

> **Note**: The notebook acts as a minimal launcher that imports the code from `src/` and runs the training/evaluation pipelines as subprocesses. The core architecture and logic strictly reside in the modular Python files.
> **Note**: The notebook acts as a minimal launcher that imports the code from `src/` and runs the training/evaluation pipelines as subprocesses. The core architecture and logic strictly reside in the modular Python files.

---

## Quick Start (Recommended)

The simplest way to reproduce all results is by running the notebook in a pre-configured environment (Kaggle or Google Colab). The notebook automatically:

1. Detects the environment (Kaggle / Colab / local)
2. Clones the repository from GitHub
3. Installs missing dependencies
4. Loads pre-trained weights from the corresponding source
5. Runs all evaluations, latent interpolations, and generative rollouts

### Option 1 — Kaggle (recommended)

1. Upload `notebook/PacMan_WorldModel.ipynb` to Kaggle
2. Enable a **T4 GPU** accelerator
3. Attach the following public Kaggle Models / Datasets via *Add Input*:
   - `leonardocostantini02/modeels` (VQ-VAE, Token Prior, Classifier)
   - `leonardocostantini02/modeels2` (Frame Prior)
   - `leonardocostantini02/modeels3` (Generated GIFs)
   - `leonardocostantini02/dataseets` (50k frame dataset cache)
4. Set optionally `WANDB_API_KEY`
5. Run all cells

### Option 2 — Google Colab

1. Open `notebook/PacMan_WorldModel.ipynb` on Colab
2. Enable a GPU runtime (`Runtime → Change runtime type → T4 GPU`)
3. Mount Google Drive and place the pre-trained weights in `/content/drive/MyDrive/pacman-pesi/`. The weights folder is available here:
   - **[Pre-trained weights (Google Drive)](https://drive.google.com/drive/folders/1-xMEXMLGdC1u5SMr4qf8qOoH15-nyaRy?usp=drive_link)** *(public)*
4. Set the Colab secret `GITHUB_TOKEN` (and optionally `WANDB_API_KEY`)
5. Run all cells

### Option 3 — Local execution

```bash
git clone https://github.com/leonardoCosta02/pacman-world-model.git
cd pacman-world-model
pip install -r requirements.txt
jupyter notebook notebook/PacMan_WorldModel.ipynb
```

For local execution, place the pre-trained weights inside a `checkpoints/` folder at the repository root. The required files are listed below.

---

## Pre-trained Weights

To run the notebook without retraining (≈3 hours on a T4 GPU), the following files are required:

| File | Description |
| :--- | :--- |
| `baseline_checkpoint.pth` | Baseline `PacmanWorldModel` |
| `vqvae_checkpoint.pth` | VQ-VAE with 128-token codebook |
| `transformer_classifier_checkpoint.pth` | Temporal Transformer Classifier |
| `transformer_prior_checkpoint.pth` | Token-level Prior |
| `frame_prior_checkpoint.pth` | Frame-level Prior |
| `raw_frames_50k.npz` | Cached 50k frame dataset |
| `pacman_dream.gif` | Token-level rollout GIF |
| `pacman_dream_framelevel.gif` | Frame-level rollout GIF |

All weights are publicly available on:

- **Google Drive (`pacman-pesi` folder):** [Download here](https://drive.google.com/drive/folders/1-xMEXMLGdC1u5SMr4qf8qOoH15-nyaRy?usp=drive_link)
- **Kaggle Models / Datasets:** `leonardocostantini02/modeels`, `modeels2`, `modeels3`, `dataseets`

---

## Manual Training (Optional)

If you wish to retrain the models from scratch instead of using the pre-trained weights, run the scripts in the following order. Each model depends on the previous one (the Transformers reuse the frozen VQ-VAE as feature extractor):

```bash
# 1. Baseline multitask model (10 epochs, ~3 min on T4)
python train_baseline.py

# 2. VQ-VAE (50 epochs, ~25 min on T4)
python train_vqvae.py

# 3. Temporal Transformer Classifier (10 epochs, ~10 min)
python train_transformer.py

# 4. Token-level Prior (50 epochs, ~45 min)
python train_token_prior.py

# 5. Frame-level Prior on 50k dataset (30 epochs, ~30 min)
python train_frame_prior.py dataset=pacman_50k
```

### Evaluation

To evaluate any of the classifiers on the test set:

```bash
python test.py model=baseline
python test.py model=vqvae
python test.py model=transformer_classifier
```

---

## Experimental Results

### Classification

Adding temporal context drastically improves detection of the rare **DANGER** class, significantly reducing false positives while preserving recall.

| Model | Input | Test Accuracy | F1 (DANGER) |
| :--- | :--- | :--- | :--- |
| Baseline (continuous latent) | 1 frame | 92.40% | 0.62 |
| VQ-VAE (discrete latent) | 1 frame | 96.00% | 0.76 |
| **Temporal Transformer** | **8 frames** | **97.85%** | **0.83** |

**Inference benchmark:** The Temporal Transformer processes 8-frame sequences in **1.59 ms** on a T4 GPU (629 sequences/sec), with a 10× safety margin against the native 60 FPS refresh rate.

### Pruning

Applying **20% L1 Unstructured Global Pruning** to the VQ-VAE encoder leaves downstream accuracy completely intact (97.85% → 97.85%, Δ = 0.00%). The pruning is non-uniform across layers (largest layer: 23.0% sparse, first layer: 2.9% sparse), indicating healthy over-parameterization and robustness of the learned features.

### Generative Priors

- **Token-level Prior** generates 100 tokens per frame autoregressively. Locally coherent structures emerge in the first 1–2 frames, but exposure bias and categorical drift produce noise after ~200 autoregressive steps.
- **Frame-level Prior** generates entire latents in a single pass, maintaining coherent maze structures for 15+ steps. However, the deterministic MSE objective induces regression-to-the-mean, slightly blurring moving entities (Pac-Man, ghosts).

The two priors illustrate a classical trade-off in autoregressive generative models: **fine granularity + stochastic sampling** (Token Prior) versus **global operation + deterministic prediction** (Frame Prior).

---

## Reproducibility

All random seeds are strictly set to **42** (`numpy`, `random`, `torch`, `cuda`, and deterministic `cudnn`). Train/test splits and `WeightedRandomSampler` instances use explicit PyTorch `Generator` objects. Model initialization is re-seeded immediately before instantiation, guaranteeing perfectly reproducible weights regardless of execution order.

Subprocess-based training in the notebook may show small (±1.5%) accuracy variations on the Baseline due to RNG state isolation between the parent kernel and the launched script.

---

## Author

**Leonardo Costantini** — Computer Science Student  
Sapienza University of Rome  
Deep Learning & Applied AI (2025/2026)
├── train_frame_prior.py              # Frame-level Prior training script
├── test.py                           # Evaluation script
└── requirements.txt                  # Python dependencies
