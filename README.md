# Multitask World Model for Ms. Pac-Man

Project developed for the **Deep Learning & Applied AI (DLAI) 2025/2026** course at Sapienza University of Rome, under the supervision of Prof. Emanuele Rodolà.

## Overview

A multitask system based on a shared Encoder and two heads (Decoder + Classifier), progressively extended with a VQ-VAE, a Temporal Transformer, and two autoregressive generative models (Token-level Prior and Frame-level Prior). The system builds a *World Model* capable of encoding, reconstructing, classifying, and generating future frames of the Ms. Pac-Man arcade game.

## Project Structure

* `src/` - Modular source code
  * `utils.py` - set_seed, make_averager, helpers
  * `dataset.py` - Frame collection, PyTorch Datasets, DataLoaders
  * `models.py` - Baseline, VectorQuantizer, VQ-VAE
  * `transformers.py` - Temporal Classifier, Token Prior, Frame Prior
* `conf/` - Hydra configurations
  * `config.yaml` - Main config file
  * `model/` - Hyperparameters for each model
  * `dataset/` - Dataset configs (10k, 50k)
* `notebook/` - Exploratory notebook (launcher)
  * `PacMan_WorldModel.ipynb`
* `train_baseline.py` - Baseline training script
* `train_vqvae.py` - VQ-VAE training script
* `train_transformer.py` - Temporal Transformer training script
* `train_token_prior.py` - Token-level Prior training script
* `train_frame_prior.py` - Frame-level Prior training script
* `test.py` - Evaluation script for test set metrics

> **Note**: The notebook acts as a minimal launcher that imports the code from `src/` and runs the training/evaluation pipelines. The core architecture and logic strictly reside in the modular Python files.

## Installation

```bash
git clone [https://github.com/leonardoCosta02/pacman-world-model.git](https://github.com/leonardoCosta02/pacman-world-model.git)
cd pacman-world-model
pip install -r requirements.txt
## Usage

### Training (Sequential Order)

The models must be trained sequentially because each relies on the previous one (e.g., the Transformers use the pre-trained VQ-VAE as a frozen feature extractor):

```bash
# 1. Baseline multitask
python train_baseline.py

# 2. VQ-VAE (requires the 10k dataset)
python train_vqvae.py

# 3. Temporal Transformer Classifier (uses frozen VQ-VAE)
python train_transformer.py

# 4. Token-level Prior (uses frozen VQ-VAE)
python train_token_prior.py

# 5. Frame-level Prior (requires the 50k dataset)
python train_frame_prior.py dataset=pacman_50k
### Evaluation

To evaluate the classifiers on the test set and extract Accuracy, Precision, Recall, and F1 scores:

```bash
python test.py model=baseline
python test.py model=vqvae
python test.py model=transformer_classifier
### Exploratory Notebook

For interactive visual analysis, latent space interpolations, and generative rollouts:

```bash
jupyter notebook notebook/PacMan_WorldModel.ipynb
## Pre-trained Weights

To run the notebook or evaluation scripts without retraining from scratch, pre-trained weights are available:

- **Kaggle Models (Public):** search for `leonardocostantini02/modeels`, `modeels2`, `modeels3`, and the dataset `dataseets`.
- **Google Drive (`pacman-pesi` folder):** for Google Colab integration.

**Environment Setup:**

- **Kaggle:** Attach the Kaggle Models via the "Add Input" panel.
- **Colab:** Copy the `.pth` files into `/content/drive/MyDrive/pacman-pesi/`.
- **Local:** Save the `.pth` files inside the `checkpoints/` directory.

## Experimental Results

### Classification

Adding temporal context drastically improves the detection of the rare DANGER class, significantly reducing false positives.

| Model | Input | Test Accuracy | F1 (DANGER) |
| :--- | :--- | :--- | :--- |
| Baseline (Continuous) | 1 frame | 92.40% | 0.62 |
| VQ-VAE (Discrete) | 1 frame | 96.00% | 0.76 |
| Temporal Transformer | 8 frames | **97.85%** | **0.83** |

*Inference Benchmark:* The Temporal Transformer processes 8-frame sequences in 1.48 ms on a T4 GPU (675 sequences/sec), performing well within real-time 60 FPS constraints.

### Pruning

Applying a 20% L1 Unstructured Global Pruning to the VQ-VAE encoder leaves downstream accuracy completely intact (97.85%). The sparsity disproportionately targets the largest convolutional layer (23%) while preserving the edge-detecting first layer (2.9%), indicating a healthy over-parameterization for the multitask objective.

### Generative Priors

- **Frame-level Prior:** Generates full latents in a single pass, maintaining coherent maze structures for 15+ steps. However, the MSE objective induces regression-to-the-mean, blurring moving entities.
- **Token-level Prior:** Generates locally coherent structures but requires 100 autoregressive steps per frame, leading to exposure bias and rapid categorical drift after 1-2 frames.

## Reproducibility

All random seeds are strictly set to 42 (`numpy`, `random`, `torch`, `cuda`, and deterministic `cudnn`). Train/test splits and `WeightedRandomSampler` instances use explicit PyTorch Generators. Model initialization is re-seeded before instantiation to guarantee perfectly reproducible weights regardless of the execution order.

## Author

**Leonardo Costantini** - Computer Science Student — Sapienza University of Rome  
Deep Learning & Applied AI (2025/2026)
