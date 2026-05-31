# Multitask World Model for Ms. Pac-Man

Project for the course **Deep Learning & Applied AI (DLAI)**, a.y. 2025/26 — Sapienza University of Rome, Prof. Emanuele Rodolà.

This project investigates the trade-off between continuous and discrete latent spaces for a multitask world model on Ms. Pac-Man, evaluating five architectures progressively extended from a shared encoder toward full autoregressive future simulation.

---

## Research Question

Given a severely class-imbalanced, non-Markovian prediction task on raw Atari frames, when does a discrete codebook outperform a continuous bottleneck, and can a Temporal Transformer bridge the gap between single-frame snapshots and predictive look-ahead labels?

---

## Models

The system is organized as five models, each motivated by a limitation of the previous one:

**M1 — Baseline.** Shared 3-layer CNN encoder with a reconstruction decoder (MSE) and a binary classifier (BCE). Labels are obtained via look-ahead self-supervision: a frame at time *t* is marked DANGER if a life is lost within the next 15 frames. Establishes the continuous-latent reference.

**M2 — VQ-VAE.** Replaces the continuous bottleneck with a discrete 128-token codebook (d=64), updated via EMA to prevent index collapse. Adds a cross-entropy classification head on the quantized grid.

**M3 — Temporal Transformer Classifier.** Processes 8-frame histories of frozen VQ-VAE grids with a 4-layer self-attention encoder and a learnable [CLS] token, directly targeting the non-Markovian label.

**M4 — Token-level Prior.** Causal Transformer decoder predicting 800 flattened tokens autoregressively (100 tokens/frame × 8 frames) via top-k sampling with scheduled sampling and noise injection to mitigate exposure bias.

**M5 — Frame-level Prior.** Transformer encoder predicting the next 64×10×10 unquantized latent in a single pass (MSE objective), trading stochasticity for geometric coherence.

---

## Project Structure

```
pacman-world-model/
├── src/
│   ├── utils.py                # set_seed, make_averager, helpers
│   ├── dataset.py              # Frame collection, Datasets, DataLoaders
│   ├── models.py               # Baseline, VectorQuantizer, VQ-VAE
│   └── transformers.py         # Temporal Classifier, Token Prior, Frame Prior
├── conf/
│   ├── config.yaml             # Main Hydra config
│   ├── model/                  # Per-model hyperparameters (5 yaml files)
│   └── dataset/                # Dataset configs (10k, 50k frames)
├── notebook/
│   └── PacMan_WorldModel.ipynb # Launcher notebook (Kaggle / Colab / local)
├── train_baseline.py
├── train_vqvae.py
├── train_transformer.py
├── train_token_prior.py
├── train_frame_prior.py
├── rollout_psnr.py             # Quantitative PSNR rollout comparison (Kaggle paths)
├── test.py
└── requirements.txt
```

> The notebook is a launcher that clones the repo, installs dependencies, loads pre-trained weights, and runs evaluations as subprocesses. All architecture and logic reside in the modular src/ files.

---

## Setup

```bash
git clone https://github.com/leonardoCosta02/pacman-world-model.git
cd pacman-world-model
pip install -r requirements.txt
```

Training requires a GPU (all experiments run on a free Kaggle/Colab T4). Local CPU is sufficient only for analysis.

---

## Reproducing the Experiments

Run scripts in dependency order; each model requires the previous checkpoint as a frozen feature extractor:

```bash
# M1 — Baseline (10 epochs, ~3 min on T4)
python train_baseline.py

# M2 — VQ-VAE (50 epochs, ~25 min on T4)
python train_vqvae.py

# M3 — Temporal Transformer Classifier (10 epochs, ~10 min on T4)
python train_transformer.py

# M4 — Token-level Prior (50 epochs, ~45 min on T4)
python train_token_prior.py

# M5 — Frame-level Prior on 50k dataset (30 epochs, ~30 min on T4)
python train_frame_prior.py dataset=pacman_50k
```

Evaluate classifiers on the test set:

```bash
python test.py model=baseline
python test.py model=vqvae
python test.py model=transformer_classifier
```

Quantitative PSNR rollout comparison (requires Kaggle paths configured in `rollout_psnr.py`):

```bash
python rollout_psnr.py
```

---

## Pre-trained Weights

To reproduce all results without retraining, attach the following resources on Kaggle or download from Google Drive:

| File | Description |
| :--- | :--- |
| `baseline_checkpoint.pth` | Baseline `PacmanWorldModel` |
| `vqvae_checkpoint.pth` | VQ-VAE, K=128 codebook |
| `transformer_classifier_checkpoint.pth` | Temporal Transformer Classifier |
| `transformer_prior_checkpoint.pth` | Token-level Prior |
| `frame_prior_checkpoint.pth` | Frame-level Prior |
| `raw_frames_50k.npz` | Cached 50k-frame dataset |
| `pacman_dream.gif` | Token-level autoregressive rollout |
| `pacman_dream_framelevel.gif` | Frame-level autoregressive rollout |

**Kaggle** (recommended): attach via *Add Input*:
- `leonardocostantini02/modeels` — VQ-VAE, Token Prior, Classifier
- `leonardocostantini02/modeels2` — Frame Prior
- `leonardocostantini02/modeels3` — Generated GIFs
- `leonardocostantini02/dataseets` — 50k frame dataset cache

**Google Drive**: [pacman-pesi folder (public)](https://drive.google.com/drive/folders/1-xMEXMLGdC1u5SMr4qf8qOoH15-nyaRy?usp=drive_link) — place under `checkpoints/` for local execution or `/content/drive/MyDrive/pacman-pesi/` for Colab.

---

## Results

Performance averaged over 3 independent random seeds (seed ∈ {7, 42, 123}):

| Model | Input | Accuracy (%) | F1 (DANGER) |
| :--- | :--- | :--- | :--- |
| Baseline (continuous) | 1 frame | 92.02 ± 1.13 | 0.60 ± 0.03 |
| VQ-VAE (discrete) | 1 frame | 96.25 ± 0.23 | 0.76 ± 0.01 |
| **Temporal Transformer** | **8 frames** | **98.32 ± 0.78** | **0.87 ± 0.06** |
| VQ-VAE + 20% L1 pruning | 1 frame | 96.38 ± 0.60 | 0.76 ± 0.03 |

**Pruning.** 20% unstructured L1 magnitude pruning of the VQ-VAE encoder 
yields $\Delta = +0.13 \pm 0.65$ pp across three seeds, confirming no 
statistically significant accuracy degradation. Global L1 pruning concentrates on the deepest convolutional layer (23% sparsity) while barely touching the early edge-detectors (2.9%), indicating high redundancy in the encoder's mapping into the codebook.

**Autoregressive rollouts.** PSNR measured over 15 steps across 50 unseen contexts: the token-level prior collapses to 14.33 ± 0.12 dB at step 1 due to categorical drift; the frame-level prior stabilizes at 30.79 ± 2.98 dB after 15 steps but biases moving agents toward the conditional mean (entity vanishing).

**Inference latency.** The Temporal Transformer processes 8-frame sequences in 1.45  ms on T4 GPU (690 sequences/s), satisfying 60 FPS real-time constraints with a 10× safety margin.

---

## Reproducibility

Single runs fix all seeds to 42 (`random`, `numpy`, `torch`, `cuda`, `cudnn.deterministic = True`); the reported ± std metrics come from multi-seed runs over {7, 42, 123}. Train/test splits and `WeightedRandomSampler` instances use explicit `torch.Generator` objects. The multi-seed variants (`train_*_multiseed.py`) and result aggregation (`aggregate_multiseed.py`) reproduce these numbers.

---

## References

- Ha & Schmidhuber. *Recurrent World Models Facilitate Policy Evolution.* NeurIPS 2018.
- van den Oord, Vinyals & Kavukcuoglu. *Neural Discrete Representation Learning.* NeurIPS 2017.
- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL-HLT 2019.
- Micheli, Alonso & Fleuret. *Transformers are Sample-Efficient World Models.* ICLR 2023.
- Hafner et al. *Mastering Diverse Domains through World Models.* arXiv 2301.04104, 2023.
- Esser, Rombach & Ommer. *Taming Transformers for High-Resolution Image Synthesis.* CVPR 2021.
- Kingma & Welling. *Auto-Encoding Variational Bayes.* ICLR 2014.
- Bengio et al. *Scheduled Sampling for Sequence Prediction with RNNs.* NeurIPS 2015.
- Han et al. *Learning both Weights and Connections for Efficient Neural Networks.* NeurIPS 2015.
- Frankle & Carbin. *The Lottery Ticket Hypothesis.* ICLR 2019.
- Bellemare et al. *The Arcade Learning Environment.* JAIR 47, 2013.
- Towers et al. *Gymnasium.* https://gymnasium.farama.org, 2023.
- Caruana. *Multitask Learning.* Machine Learning 28(1), 1997.

---

## AI Usage

In accordance with course guidelines, Google's Gemini was used as a coding and writing assistant to: (i) generate boilerplate PyTorch code and Hydra configurations for the five evaluated architectures; (ii) assist in debugging tensor shape mismatches during Kaggle deployments; (iii) correct grammar and improve the conciseness of the report manuscript. All core design decisions — model selection, the non-Markovian labelling scheme, the pruning experiments, and the formulation of the generative priors — were independently conceived. All numerical results were generated from independent executions on Kaggle GPUs and verified against the released checkpoints.

---

## Author

**Leonardo Costantini** — Computer Science Student, Sapienza University of Rome
`costantini.2009905@studenti.uniroma1.it`
