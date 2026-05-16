# Multitask World Model per Ms. Pac-Man

Progetto sviluppato per il corso **Deep Learning & Applied AI (DLAI) 2026** presso Sapienza Università di Roma, Prof. Rodolà.

## Panoramica

Sistema multitask basato su Encoder + due teste (Decoder + Classifier), esteso con VQ-VAE, Temporal Transformer e modelli generativi autoregressivi (Token-level Prior e Frame-level Prior). Il sistema costruisce un *World Model* capace di codificare, ricostruire, classificare e generare frame del gioco Ms. Pac-Man.

## Struttura del progetto
pacman-worldmodel/
├── src/                       # Codice sorgente modulare
│   ├── utils.py               # set_seed, make_averager, helpers
│   ├── dataset.py             # Raccolta frame, Dataset PyTorch
│   ├── models.py              # Baseline, VectorQuantizer, VQVAE
│   └── transformers.py        # Temporal Classifier, Token Prior, Frame Prior
├── conf/                      # Configurazioni Hydra
│   ├── config.yaml
│   ├── model/                 # Iperparametri per modello
│   └── dataset/               # Configurazioni dataset (10k, 50k)
├── notebook/                  # Notebook esplorativo
│   └── PacMan_WorldModel.ipynb
├── train_baseline.py          # Training script Baseline
├── train_vqvae.py             # Training script VQ-VAE
├── train_transformer.py       # Training script Temporal Transformer
├── train_token_prior.py       # Training script Token Prior
├── train_frame_prior.py       # Training script Frame Prior
└── test.py                    # Valutazione modelli sul test set
## Installazione

```bash
git clone https://github.com/<username>/pacman-worldmodel.git
cd pacman-worldmodel
pip install -r requirements.txt
```

## Uso

### Training (ordine obbligato)

I modelli devono essere addestrati in sequenza perché ciascuno dipende dal precedente:

```bash
# 1. Baseline multitask
python train_baseline.py

# 2. VQ-VAE (richiede il dataset 10k già raccolto)
python train_vqvae.py

# 3. Temporal Transformer Classifier (usa VQ-VAE frozen)
python train_transformer.py

# 4. Token-level Prior (usa VQ-VAE frozen)
python train_token_prior.py

# 5. Frame-level Prior (richiede dataset 50k)
python train_frame_prior.py
```

### Valutazione

```bash
python test.py model=baseline
python test.py model=vqvae
python test.py model=transformer_classifier
```

### Notebook

Per un'analisi visiva interattiva:

```bash
jupyter notebook notebook/PacMan_WorldModel.ipynb
```

## Risultati

| Modello | Input | Test Accuracy | F1 DANGER |
|---|---|---|---|
| Baseline (continuo) | 1 frame | 93.75% | 0.66 |
| VQ-VAE (discreto) | 1 frame | 96.15% | 0.76 |
| Temporal Transformer | 8 frame | **97.85%** | **0.83** |

Pruning del VQ-VAE encoder al 20%: accuracy preservata, sparsità globale del 20%.

Rollout generativo Frame Prior: 15+ frame autoregressivi coerenti.

## Autore

Leonardo Costantini --- Sapienza Università di Roma --- DLAI 2026