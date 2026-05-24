# Multitask World Model per Ms. Pac-Man

Progetto sviluppato per il corso **Deep Learning & Applied AI (DLAI) 2025/2026** presso Sapienza Università di Roma, Prof. Rodolà.

## Panoramica

Sistema multitask basato su Encoder + due teste (Decoder + Classifier), esteso con VQ-VAE, Temporal Transformer e modelli generativi autoregressivi (Token-level Prior e Frame-level Prior). Il sistema costruisce un *World Model* capace di codificare, ricostruire, classificare e generare frame del gioco Ms. Pac-Man.

## Struttura del progetto

\`\`\`
pacman-world-model/
├── src/                       # Codice sorgente modulare
│   ├── utils.py               # set_seed, make_averager, helpers
│   ├── dataset.py             # Raccolta frame, Dataset PyTorch
│   ├── models.py              # Baseline, VectorQuantizer, VQVAE
│   └── transformers.py        # Temporal Classifier, Token Prior, Frame Prior
├── conf/                      # Configurazioni Hydra
│   ├── config.yaml
│   ├── model/                 # Iperparametri per modello
│   └── dataset/               # Configurazioni dataset (10k, 50k)
├── notebook/                  # Notebook esplorativo (launcher)
│   └── PacMan_WorldModel.ipynb
├── train_baseline.py          # Training script Baseline
├── train_vqvae.py             # Training script VQ-VAE
├── train_transformer.py       # Training script Temporal Transformer
├── train_token_prior.py       # Training script Token Prior
├── train_frame_prior.py       # Training script Frame Prior
└── test.py                    # Valutazione modelli sul test set
\`\`\`

> **Nota**: il notebook è un launcher minimale che importa il codice da `src/` e lancia i `train_*.py`. Tutto il codice del progetto è nei file modulari Python.

## Installazione

\`\`\`bash
git clone https://github.com/leonardoCosta02/pacman-world-model.git
cd pacman-world-model
pip install -r requirements.txt
\`\`\`

## Uso

### Training (ordine obbligato)

I modelli devono essere addestrati in sequenza perché ciascuno dipende dal precedente (i Transformer usano il VQ-VAE come feature extractor frozen):

\`\`\`bash
# 1. Baseline multitask
python train_baseline.py

# 2. VQ-VAE (richiede il dataset 10k già raccolto)
python train_vqvae.py

# 3. Temporal Transformer Classifier (usa VQ-VAE frozen)
python train_transformer.py

# 4. Token-level Prior (usa VQ-VAE frozen)
python train_token_prior.py

# 5. Frame-level Prior (richiede dataset 50k)
python train_frame_prior.py dataset=pacman_50k
\`\`\`

### Valutazione

\`\`\`bash
python test.py model=baseline
python test.py model=vqvae
python test.py model=transformer_classifier
\`\`\`

### Notebook

Per un'analisi visiva interattiva con tutti i test, le interpolazioni latenti e i rollout generativi:

\`\`\`bash
jupyter notebook notebook/PacMan_WorldModel.ipynb
\`\`\`

Il notebook è multi-ambiente: rileva automaticamente Kaggle / Colab / locale e configura i path di conseguenza.

## Pre-trained Weights

Per eseguire il notebook senza riaddestrare i modelli da zero, i pesi pre-addestrati sono disponibili su:

- **Kaggle Models** (pubblici): cerca `leonardocostantini02/modeels`, `modeels2`, `modeels3` e il dataset `dataseets`
- **Google Drive** (cartella `pacman-pesi`): per Colab

Configurazione per ambiente:
- **Kaggle**: attacca i Kaggle Models dal pannello "Add Input"
- **Colab**: copia i file `.pth` in `/content/drive/MyDrive/pacman-pesi/`
- **Locale**: salva i file `.pth` in `checkpoints/`

## Risultati

| Modello | Input | Test Accuracy | F1 DANGER |
|---|---|---|---|
| Baseline (continuo) | 1 frame | 92.40% | 0.62 |
| VQ-VAE (discreto) | 1 frame | 96.00% | 0.76 |
| Temporal Transformer | 8 frame | **97.85%** | **0.83** |

**Pruning** del VQ-VAE encoder al 20% (L1 Unstructured Global): accuracy preservata (97.85%), sparsità globale del 20%.

**Generazione autoregressiva**: il Frame-level Prior produce 15 frame coerenti, mantenendo la struttura del labirinto. Il Token-level Prior soffre invece di error accumulation dopo 1-2 frame.

## Riproducibilità

Tutti i seed sono fissati a 42 (`numpy`, `torch`, `cuda`, `cudnn` deterministico). Lo split train/test e i `WeightedRandomSampler` usano `Generator` espliciti. L'init di ogni modello viene riseedato per garantire riproducibilità anche con esecuzioni in ordini diversi.

## Autore

Leonardo Costantini --- Sapienza Università di Roma --- DLAI 2025/2026
