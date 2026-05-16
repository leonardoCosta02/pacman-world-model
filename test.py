"""
Evaluation script for the classifier models.
Loads a checkpoint and reports accuracy + classification report on the test set.

Usage:
    python test.py model=baseline
    python test.py model=vqvae
    python test.py model=transformer_classifier
"""
import os
import hydra
import torch
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from src.utils import set_seed, get_device
from src.dataset import load_or_collect, build_loaders_single, build_loaders_sequence
from src.models import PacmanWorldModel, VQVAE
from src.transformers import TemporalTransformerClassifier


def evaluate_baseline(cfg, device):
    """Evaluate the Baseline (PacmanWorldModel)."""
    frames, labels = load_or_collect(
        cfg.dataset.cache_path, cfg.dataset.num_frames,
        danger_window=cfg.dataset.danger_window, seed=cfg.dataset.seed
    )
    _, test_loader, _ = build_loaders_single(
        frames, labels, batch_size=cfg.training.batch_size,
        split_seed=cfg.training.seed, sampler_seed=cfg.training.seed
    )

    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "baseline.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Baseline checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = PacmanWorldModel(latent_dim=cfg.model.latent_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.to(device)
            _, pred, _ = model(images)
            pred = (pred.squeeze() >= 0.5).long().cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(lbls.numpy())
    return all_preds, all_labels


def evaluate_vqvae(cfg, device):
    """Evaluate the VQ-VAE classifier head."""
    frames, labels = load_or_collect(
        cfg.dataset.cache_path, cfg.dataset.num_frames,
        danger_window=cfg.dataset.danger_window, seed=cfg.dataset.seed
    )
    _, test_loader, _ = build_loaders_single(
        frames, labels, batch_size=cfg.training.batch_size,
        split_seed=cfg.training.seed, sampler_seed=cfg.training.seed
    )

    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "vqvae.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_model = ckpt.get('config', {})

    model = VQVAE(
        num_embeddings=cfg_model.get('num_embeddings', 128),
        embedding_dim=cfg_model.get('embedding_dim', 64),
        commitment_cost=cfg_model.get('commitment_cost', 1.0),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.to(device)
            _, _, logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())
    return all_preds, all_labels


def evaluate_transformer(cfg, device):
    """Evaluate the Temporal Transformer Classifier."""
    frames, labels = load_or_collect(
        cfg.dataset.cache_path, cfg.dataset.num_frames,
        danger_window=cfg.dataset.danger_window, seed=cfg.dataset.seed
    )
    _, _, test_loader, _ = build_loaders_sequence(
        frames, labels,
        seq_len=cfg.dataset.seq_len,
        batch_size=cfg.training.batch_size,
        split_seed=cfg.training.seed,
        sampler_seed=cfg.training.seed,
    )

    # Load VQ-VAE
    vqvae_ckpt = torch.load(
        cfg.model.training.vqvae_checkpoint,
        map_location=device, weights_only=False
    )
    cfg_vq = vqvae_ckpt.get('config', {})
    vqvae = VQVAE(
        num_embeddings=cfg_vq.get('num_embeddings', 128),
        embedding_dim=cfg_vq.get('embedding_dim', 64),
        commitment_cost=cfg_vq.get('commitment_cost', 1.0),
    ).to(device)
    vqvae.load_state_dict(vqvae_ckpt['model_state_dict'])
    vqvae.eval()

    # Load Transformer
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "transformer_classifier.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Transformer checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = TemporalTransformerClassifier(
        vqvae_encoder=vqvae.encoder,
        vq_module=vqvae.vq,
        seq_len=cfg.dataset.seq_len,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_seqs, lbls in test_loader:
            batch_seqs = batch_seqs.to(device)
            logits = model(batch_seqs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())
    return all_preds, all_labels


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.training.seed)
    device = get_device()
    print(f"Evaluating model: {cfg.model.name} on {device}")

    if cfg.model.name == "baseline":
        preds, labels = evaluate_baseline(cfg, device)
    elif cfg.model.name == "vqvae":
        preds, labels = evaluate_vqvae(cfg, device)
    elif cfg.model.name == "transformer_classifier":
        preds, labels = evaluate_transformer(cfg, device)
    else:
        raise ValueError(f"Unknown model for evaluation: {cfg.model.name}")

    print("\n========== RESULTS ==========")
    print(f"Accuracy:  {accuracy_score(labels, preds):.4f}")
    print(f"Precision: {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(labels, preds, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(labels, preds, zero_division=0):.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(labels, preds, target_names=['SAFE', 'DANGER'], zero_division=0))
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()