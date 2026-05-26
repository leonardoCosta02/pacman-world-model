"""
Multi-seed training script for the Temporal Transformer Classifier.

PREREQUISITES: a VQ-VAE checkpoint must exist at:
    checkpoints/vqvae_seed{SEED}.pth

For the multi-seed experiment we pair each Temporal seed with the
SAME-SEED VQ-VAE checkpoint, to keep the chain of randomness deterministic.

USAGE:
    python train_transformer_multiseed.py training.seed=42
    python train_transformer_multiseed.py training.seed=123
    python train_transformer_multiseed.py training.seed=7
"""
import os
import json
import hydra
import torch
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import set_seed, make_averager, get_device
from src.dataset import load_or_collect, build_loaders_sequence
from src.models import VQVAE
from src.transformers import TemporalTransformerClassifier


def load_vqvae_for_seed(ckpt_dir, seed, device):
    """Carica il VQ-VAE corrispondente a questo seed."""
    ckpt_path = os.path.join(ckpt_dir, f"vqvae_seed{seed}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"VQ-VAE checkpoint for seed={seed} not found at {ckpt_path}. "
            f"Run train_vqvae_multiseed.py first with training.seed={seed}"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_model = ckpt.get('config', {})
    model = VQVAE(
        num_embeddings=cfg_model.get('num_embeddings', 128),
        embedding_dim=cfg_model.get('embedding_dim', 64),
        commitment_cost=cfg_model.get('commitment_cost', 1.0),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded VQ-VAE from {ckpt_path}")
    return model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    SEED = cfg.training.seed
    print(f"\n{'='*70}\nMulti-seed Temporal Transformer -- SEED={SEED}\n{'='*70}\n")

    set_seed(SEED)
    device = get_device()

    # Load data with this seed
    os.makedirs(os.path.dirname(cfg.dataset.cache_path), exist_ok=True)
    frames, labels = load_or_collect(
        cfg.dataset.cache_path, cfg.dataset.num_frames,
        danger_window=cfg.dataset.danger_window, seed=cfg.dataset.seed
    )
    train_loader, _, test_loader, _ = build_loaders_sequence(
        frames, labels,
        seq_len=cfg.dataset.seq_len,
        batch_size=cfg.training.batch_size,
        split_seed=SEED,
        sampler_seed=SEED,
    )

    # Load the matching-seed VQ-VAE
    vqvae = load_vqvae_for_seed(cfg.training.checkpoint_dir, SEED, device)

    # Build Transformer
    set_seed(SEED)
    transformer = TemporalTransformerClassifier(
        vqvae_encoder=vqvae.encoder,
        vq_module=vqvae.vq,
        seq_len=cfg.dataset.seq_len,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, transformer.parameters()),
        lr=cfg.model.training.learning_rate,
        weight_decay=cfg.model.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.model.training.scheduler_step,
        gamma=cfg.model.training.scheduler_gamma,
    )
    class_weights = torch.tensor(
        [1.0, cfg.model.training.class_weight_danger],
        dtype=torch.float32
    ).to(device)

    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    print(f"Training on {device}...")

    for epoch in range(cfg.model.training.epochs):
        transformer.train()
        loss_avg = make_averager()
        for batch_seqs, batch_labels in tqdm(train_loader, desc=f"Ep [{epoch+1}/{cfg.model.training.epochs}]"):
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device).long()
            logits = transformer(batch_seqs)
            loss = F.cross_entropy(logits, batch_labels, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            loss_avg(loss.item())
        scheduler.step()
        print(f"  Ep {epoch+1}: loss={loss_avg():.4f}")

    # ========== EVALUATION ==========
    print("\nEvaluating on test set...")
    transformer.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_seqs, batch_labels in test_loader:
            batch_seqs = batch_seqs.to(device)
            logits = transformer(batch_seqs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec_d = precision_score(all_labels, all_preds, zero_division=0)
    rec_d = recall_score(all_labels, all_preds, zero_division=0)
    f1_d = f1_score(all_labels, all_preds, zero_division=0)

    print(f"\n{'='*70}")
    print(f"RESULTS for seed={SEED}:")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision DANGER: {prec_d:.4f}")
    print(f"  Recall DANGER   : {rec_d:.4f}")
    print(f"  F1 DANGER       : {f1_d:.4f}")
    print(f"{'='*70}\n")

    ckpt_path = os.path.join(cfg.training.checkpoint_dir, f"transformer_seed{SEED}.pth")
    torch.save({
        'model_state_dict': transformer.state_dict(),
        'seed': SEED,
        'config': OmegaConf.to_container(cfg.model, resolve=True),
    }, ckpt_path)

    metrics = {
        'model': 'transformer',
        'seed': SEED,
        'accuracy': float(acc),
        'precision_danger': float(prec_d),
        'recall_danger': float(rec_d),
        'f1_danger': float(f1_d),
        'n_test': len(all_labels),
    }
    metrics_path = os.path.join(cfg.training.checkpoint_dir, f"transformer_seed{SEED}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {ckpt_path}\nSaved: {metrics_path}")


if __name__ == "__main__":
    main()
