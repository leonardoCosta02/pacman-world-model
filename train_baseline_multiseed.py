"""
Multi-seed training script for the Baseline (PacmanWorldModel).

USAGE on Kaggle/Colab (from project root):
    python train_baseline_multiseed.py training.seed=42
    python train_baseline_multiseed.py training.seed=123
    python train_baseline_multiseed.py training.seed=7

Each run:
  - Re-collects/re-loads the 10k dataset with the given seed
  - Trains the Baseline for 10 epochs
  - Evaluates on the test set (accuracy + precision/recall/F1 for DANGER)
  - Saves checkpoint as: checkpoints/baseline_seed{SEED}.pth
  - Saves metrics as:   checkpoints/baseline_seed{SEED}_metrics.json
"""
import os
import json
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import set_seed, make_averager, get_device
from src.dataset import load_or_collect, build_loaders_single
from src.models import PacmanWorldModel


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    SEED = cfg.training.seed
    print(f"\n{'='*70}\nMulti-seed Baseline training -- SEED={SEED}\n{'='*70}\n")

    set_seed(SEED)
    device = get_device()

    # Load data (deterministic w.r.t. seed)
    os.makedirs(os.path.dirname(cfg.dataset.cache_path), exist_ok=True)
    frames, labels = load_or_collect(
        cfg.dataset.cache_path, cfg.dataset.num_frames,
        danger_window=cfg.dataset.danger_window, seed=cfg.dataset.seed
    )
    train_loader, test_loader, class_counts = build_loaders_single(
        frames, labels, batch_size=cfg.training.batch_size,
        split_seed=SEED, sampler_seed=SEED
    )

    # Build model
    set_seed(SEED)
    model = PacmanWorldModel(latent_dim=cfg.model.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.model.training.learning_rate,
                           weight_decay=cfg.model.training.weight_decay)

    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCELoss(reduction='none')
    weight_for_danger = float(class_counts[0] / class_counts[1])
    print(f"Class weight DANGER: {weight_for_danger:.2f}")

    # Training loop
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    print(f"Training on {device}...")

    for epoch in range(cfg.model.training.epochs):
        model.train()
        loss_avg = make_averager()
        for images, lbls in tqdm(train_loader, desc=f"Ep [{epoch+1}/{cfg.model.training.epochs}]"):
            images = images.to(device)
            lbls = lbls.to(device).float().unsqueeze(1)
            recon, pred, _ = model(images)
            loss_mse = mse_loss_fn(recon, images)
            loss_bce_un = bce_loss_fn(pred, lbls)
            weights = torch.where(lbls == 1.0,
                                  torch.tensor(weight_for_danger, device=device),
                                  torch.tensor(1.0, device=device))
            loss_bce = (loss_bce_un * weights).mean()
            total_loss = cfg.model.training.alpha_mse * loss_mse + cfg.model.training.beta_bce * loss_bce
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_avg(total_loss.item())
        print(f"  Ep {epoch+1}: loss={loss_avg():.4f}")

    # ========== EVALUATION ==========
    print("\nEvaluating on test set...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.to(device)
            _, pred, _ = model(images)
            pred_bin = (pred.squeeze() >= 0.5).long().cpu().numpy()
            all_preds.extend(pred_bin)
            all_labels.extend(lbls.numpy())

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

    # Save checkpoint + metrics with seed in filename
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, f"baseline_seed{SEED}.pth")
    torch.save({'model_state_dict': model.state_dict(), 'seed': SEED}, ckpt_path)

    metrics = {
        'model': 'baseline',
        'seed': SEED,
        'accuracy': float(acc),
        'precision_danger': float(prec_d),
        'recall_danger': float(rec_d),
        'f1_danger': float(f1_d),
        'n_test': len(all_labels),
    }
    metrics_path = os.path.join(cfg.training.checkpoint_dir, f"baseline_seed{SEED}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved metrics:    {metrics_path}")


if __name__ == "__main__":
    main()
