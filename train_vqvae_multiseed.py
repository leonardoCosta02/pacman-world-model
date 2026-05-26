"""
Multi-seed training script for the VQ-VAE multitask.

USAGE on Kaggle/Colab:
    python train_vqvae_multiseed.py training.seed=42
    python train_vqvae_multiseed.py training.seed=123
    python train_vqvae_multiseed.py training.seed=7

Saves checkpoint as: checkpoints/vqvae_seed{SEED}.pth
Saves metrics as:    checkpoints/vqvae_seed{SEED}_metrics.json
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
from src.dataset import load_or_collect, build_loaders_single
from src.models import VQVAE


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    SEED = cfg.training.seed
    print(f"\n{'='*70}\nMulti-seed VQ-VAE training -- SEED={SEED}\n{'='*70}\n")

    set_seed(SEED)
    device = get_device()

    # Load data
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
    model = VQVAE(
        num_embeddings=cfg.model.num_embeddings,
        embedding_dim=cfg.model.embedding_dim,
        commitment_cost=cfg.model.commitment_cost,
    ).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.model.training.learning_rate,
                           weight_decay=cfg.model.training.weight_decay)
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

    ALPHA = cfg.model.training.alpha_mse
    BETA = cfg.model.training.beta_ce

    for epoch in range(cfg.model.training.epochs):
        model.train()
        loss_avg = make_averager()
        for batch_images, batch_labels in tqdm(train_loader, desc=f"Ep [{epoch+1}/{cfg.model.training.epochs}]"):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device).long()
            x_recon, vq_loss, logits = model(batch_images)
            recon_loss = F.mse_loss(x_recon, batch_images)
            class_loss = F.cross_entropy(logits, batch_labels, weight=class_weights)
            total_loss = ALPHA * recon_loss + BETA * class_loss + vq_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_avg(total_loss.item())
        scheduler.step()
        print(f"  Ep {epoch+1}: loss={loss_avg():.4f}")

    # ========== EVALUATION ==========
    print("\nEvaluating on test set...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.to(device)
            _, _, logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec_d = precision_score(all_labels, all_preds, zero_division=0)
    rec_d = recall_score(all_labels, all_preds, zero_division=0)
    f1_d = f1_score(all_labels, all_preds, zero_division=0)

    # Conta token attivi
    all_indices = []
    with torch.no_grad():
        for batch_images, _ in test_loader:
            batch_images = batch_images.to(device)
            z = model.encoder(batch_images)
            z_flat = z.permute(0,2,3,1).contiguous().view(-1, cfg.model.embedding_dim)
            dists = (torch.sum(z_flat**2, dim=1, keepdim=True)
                     + torch.sum(model.vq.embeddings**2, dim=1)
                     - 2 * torch.matmul(z_flat, model.vq.embeddings.t()))
            idx = torch.argmin(dists, dim=1)
            all_indices.append(idx.cpu())
    active_tokens = len(torch.unique(torch.cat(all_indices)))

    print(f"\n{'='*70}")
    print(f"RESULTS for seed={SEED}:")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision DANGER: {prec_d:.4f}")
    print(f"  Recall DANGER   : {rec_d:.4f}")
    print(f"  F1 DANGER       : {f1_d:.4f}")
    print(f"  Active tokens   : {active_tokens}/{cfg.model.num_embeddings}")
    print(f"{'='*70}\n")

    ckpt_path = os.path.join(cfg.training.checkpoint_dir, f"vqvae_seed{SEED}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'seed': SEED,
        'config': OmegaConf.to_container(cfg.model, resolve=True),
    }, ckpt_path)

    metrics = {
        'model': 'vqvae',
        'seed': SEED,
        'accuracy': float(acc),
        'precision_danger': float(prec_d),
        'recall_danger': float(rec_d),
        'f1_danger': float(f1_d),
        'active_tokens': int(active_tokens),
        'n_test': len(all_labels),
    }
    metrics_path = os.path.join(cfg.training.checkpoint_dir, f"vqvae_seed{SEED}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {ckpt_path}\nSaved: {metrics_path}")


if __name__ == "__main__":
    main()
