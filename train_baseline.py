"""
Training script for the Baseline (PacmanWorldModel).
Usage: python train_baseline.py
"""
import os
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.utils import set_seed, make_averager, get_device
from src.dataset import load_or_collect, build_loaders_single
from src.models import PacmanWorldModel


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.training.seed)
    device = get_device()

    # Init wandb
    wandb.init(
        project=cfg.wandb.project,
        name=f"Baseline-{cfg.model.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
    )

    # Load data
    os.makedirs(os.path.dirname(cfg.dataset.cache_path), exist_ok=True)
    frames, labels = load_or_collect(
        cfg.dataset.cache_path, cfg.dataset.num_frames,
        danger_window=cfg.dataset.danger_window, seed=cfg.dataset.seed
    )
    train_loader, test_loader, class_counts = build_loaders_single(
        frames, labels, batch_size=cfg.training.batch_size,
        split_seed=cfg.training.seed, sampler_seed=cfg.training.seed
    )

    # Build model
    set_seed(cfg.training.seed)  # re-seed before model init
    model = PacmanWorldModel(latent_dim=cfg.model.latent_dim).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.model.training.learning_rate,
        weight_decay=cfg.model.training.weight_decay
    )

    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCELoss(reduction='none')
    weight_for_danger = float(class_counts[0] / class_counts[1])
    print(f"Class weights: SAFE=1.0, DANGER={weight_for_danger:.2f}")

    # Training loop
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    history = {'tot': [], 'mse': [], 'bce': []}
    print(f"Training Baseline on {device}...")

    for epoch in range(cfg.model.training.epochs):
        model.train()
        loss_avg = make_averager()
        mse_avg = make_averager()
        bce_avg = make_averager()

        for images, lbls in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{cfg.model.training.epochs}]"):
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
            mse_avg(loss_mse.item())
            bce_avg(loss_bce.item())

        ep_tot, ep_mse, ep_bce = loss_avg(), mse_avg(), bce_avg()
        history['tot'].append(ep_tot)
        history['mse'].append(ep_mse)
        history['bce'].append(ep_bce)
        wandb.log({"epoch": epoch + 1, "total_loss": ep_tot, "mse": ep_mse, "bce": ep_bce})
        print(f"  Epoch {epoch+1}: tot={ep_tot:.4f}, mse={ep_mse:.6f}, bce={ep_bce:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "baseline.pth")
    torch.save({'model_state_dict': model.state_dict(), 'history': history}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    main()