"""
Training script for the Frame-level Prior.
Predicts next latent z (continuous) given 8 previous latents.
Uses MSE loss on latent space (regression).

Requires dataset=pacman_50k for adequate training data.

Usage: python train_frame_prior.py dataset=pacman_50k
"""
import os
import hydra
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.utils import set_seed, make_averager, get_device
from src.dataset import load_or_collect, build_loaders_frame_prior
from src.models import VQVAE
from src.transformers import FrameLevelPrior


def load_vqvae(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"VQ-VAE checkpoint not found at {ckpt_path}.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_model = ckpt.get('config', {})
    model = VQVAE(
        num_embeddings=cfg_model.get('num_embeddings', 128),
        embedding_dim=cfg_model.get('embedding_dim', 64),
        commitment_cost=cfg_model.get('commitment_cost', 1.0),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.training.seed)
    device = get_device()

    wandb.init(
        project=cfg.wandb.project,
        name=f"FramePrior-{cfg.dataset.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
    )

    # Load data
    os.makedirs(os.path.dirname(cfg.dataset.cache_path), exist_ok=True)
    frames, _ = load_or_collect(
        cfg.dataset.cache_path, cfg.dataset.num_frames,
        danger_window=cfg.dataset.danger_window, seed=cfg.dataset.seed
    )
    train_loader, test_loader = build_loaders_frame_prior(
        frames,
        seq_len=cfg.dataset.seq_len,
        target_len=cfg.model.target_len,
        batch_size=cfg.model.training.batch_size,
        split_seed=cfg.training.seed,
    )

    # Load frozen VQ-VAE
    vqvae = load_vqvae(cfg.model.training.vqvae_checkpoint, device)

    # Build Frame Prior
    set_seed(cfg.training.seed)
    frame_prior = FrameLevelPrior(
        latent_channels=vqvae.vq.embedding_dim,
        latent_h=10, latent_w=10,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = optim.Adam(
        frame_prior.parameters(),
        lr=cfg.model.training.learning_rate,
        weight_decay=cfg.model.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.model.training.scheduler_step,
        gamma=cfg.model.training.scheduler_gamma,
    )

    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "frame_prior.pth")

    history = {'train_mse': [], 'test_mse': [], 'lr': []}
    print(f"Training Frame-level Prior on {device}...")

    for epoch in range(cfg.model.training.epochs):
        # TRAINING
        frame_prior.train()
        train_avg = make_averager()

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{cfg.model.training.epochs}]")
        for context, target in loop:
            context = context.to(device)
            target = target.to(device)
            B, T, C, H, W = context.shape

            with torch.no_grad():
                ctx_flat = context.view(B * T, C, H, W)
                z_ctx = vqvae.encoder(ctx_flat).view(B, T, vqvae.vq.embedding_dim, 10, 10)

                tgt_flat = target.view(B * cfg.model.target_len, C, H, W)
                z_target = vqvae.encoder(tgt_flat)
                if z_target.dim() == 5:
                    z_target = z_target[:, 0]

            z_pred = frame_prior(z_ctx)
            loss = F.mse_loss(z_pred, z_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(frame_prior.parameters(), max_norm=1.0)
            optimizer.step()

            train_avg(loss.item())
            loop.set_postfix(mse=train_avg())

        # EVAL
        frame_prior.eval()
        test_avg = make_averager()
        with torch.no_grad():
            for context, target in test_loader:
                context = context.to(device)
                target = target.to(device)
                B, T, C, H, W = context.shape

                ctx_flat = context.view(B * T, C, H, W)
                z_ctx = vqvae.encoder(ctx_flat).view(B, T, vqvae.vq.embedding_dim, 10, 10)

                tgt_flat = target.view(B * cfg.model.target_len, C, H, W)
                z_target = vqvae.encoder(tgt_flat)
                if z_target.dim() == 5:
                    z_target = z_target[:, 0]

                z_pred = frame_prior(z_ctx)
                loss = F.mse_loss(z_pred, z_target)
                test_avg(loss.item())

        ep_train = train_avg()
        ep_test = test_avg()

        history['train_mse'].append(ep_train)
        history['test_mse'].append(ep_test)
        history['lr'].append(scheduler.get_last_lr()[0])

        wandb.log({
            "epoch": epoch + 1,
            "train_mse": ep_train,
            "test_mse": ep_test,
            "lr": scheduler.get_last_lr()[0],
        })
        print(f"  Train MSE: {ep_train:.6f} | Test MSE: {ep_test:.6f}")

        scheduler.step()

        temp_path = ckpt_path + '.tmp'
        torch.save({
            'epoch': epoch,
            'model_state_dict': frame_prior.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'config': OmegaConf.to_container(cfg.model, resolve=True),
        }, temp_path)
        os.replace(temp_path, ckpt_path)

    print(f"Saved checkpoint to {ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    main()