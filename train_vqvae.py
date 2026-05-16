"""
Training script for the VQ-VAE multitask model.
Loads (or collects) the 10k frame dataset, then trains the VQ-VAE with EMA
codebook updates, MSE reconstruction loss, weighted CE classification loss,
and the commitment loss from the VQ layer.

Usage: python train_vqvae.py
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
from src.dataset import load_or_collect, build_loaders_single
from src.models import VQVAE


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.training.seed)
    device = get_device()

    wandb.init(
        project=cfg.wandb.project,
        name=f"VQ-VAE-{cfg.model.num_embeddings}codes",
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
    set_seed(cfg.training.seed)
    model = VQVAE(
        num_embeddings=cfg.model.num_embeddings,
        embedding_dim=cfg.model.embedding_dim,
        commitment_cost=cfg.model.commitment_cost,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
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
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "vqvae.pth")

    history = {'tot': [], 'mse': [], 'ce': [], 'vq': [], 'lr': [], 'active_tokens': []}
    print(f"Training VQ-VAE on {device}...")

    ALPHA = cfg.model.training.alpha_mse
    BETA = cfg.model.training.beta_ce

    for epoch in range(cfg.model.training.epochs):
        model.train()
        loss_total_avg = make_averager()
        loss_recon_avg = make_averager()
        loss_class_avg = make_averager()
        loss_vq_avg = make_averager()
        all_indices = []

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{cfg.model.training.epochs}]")
        for batch_images, batch_labels in loop:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device).long()

            x_recon, vq_loss, logits = model(batch_images)
            recon_loss = F.mse_loss(x_recon, batch_images)
            class_loss = F.cross_entropy(logits, batch_labels, weight=class_weights)
            total_loss = ALPHA * recon_loss + BETA * class_loss + vq_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track active codebook tokens
            with torch.no_grad():
                z = model.encoder(batch_images)
                z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, cfg.model.embedding_dim)
                dists = (
                    torch.sum(z_flat ** 2, dim=1, keepdim=True)
                    + torch.sum(model.vq.embeddings ** 2, dim=1)
                    - 2 * torch.matmul(z_flat, model.vq.embeddings.t())
                )
                indices = torch.argmin(dists, dim=1)
                all_indices.append(indices.cpu())

            loss_total_avg(total_loss.item())
            loss_recon_avg(recon_loss.item())
            loss_class_avg(class_loss.item())
            loss_vq_avg(vq_loss.item())

            loop.set_postfix(
                tot=loss_total_avg(),
                mse=loss_recon_avg(),
                ce=loss_class_avg(),
                vq=loss_vq_avg()
            )

        active = len(torch.unique(torch.cat(all_indices)))
        history['tot'].append(loss_total_avg())
        history['mse'].append(loss_recon_avg())
        history['ce'].append(loss_class_avg())
        history['vq'].append(loss_vq_avg())
        history['lr'].append(scheduler.get_last_lr()[0])
        history['active_tokens'].append(active)

        wandb.log({
            "epoch": epoch + 1,
            "total_loss": loss_total_avg(),
            "mse": loss_recon_avg(),
            "ce": loss_class_avg(),
            "vq_loss": loss_vq_avg(),
            "lr": scheduler.get_last_lr()[0],
            "active_tokens": active,
        })
        print(f"  Epoch {epoch+1}: active tokens = {active}/{cfg.model.num_embeddings}")

        scheduler.step()

        # Atomic checkpoint save
        temp_path = ckpt_path + '.tmp'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
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