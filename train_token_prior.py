"""
Training script for the Token-level Prior (autoregressive generative model).
Predicts next discrete token in a sequence of 800 tokens (8 frames x 100 tokens).
Uses Scheduled Sampling + Noise Injection to reduce exposure bias.

Usage: python train_token_prior.py
"""
import os
import hydra
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import set_seed, make_averager, get_device
from src.dataset import (
    load_or_collect, get_transform_pipeline, SequencePacmanDataset
)
from src.models import VQVAE
from src.transformers import TransformerPrior


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

    TOKENS_PER_FRAME = 100
    TOTAL_TOKENS = cfg.dataset.seq_len * TOKENS_PER_FRAME

    wandb.init(
        project=cfg.wandb.project,
        name="TokenPrior-ScheduledSampling",
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
    )

    # Load data + build sequential dataset (without sampler, for prior training)
    os.makedirs(os.path.dirname(cfg.dataset.cache_path), exist_ok=True)
    frames, labels = load_or_collect(
        cfg.dataset.cache_path, cfg.dataset.num_frames,
        danger_window=cfg.dataset.danger_window, seed=cfg.dataset.seed
    )
    transform = get_transform_pipeline()
    full_seq_ds = SequencePacmanDataset(
        frames, labels, seq_len=cfg.dataset.seq_len, transform=transform
    )

    train_size = int(0.8 * len(full_seq_ds))
    test_size = len(full_seq_ds) - train_size
    split_gen = torch.Generator().manual_seed(cfg.training.seed)
    train_seq_ds, _ = torch.utils.data.random_split(
        full_seq_ds, [train_size, test_size], generator=split_gen
    )

    train_loader = DataLoader(
        train_seq_ds,
        batch_size=cfg.model.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    # Load frozen VQ-VAE
    vqvae = load_vqvae(cfg.model.training.vqvae_checkpoint, device)
    NUM_EMBEDDINGS = vqvae.vq.num_embeddings

    # Build Prior
    set_seed(cfg.training.seed)
    transformer_prior = TransformerPrior(
        num_embeddings=NUM_EMBEDDINGS,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = optim.Adam(
        transformer_prior.parameters(),
        lr=cfg.model.training.learning_rate,
        weight_decay=cfg.model.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.model.training.scheduler_step,
        gamma=cfg.model.training.scheduler_gamma,
    )

    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "token_prior.pth")

    history = {'vocab_loss': [], 'lr': [], 'noise_prob': [], 'ss_prob': []}
    print(f"Training Token-level Prior on {device}...")

    NUM_EPOCHS = cfg.model.training.epochs
    NOISE_INITIAL = cfg.model.training.noise_initial
    SS_MAX = cfg.model.training.ss_max

    for epoch in range(NUM_EPOCHS):
        transformer_prior.train()
        loss_avg = make_averager()

        # Noise injection: descends from NOISE_INITIAL to 0 over 60% of training
        noise_prob = max(0.0, NOISE_INITIAL * (1.0 - epoch / (NUM_EPOCHS * 0.6)))

        # Scheduled sampling: grows from 0 to SS_MAX starting at epoch 5
        if epoch < 5:
            ss_prob = 0.0
        else:
            ss_prob = min(SS_MAX, (epoch - 5) / 20.0 * SS_MAX)

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] noise={noise_prob*100:.0f}% ss={ss_prob*100:.0f}%")

        for batch_seqs, _ in loop:
            B, S, C, H, W = batch_seqs.shape
            batch_seqs = batch_seqs.view(B * S, C, H, W).to(device)

            # Encode all frames via frozen VQ-VAE
            with torch.no_grad():
                z = vqvae.encoder(batch_seqs)
                z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, vqvae.vq.embedding_dim)
                dists = (
                    torch.sum(z_flat ** 2, dim=1, keepdim=True)
                    + torch.sum(vqvae.vq.embeddings ** 2, dim=1)
                    - 2 * torch.matmul(z_flat, vqvae.vq.embeddings.t())
                )
                indices = torch.argmin(dists, dim=1).view(B, TOTAL_TOKENS)

            x_input = indices[:, :-1].clone()
            y_target = indices[:, 1:]

            # Noise injection
            if noise_prob > 0:
                mask = torch.rand(x_input.shape, device=device) < noise_prob
                random_tokens = torch.randint(0, NUM_EMBEDDINGS, x_input.shape, device=device)
                x_input[mask] = random_tokens[mask]

            # Scheduled sampling
            if ss_prob > 0:
                with torch.no_grad():
                    logits_pre = transformer_prior(x_input)
                    pred_tokens = torch.argmax(logits_pre, dim=-1)
                ss_mask = torch.rand(x_input.shape, device=device) < ss_prob
                x_input = torch.where(ss_mask, pred_tokens, x_input)

            # Forward + loss
            logits = transformer_prior(x_input)
            loss = F.cross_entropy(logits.reshape(-1, NUM_EMBEDDINGS), y_target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_prior.parameters(), max_norm=1.0)
            optimizer.step()

            loss_avg(loss.item())
            loop.set_postfix(loss=loss_avg())

        ep_loss = loss_avg()
        history['vocab_loss'].append(ep_loss)
        history['lr'].append(scheduler.get_last_lr()[0])
        history['noise_prob'].append(noise_prob)
        history['ss_prob'].append(ss_prob)

        wandb.log({
            "epoch": epoch + 1,
            "vocab_loss": ep_loss,
            "lr": scheduler.get_last_lr()[0],
            "noise_prob": noise_prob,
            "ss_prob": ss_prob,
        })
        print(f"  Epoch {epoch+1}: loss={ep_loss:.4f}")

        scheduler.step()

        temp_path = ckpt_path + '.tmp'
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer_prior.state_dict(),
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