"""
Training script for the Temporal Transformer Classifier.
Uses a pre-trained, FROZEN VQ-VAE as feature extractor.
Predicts SAFE/DANGER for sequences of 8 frames.

Usage: python train_transformer.py
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
from src.dataset import load_or_collect, build_loaders_sequence
from src.models import VQVAE
from src.transformers import TemporalTransformerClassifier


def load_vqvae(ckpt_path, device):
    """Load a frozen pre-trained VQ-VAE."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"VQ-VAE checkpoint not found at {ckpt_path}. "
            "Run train_vqvae.py first."
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
    return model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.training.seed)
    device = get_device()

    wandb.init(
        project=cfg.wandb.project,
        name="TemporalTransformerClassifier",
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
    )

    # Load data
    os.makedirs(os.path.dirname(cfg.dataset.cache_path), exist_ok=True)
    frames, labels = load_or_collect(
        cfg.dataset.cache_path, cfg.dataset.num_frames,
        danger_window=cfg.dataset.danger_window, seed=cfg.dataset.seed
    )
    train_loader_cls, _, test_loader, _ = build_loaders_sequence(
        frames, labels,
        seq_len=cfg.dataset.seq_len,
        batch_size=cfg.training.batch_size,
        split_seed=cfg.training.seed,
        sampler_seed=cfg.training.seed,
    )

    # Load frozen VQ-VAE
    vqvae_path = cfg.model.training.vqvae_checkpoint
    print(f"Loading frozen VQ-VAE from {vqvae_path}...")
    vqvae = load_vqvae(vqvae_path, device)

    # Build Transformer
    set_seed(cfg.training.seed)
    transformer_classifier = TemporalTransformerClassifier(
        vqvae_encoder=vqvae.encoder,
        vq_module=vqvae.vq,
        seq_len=cfg.dataset.seq_len,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, transformer_classifier.parameters()),
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
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "transformer_classifier.pth")

    history = {'ce': [], 'acc_train': [], 'acc_test': [], 'lr': []}
    print(f"Training Temporal Transformer Classifier on {device}...")

    for epoch in range(cfg.model.training.epochs):
        # TRAINING
        transformer_classifier.train()
        loss_avg = make_averager()
        correct_train, total_train = 0, 0

        loop = tqdm(train_loader_cls, desc=f"Epoch [{epoch+1}/{cfg.model.training.epochs}]")
        for batch_seqs, batch_labels in loop:
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device).long()

            logits = transformer_classifier(batch_seqs)
            loss = F.cross_entropy(logits, batch_labels, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_classifier.parameters(), max_norm=1.0)
            optimizer.step()

            loss_avg(loss.item())
            correct_train += (logits.argmax(dim=1) == batch_labels).sum().item()
            total_train += batch_labels.size(0)

            loop.set_postfix(ce=loss_avg(), acc_train=100. * correct_train / total_train)

        # EVAL
        transformer_classifier.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for batch_seqs, batch_labels in test_loader:
                batch_seqs = batch_seqs.to(device)
                batch_labels = batch_labels.to(device).long()
                logits = transformer_classifier(batch_seqs)
                correct_test += (logits.argmax(dim=1) == batch_labels).sum().item()
                total_test += batch_labels.size(0)

        ep_ce = loss_avg()
        ep_acc_train = 100. * correct_train / total_train
        ep_acc_test = 100. * correct_test / total_test

        history['ce'].append(ep_ce)
        history['acc_train'].append(ep_acc_train)
        history['acc_test'].append(ep_acc_test)
        history['lr'].append(scheduler.get_last_lr()[0])

        wandb.log({
            "epoch": epoch + 1,
            "ce_loss": ep_ce,
            "acc_train": ep_acc_train,
            "acc_test": ep_acc_test,
            "lr": scheduler.get_last_lr()[0],
        })
        print(f"  Train acc: {ep_acc_train:.2f}% | Test acc: {ep_acc_test:.2f}%")

        scheduler.step()

        temp_path = ckpt_path + '.tmp'
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer_classifier.state_dict(),
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