"""
Quantitative comparison of Token-level Prior vs Frame-level Prior rollouts.
...
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import set_seed, get_device
from src.dataset import load_or_collect, get_transform_pipeline
from src.models import VQVAE
from src.transformers import TransformerPrior, FrameLevelPrior


# ---------- CONFIG ----------

VQVAE_CKPT = "/kaggle/input/models/leonardocostantini02/modeels/pytorch/default/1/vqvae_checkpoint.pth"
TOKEN_CKPT = "/kaggle/input/models/leonardocostantini02/modeels/pytorch/default/1/transformer_prior_checkpoint.pth"
FRAME_CKPT = "/kaggle/input/models/leonardocostantini02/modeels2/pytorch/default/1/frame_prior_checkpoint.pth"

DATASET_CACHE = "/kaggle/input/datasets/leonardocostantini02/dataseets/raw_frames_50k.npz"
NUM_CONTEXTS = 50
SEQ_LEN = 8
ROLLOUT_STEPS = 15
TOKENS_PER_FRAME = 100
TOTAL_TOKENS = SEQ_LEN * TOKENS_PER_FRAME
TOP_K = 64
TEMPERATURE = 0.9
SEED = 42
# ----------------------------


def psnr(img_real, img_pred, max_val=1.0):
    mse = torch.mean((img_real - img_pred) ** 2).item()
    if mse < 1e-12:
        return 60.0
    return 10.0 * np.log10((max_val ** 2) / mse)


def load_vqvae(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    m = VQVAE(
        num_embeddings=cfg.get('num_embeddings', 128),
        embedding_dim=cfg.get('embedding_dim', 64),
        commitment_cost=cfg.get('commitment_cost', 1.0),
    ).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m


def load_token_prior(path, num_embeddings, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    m = TransformerPrior(
        num_embeddings=num_embeddings,
        d_model=cfg.get('d_model', 256),
        nhead=cfg.get('nhead', 4),
        num_layers=cfg.get('num_layers', 4),
        dropout=cfg.get('dropout', 0.15),
    ).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m


def load_frame_prior(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    m = FrameLevelPrior(
        latent_channels=64, latent_h=10, latent_w=10,
        d_model=cfg.get('d_model', 256),
        nhead=cfg.get('nhead', 8),
        num_layers=cfg.get('num_layers', 4),
        dropout=cfg.get('dropout', 0.1),
    ).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m


def frames_to_tokens(frames_tensor, vqvae, device):
    with torch.no_grad():
        z = vqvae.encoder(frames_tensor.to(device))
        T, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)
        dists = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(vqvae.vq.embeddings ** 2, dim=1)
            - 2 * torch.matmul(z_flat, vqvae.vq.embeddings.t())
        )
        idx = torch.argmin(dists, dim=1)
        return idx.view(T, H * W)


def tokens_to_frame(tokens_1d, vqvae, device):
    with torch.no_grad():
        z_q = vqvae.vq.embeddings[tokens_1d]
        z_q = z_q.view(1, 10, 10, 64).permute(0, 3, 1, 2).contiguous()
        img = vqvae.decoder(z_q)
    return img.squeeze(0)


def sample_top_k(logits, k=64, temperature=1.0):
    logits = logits / temperature
    top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
    probs = F.softmax(top_vals, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return top_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def rollout_token_prior(ctx_tokens, token_prior, vqvae, n_steps, device):
    seq = ctx_tokens.view(-1).to(device)
    pred_frames = []
    for step in range(n_steps):
        new_tokens = []
        for _ in range(TOKENS_PER_FRAME):
            inp = seq[-TOTAL_TOKENS:].unsqueeze(0)
            logits = token_prior(inp)
            next_logit = logits[0, -1, :]
            next_tok = sample_top_k(next_logit, k=TOP_K, temperature=TEMPERATURE)
            seq = torch.cat([seq, next_tok.unsqueeze(0)])
            new_tokens.append(next_tok.item())
        new_tokens_t = torch.tensor(new_tokens, device=device)
        frame_img = tokens_to_frame(new_tokens_t, vqvae, device)
        pred_frames.append(frame_img.cpu())
    return pred_frames


@torch.no_grad()
def rollout_frame_prior(ctx_frames, frame_prior, vqvae, n_steps, device):
    z_ctx = vqvae.encoder(ctx_frames.to(device))
    z_hist = z_ctx.unsqueeze(0)

    pred_frames = []
    for step in range(n_steps):
        z_next = frame_prior(z_hist)
        z_q, _ = vqvae.vq(z_next)
        frame_img = vqvae.decoder(z_q).squeeze(0)
        pred_frames.append(frame_img.cpu())
        z_hist = torch.cat([z_hist, z_next.unsqueeze(1)], dim=1)
        if z_hist.shape[1] > SEQ_LEN:
            z_hist = z_hist[:, -SEQ_LEN:]
    return pred_frames


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    print("Loading VQ-VAE...")
    vqvae = load_vqvae(VQVAE_CKPT, device)
    print("Loading Token Prior...")
    token_prior = load_token_prior(TOKEN_CKPT, vqvae.vq.num_embeddings, device)
    print("Loading Frame Prior...")
    frame_prior = load_frame_prior(FRAME_CKPT, device)

    print(f"Loading frames from {DATASET_CACHE}...")
    frames_np, _ = load_or_collect(DATASET_CACHE, num_frames=50000, seed=42)
    transform = get_transform_pipeline()

    rng = np.random.RandomState(SEED)
    max_start = len(frames_np) - (SEQ_LEN + ROLLOUT_STEPS) - 1
    start_idxs = rng.choice(max_start, size=NUM_CONTEXTS, replace=False)

    psnr_token = np.zeros((NUM_CONTEXTS, ROLLOUT_STEPS))
    psnr_frame = np.zeros((NUM_CONTEXTS, ROLLOUT_STEPS))

    for i, start in enumerate(start_idxs):
        ctx_frames = []
        target_frames = []
        for t in range(SEQ_LEN):
            f = frames_np[start + t][0:170, :]
            ctx_frames.append(transform(f))
        for t in range(ROLLOUT_STEPS):
            f = frames_np[start + SEQ_LEN + t][0:170, :]
            target_frames.append(transform(f))
        ctx = torch.stack(ctx_frames)
        tgt = torch.stack(target_frames)

        ctx_tokens = frames_to_tokens(ctx, vqvae, device)
        pred_token = rollout_token_prior(ctx_tokens, token_prior, vqvae, ROLLOUT_STEPS, device)
        pred_frame = rollout_frame_prior(ctx, frame_prior, vqvae, ROLLOUT_STEPS, device)

        for t in range(ROLLOUT_STEPS):
            psnr_token[i, t] = psnr(tgt[t], pred_token[t])
            psnr_frame[i, t] = psnr(tgt[t], pred_frame[t])

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{NUM_CONTEXTS}] done. "
                  f"Token PSNR(t=1): {psnr_token[i,0]:.2f}  "
                  f"Frame PSNR(t=1): {psnr_frame[i,0]:.2f}")

    psnr_token_mean = psnr_token.mean(axis=0)
    psnr_token_std = psnr_token.std(axis=0, ddof=1)
    psnr_frame_mean = psnr_frame.mean(axis=0)
    psnr_frame_std = psnr_frame.std(axis=0, ddof=1)

    print("\n" + "="*70)
    print("PSNR (dB) vs rollout step, averaged over", NUM_CONTEXTS, "contexts")
    print("="*70)
    print(f"{'step':>4} | {'Token mean ± std':>20} | {'Frame mean ± std':>20}")
    print("-"*70)
    for t in range(ROLLOUT_STEPS):
        print(f"{t+1:>4} | {psnr_token_mean[t]:>10.2f} ± {psnr_token_std[t]:>5.2f}    | "
              f"{psnr_frame_mean[t]:>10.2f} ± {psnr_frame_std[t]:>5.2f}")

    out = {
        'config': {
            'num_contexts': NUM_CONTEXTS,
            'seq_len': SEQ_LEN,
            'rollout_steps': ROLLOUT_STEPS,
            'top_k': TOP_K,
            'temperature': TEMPERATURE,
            'seed': SEED,
        },
        'token_psnr_mean': psnr_token_mean.tolist(),
        'token_psnr_std': psnr_token_std.tolist(),
        'frame_psnr_mean': psnr_frame_mean.tolist(),
        'frame_psnr_std': psnr_frame_std.tolist(),
    }
    with open('rollout_psnr_curves.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\n✓ Saved rollout_psnr_curves.json")

    steps = np.arange(1, ROLLOUT_STEPS + 1)
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(steps, psnr_token_mean, yerr=psnr_token_std,
                 label='Token-level Prior', marker='o', capsize=3, linewidth=2)
    plt.errorbar(steps, psnr_frame_mean, yerr=psnr_frame_std,
                 label='Frame-level Prior', marker='s', capsize=3, linewidth=2)
    plt.xlabel('Rollout step $t$')
    plt.ylabel('PSNR (dB) vs ground truth')
    plt.title(f'Autoregressive rollout quality (n={NUM_CONTEXTS} contexts)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('rollout_psnr.png', dpi=150)
    print("✓ Saved rollout_psnr.png")


if __name__ == "__main__":
    main()
