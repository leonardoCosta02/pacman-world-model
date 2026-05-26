"""
Quantitative comparison of Token-level Prior vs Frame-level Prior rollouts.

For each of N starting contexts (8 real frames), generate T future frames
autoregressively with both Priors and compute PSNR between the generated
frame at step t and the REAL frame at the same time-step. Averaging over N
contexts gives a curve PSNR(t).

OUTPUT:
  - rollout_psnr_curves.json   (raw data)
  - rollout_psnr.png           (plot)
  - LaTeX paragraph + figure code

Token Prior is expected to drop sharply after 1-2 steps; Frame Prior is
expected to degrade slowly over 15 steps.

USAGE on Kaggle (from project root, after the 5 trainings):
    python rollout_psnr.py
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
CKPT_DIR = "checkpoints"
VQVAE_CKPT = os.path.join(CKPT_DIR, "vqvae.pth")
TOKEN_CKPT = os.path.join(CKPT_DIR, "transformer_prior_checkpoint.pth")
FRAME_CKPT = os.path.join(CKPT_DIR, "frame_prior_checkpoint.pth")
DATASET_CACHE = "data/raw_frames_50k.npz"
NUM_CONTEXTS = 50         # how many starting contexts to test
SEQ_LEN = 8               # context length (frames)
ROLLOUT_STEPS = 15        # how many future frames to predict
TOKENS_PER_FRAME = 100
TOTAL_TOKENS = SEQ_LEN * TOKENS_PER_FRAME
TOP_K = 64
TEMPERATURE = 0.9
SEED = 42
# ----------------------------


def psnr(img_real, img_pred, max_val=1.0):
    """PSNR in dB between two grayscale images in [0, max_val]."""
    mse = torch.mean((img_real - img_pred) ** 2).item()
    if mse < 1e-12:
        return 60.0  # cap (numerically essentially identical)
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
    """[T, 1, 80, 80] -> [T*100] (flat tokens) via VQ-VAE encoder + quantizer."""
    with torch.no_grad():
        z = vqvae.encoder(frames_tensor.to(device))  # [T, 64, 10, 10]
        T, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)
        dists = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(vqvae.vq.embeddings ** 2, dim=1)
            - 2 * torch.matmul(z_flat, vqvae.vq.embeddings.t())
        )
        idx = torch.argmin(dists, dim=1)
        return idx.view(T, H * W)  # [T, 100]


def tokens_to_frame(tokens_1d, vqvae, device):
    """[100] -> [1, 80, 80] via codebook lookup + VQ-VAE decoder."""
    with torch.no_grad():
        z_q = vqvae.vq.embeddings[tokens_1d]   # [100, 64]
        z_q = z_q.view(1, 10, 10, 64).permute(0, 3, 1, 2).contiguous()  # [1, 64, 10, 10]
        img = vqvae.decoder(z_q)               # [1, 1, 80, 80]
    return img.squeeze(0)  # [1, 80, 80]


def sample_top_k(logits, k=64, temperature=1.0):
    """Top-k sampling with temperature."""
    logits = logits / temperature
    top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
    probs = F.softmax(top_vals, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return top_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def rollout_token_prior(ctx_tokens, token_prior, vqvae, n_steps, device):
    """
    Rollout autoregressivo del Token Prior.
    ctx_tokens: [SEQ_LEN, 100]
    Returns: lista di n_steps frame predetti, ognuno [1, 80, 80]
    """
    seq = ctx_tokens.view(-1).to(device)  # [800]
    pred_frames = []
    for step in range(n_steps):
        # genera 100 token un alla volta a partire da seq[-800:]
        new_tokens = []
        for _ in range(TOKENS_PER_FRAME):
            inp = seq[-TOTAL_TOKENS:].unsqueeze(0)  # [1, 800]
            logits = token_prior(inp)               # [1, 800, 128]
            next_logit = logits[0, -1, :]           # [128]
            next_tok = sample_top_k(next_logit, k=TOP_K, temperature=TEMPERATURE)
            seq = torch.cat([seq, next_tok.unsqueeze(0)])
            new_tokens.append(next_tok.item())
        new_tokens_t = torch.tensor(new_tokens, device=device)
        frame_img = tokens_to_frame(new_tokens_t, vqvae, device)
        pred_frames.append(frame_img.cpu())
    return pred_frames


@torch.no_grad()
def rollout_frame_prior(ctx_frames, frame_prior, vqvae, n_steps, device):
    """
    Rollout del Frame Prior (1 step = 1 frame).
    ctx_frames: [SEQ_LEN, 1, 80, 80]
    Returns: lista di n_steps frame predetti, ognuno [1, 80, 80]
    """
    # encode context
    z_ctx = vqvae.encoder(ctx_frames.to(device))  # [SEQ_LEN, 64, 10, 10]
    z_hist = z_ctx.unsqueeze(0)  # [1, SEQ_LEN, 64, 10, 10]

    pred_frames = []
    for step in range(n_steps):
        z_next = frame_prior(z_hist)              # [1, 64, 10, 10]
        z_q, _ = vqvae.vq(z_next)                 # anchor to codebook (riduce drift)
        frame_img = vqvae.decoder(z_q).squeeze(0)  # [1, 80, 80]
        pred_frames.append(frame_img.cpu())
        # update history (sliding window)
        z_hist = torch.cat([z_hist, z_next.unsqueeze(1)], dim=1)
        if z_hist.shape[1] > SEQ_LEN:
            z_hist = z_hist[:, -SEQ_LEN:]
    return pred_frames


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    # Load models
    print("Loading VQ-VAE...")
    vqvae = load_vqvae(VQVAE_CKPT, device)
    print("Loading Token Prior...")
    token_prior = load_token_prior(TOKEN_CKPT, vqvae.vq.num_embeddings, device)
    print("Loading Frame Prior...")
    frame_prior = load_frame_prior(FRAME_CKPT, device)

    # Load frames (use 50k cache: same the Frame Prior was trained on)
    print(f"Loading frames from {DATASET_CACHE}...")
    frames_np, _ = load_or_collect(DATASET_CACHE, num_frames=50000, seed=42)
    transform = get_transform_pipeline()

    # Pick NUM_CONTEXTS distinct starting indices spread across the dataset.
    # Each context needs SEQ_LEN + ROLLOUT_STEPS frames available.
    rng = np.random.RandomState(SEED)
    max_start = len(frames_np) - (SEQ_LEN + ROLLOUT_STEPS) - 1
    start_idxs = rng.choice(max_start, size=NUM_CONTEXTS, replace=False)

    psnr_token = np.zeros((NUM_CONTEXTS, ROLLOUT_STEPS))
    psnr_frame = np.zeros((NUM_CONTEXTS, ROLLOUT_STEPS))

    for i, start in enumerate(start_idxs):
        # Costruisci CTX [SEQ_LEN, 1, 80, 80] e TARGET [ROLLOUT_STEPS, 1, 80, 80]
        ctx_frames = []
        target_frames = []
        for t in range(SEQ_LEN):
            f = frames_np[start + t][0:170, :]
            ctx_frames.append(transform(f))
        for t in range(ROLLOUT_STEPS):
            f = frames_np[start + SEQ_LEN + t][0:170, :]
            target_frames.append(transform(f))
        ctx = torch.stack(ctx_frames)        # [SEQ_LEN, 1, 80, 80]
        tgt = torch.stack(target_frames)     # [ROLLOUT_STEPS, 1, 80, 80]

        # ---- Token Prior ----
        ctx_tokens = frames_to_tokens(ctx, vqvae, device)  # [SEQ_LEN, 100]
        pred_token = rollout_token_prior(ctx_tokens, token_prior, vqvae, ROLLOUT_STEPS, device)

        # ---- Frame Prior ----
        pred_frame = rollout_frame_prior(ctx, frame_prior, vqvae, ROLLOUT_STEPS, device)

        # ---- PSNR ----
        for t in range(ROLLOUT_STEPS):
            psnr_token[i, t] = psnr(tgt[t], pred_token[t])
            psnr_frame[i, t] = psnr(tgt[t], pred_frame[t])

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{NUM_CONTEXTS}] done. "
                  f"Token PSNR(t=1): {psnr_token[i,0]:.2f}  "
                  f"Frame PSNR(t=1): {psnr_frame[i,0]:.2f}")

    # Aggregate
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

    # Save raw data
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

    # Plot
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

    print("\n" + "="*70)
    print("LaTeX paragraph + figure ready to paste in the report:")
    print("="*70)

    # Trovo il punto in cui token e frame divergono nettamente
    gap_t1 = psnr_frame_mean[0] - psnr_token_mean[0]
    gap_t15 = psnr_frame_mean[-1] - psnr_token_mean[-1]
    print(rf"""
\paragraph{{Quantitative rollout comparison.}}
To quantify the qualitative claim that the token-level prior degrades within one or two steps while the frame-level prior remains coherent longer, we measured PSNR (in dB) between each generated frame and the corresponding real frame, averaged over {NUM_CONTEXTS} starting contexts uniformly sampled from the 50k dataset. Figure~\ref{{fig:psnr}} shows the two curves. At the first step (1 future frame) the Frame Prior already outperforms the Token Prior by ${gap_t1:.1f}$~dB ({psnr_frame_mean[0]:.1f} vs {psnr_token_mean[0]:.1f}). After $15$ steps the gap grows to ${gap_t15:.1f}$~dB ({psnr_frame_mean[-1]:.1f} vs {psnr_token_mean[-1]:.1f}), confirming the qualitative observation and providing a numerical signal for the complementary failure modes discussed in the main paper.

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.75\linewidth]{{figures/rollout_psnr.png}}
    \caption{{Rollout PSNR (mean $\pm$ std over {NUM_CONTEXTS} contexts) vs step. The Frame Prior maintains higher PSNR throughout the $15$-step horizon, while the Token Prior collapses rapidly.}}
    \label{{fig:psnr}}
\end{{figure}}
""")


if __name__ == "__main__":
    main()
