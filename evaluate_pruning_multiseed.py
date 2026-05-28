"""
Multi-seed pruning evaluation script.

Loads each per-seed VQ-VAE checkpoint, applies 20% unstructured L1 magnitude
pruning to the encoder, evaluates on the test set, and saves per-seed JSON
files in the SAME FORMAT as the other multiseed scripts so that
aggregate_multiseed.py can pick them up automatically.

PREREQUISITES:
    checkpoints/vqvae_seed{SEED}.pth  for each seed in SEEDS.
    Run train_vqvae_multiseed.py first.

USAGE:
    python evaluate_pruning_multiseed.py
    (then re-run aggregate_multiseed.py to include pruning in the summary)
"""

import os
import json
import copy
import numpy as np
import torch
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils import set_seed, get_device
from src.dataset import load_or_collect, build_loaders_single
from src.models import VQVAE

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEEDS          = [7, 42, 123]
PRUNE_AMOUNT   = 0.20
CHECKPOINT_DIR = "checkpoints"
DATASET_CACHE  = "data/raw_frames_10k.npz"
NUM_FRAMES     = 10000
DANGER_WINDOW  = 15
DATASET_SEED   = 42
BATCH_SIZE     = 32
# ─────────────────────────────────────────────────────────────────────────────


def load_vqvae(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt.get("config", {})
    model = VQVAE(
        num_embeddings =cfg.get("num_embeddings",  128),
        embedding_dim  =cfg.get("embedding_dim",    64),
        commitment_cost=cfg.get("commitment_cost",  1.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def apply_pruning(model, amount):
    """L1 unstructured pruning on all Conv2d layers of the encoder — permanent."""
    for module in model.encoder.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")


def sparsity_report(model):
    total_params = total_zeros = 0
    layers = {}
    for name, module in model.encoder.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            n_zeros  = int((module.weight == 0).sum().item())
            n_params = module.weight.numel()
            layers[name] = round(100.0 * n_zeros / n_params, 2)
            total_zeros  += n_zeros
            total_params += n_params
    global_sp = round(100.0 * total_zeros / total_params, 2) if total_params else 0.0
    return global_sp, layers


def evaluate(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.to(device)
            _, _, logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())
    return {
        "accuracy"         : float(accuracy_score(all_labels, all_preds)),
        "f1_danger"        : float(f1_score(all_labels, all_preds, zero_division=0)),
        "precision_danger" : float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall_danger"    : float(recall_score(all_labels, all_preds, zero_division=0)),
        "n_test"           : len(all_labels),
    }


def main():
    device = get_device()
    print(f"Device : {device}")
    print(f"Seeds  : {SEEDS}")
    print(f"Pruning: {int(PRUNE_AMOUNT*100)}% unstructured L1\n")

    os.makedirs(os.path.dirname(DATASET_CACHE), exist_ok=True)
    frames, labels = load_or_collect(
        DATASET_CACHE, num_frames=NUM_FRAMES,
        danger_window=DANGER_WINDOW, seed=DATASET_SEED,
    )

    all_results = []

    for seed in SEEDS:
        print(f"{'='*60}\nSEED = {seed}\n{'='*60}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"vqvae_seed{seed}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Missing: {ckpt_path}\n"
                f"Run: python train_vqvae_multiseed.py training.seed={seed}"
            )

        set_seed(seed)
        _, test_loader, _ = build_loaders_single(
            frames, labels,
            batch_size=BATCH_SIZE,
            split_seed=seed,
            sampler_seed=seed,
        )

        # ── Unpruned baseline ────────────────────────────────────────────────
        model_clean   = load_vqvae(ckpt_path, device)
        m_clean       = evaluate(model_clean, test_loader, device)
        print(f"  Unpruned — acc={m_clean['accuracy']*100:.4f}%  "
              f"F1={m_clean['f1_danger']:.4f}")

        # ── Pruned ───────────────────────────────────────────────────────────
        model_pruned  = copy.deepcopy(model_clean)
        apply_pruning(model_pruned, PRUNE_AMOUNT)

        global_sp, layer_sp = sparsity_report(model_pruned)
        print(f"  Global sparsity: {global_sp:.2f}%")
        for lname, sp in layer_sp.items():
            print(f"    {lname:30s}: {sp:.1f}%")

        m_pruned = evaluate(model_pruned, test_loader, device)
        delta    = (m_pruned["accuracy"] - m_clean["accuracy"]) * 100
        print(f"  Pruned   — acc={m_pruned['accuracy']*100:.4f}%  "
              f"F1={m_pruned['f1_danger']:.4f}  Δacc={delta:+.4f}%\n")

        # ── Save per-seed JSON (same format as other multiseed scripts) ───────
        # The key 'model' = 'vqvae_pruned' lets aggregate_multiseed.py
        # distinguish it from the unpruned 'vqvae' entries.
        metrics = {
            "model"            : "vqvae_pruned",
            "seed"             : seed,
            "accuracy"         : m_pruned["accuracy"],
            "precision_danger" : m_pruned["precision_danger"],
            "recall_danger"    : m_pruned["recall_danger"],
            "f1_danger"        : m_pruned["f1_danger"],
            "n_test"           : m_pruned["n_test"],
            "delta_accuracy_pp": delta,
            "global_sparsity"  : global_sp,
            "layer_sparsity"   : layer_sp,
            # also store unpruned for reference
            "unpruned_accuracy": m_clean["accuracy"],
            "unpruned_f1"      : m_clean["f1_danger"],
        }
        out_path = os.path.join(CHECKPOINT_DIR,
                                f"vqvae_pruned_seed{seed}_metrics.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved: {out_path}")
        all_results.append(metrics)

    # ── Aggregate ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE  (mean ± std,  ddof=1)")
    print(f"{'='*60}")

    for label, key in [("Accuracy (%)", "accuracy"), ("F1 DANGER", "f1_danger"),
                        ("Precision",   "precision_danger"),
                        ("Recall",      "recall_danger")]:
        vals = np.array([r[key] for r in all_results])
        if key == "accuracy":
            vals = vals * 100
        print(f"  {label:15s}: {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")

    deltas = np.array([r["delta_accuracy_pp"] for r in all_results])
    print(f"  {'Δ acc (pp)':15s}: {deltas.mean():+.4f} ± {deltas.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
