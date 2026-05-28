"""
Aggregate the metrics from the multi-seed runs.

Run this AFTER all training and pruning scripts have completed:
    train_baseline_multiseed.py     (x3 seeds)
    train_vqvae_multiseed.py        (x3 seeds)
    train_transformer_multiseed.py  (x3 seeds)
    evaluate_pruning_multiseed.py   (runs all seeds internally)

USAGE:
    python aggregate_multiseed.py

Produces:
  - Console output with mean ± std
  - checkpoints/aggregated_metrics.json
"""
import json
import os
import glob
import numpy as np


CKPT_DIR       = "checkpoints"
EXPECTED_SEEDS = [7, 42, 123]


def load_metrics(model_name):
    """Load all JSON metric files for a given model prefix."""
    pattern = os.path.join(CKPT_DIR, f"{model_name}_seed*_metrics.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"  ⚠️  No metrics found for '{model_name}'")
        return []
    runs = []
    for f in files:
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def summarize(metric_name, runs):
    """Return (mean, std, list_of_values)."""
    vals = [r[metric_name] for r in runs]
    std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return float(np.mean(vals)), std, vals


def print_model_block(display_name, runs):
    seeds_found = sorted([r["seed"] for r in runs])
    print(f"  Seeds found : {seeds_found}", end="")
    if seeds_found != sorted(EXPECTED_SEEDS):
        print(f"  ⚠️  expected {sorted(EXPECTED_SEEDS)}", end="")
    print()

    m = {}
    for metric in ["accuracy", "precision_danger", "recall_danger", "f1_danger"]:
        mean, std, vals = summarize(metric, runs)
        m[metric] = {"mean": mean, "std": std,
                     "values": [float(v) for v in vals]}
        label = metric.replace("_", " ").title()
        # accuracy is stored as fraction → display as %
        if metric == "accuracy":
            print(f"  {'Accuracy (%)':20s}: "
                  f"{mean*100:.2f} ± {std*100:.2f}"
                  f"   (runs: {[f'{v*100:.2f}' for v in vals]})")
        else:
            print(f"  {label:20s}: "
                  f"{mean:.4f} ± {std:.4f}"
                  f"   (runs: {[f'{v:.4f}' for v in vals]})")
    return m


def main():
    print("=" * 80)
    print("MULTI-SEED AGGREGATION")
    print("=" * 80)

    results = {}

    # ── Standard models ───────────────────────────────────────────────────────
    for model_key, display in [
        ("baseline",    "BASELINE (continuous)"),
        ("vqvae",       "VQ-VAE (discrete)"),
        ("transformer", "TEMPORAL TRANSFORMER"),
    ]:
        print(f"\n--- {display} ---")
        runs = load_metrics(model_key)
        if runs:
            results[model_key] = print_model_block(display, runs)

    # ── VQ-VAE + pruning ──────────────────────────────────────────────────────
    print(f"\n--- VQ-VAE + 20% L1 PRUNING ---")
    pruned_runs = load_metrics("vqvae_pruned")
    if pruned_runs:
        results["vqvae_pruned"] = print_model_block("VQ-VAE pruned", pruned_runs)

        # Extra: delta accuracy
        deltas = [r["delta_accuracy_pp"] for r in pruned_runs]
        print(f"  {'Δ Accuracy (pp)':20s}: "
              f"{np.mean(deltas):+.4f} ± {np.std(deltas, ddof=1):.4f}"
              f"   (runs: {[f'{d:+.4f}' for d in deltas]})")

        # Extra: sparsity
        global_sp = [r["global_sparsity"] for r in pruned_runs]
        print(f"  {'Global sparsity':20s}: "
              f"{np.mean(global_sp):.2f} ± {np.std(global_sp, ddof=1):.2f} %")
    else:
        print("  ⚠️  Run evaluate_pruning_multiseed.py first.")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = os.path.join(CKPT_DIR, "aggregated_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved aggregated metrics to {out_path}")

    # ── LaTeX-ready summary table ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LaTeX TABLE ROWS (copy into report)")
    print("=" * 80)

    rows = [
        ("baseline",     "Baseline (continuous)"),
        ("vqvae",        "VQ-VAE (discrete)"),
        ("transformer",  "Temporal Transformer"),
        ("vqvae_pruned", "VQ-VAE + 20\\% L1 pruning"),
    ]
    for key, label in rows:
        if key not in results:
            continue
        acc_m = results[key]["accuracy"]["mean"]   * 100
        acc_s = results[key]["accuracy"]["std"]    * 100
        f1_m  = results[key]["f1_danger"]["mean"]
        f1_s  = results[key]["f1_danger"]["std"]
        print(f"{label:35s} & ${acc_m:.2f} \\pm {acc_s:.2f}$ "
              f"& ${f1_m:.2f} \\pm {f1_s:.2f}$ \\\\")


if __name__ == "__main__":
    main()
