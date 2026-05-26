"""
Aggregate the metrics from the 9 multi-seed runs (3 models x 3 seeds).

Run this AFTER all 9 training scripts have completed.
USAGE:
    python aggregate_multiseed.py

Produces:
  - Console output with mean ± std
  - aggregated_metrics.json
"""
import json
import os
import glob
import numpy as np


CKPT_DIR = "checkpoints"  # adjust if needed
EXPECTED_SEEDS = [42, 123, 7]


def load_metrics(model_name):
    """Load all JSON metric files for a given model."""
    pattern = os.path.join(CKPT_DIR, f"{model_name}_seed*_metrics.json")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        print(f"  ⚠️  No metrics found for {model_name}")
        return []
    runs = []
    for f in files:
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def summarize(metric_name, runs):
    """Return (mean, std, list of values)."""
    vals = [r[metric_name] for r in runs]
    return np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0, vals


def main():
    print("="*80)
    print("MULTI-SEED AGGREGATION")
    print("="*80)

    results = {}
    for model in ['baseline', 'vqvae', 'transformer']:
        print(f"\n--- {model.upper()} ---")
        runs = load_metrics(model)
        if not runs:
            continue
        seeds_found = sorted([r['seed'] for r in runs])
        print(f"  Seeds: {seeds_found}")
        if seeds_found != sorted(EXPECTED_SEEDS):
            print(f"  ⚠️  Expected seeds {sorted(EXPECTED_SEEDS)}, found {seeds_found}")

        m = {}
        for metric in ['accuracy', 'precision_danger', 'recall_danger', 'f1_danger']:
            mean, std, vals = summarize(metric, runs)
            m[metric] = {'mean': float(mean), 'std': float(std), 'values': [float(v) for v in vals]}
            label = metric.replace('_', ' ').title()
            print(f"  {label:20s}: {mean:.4f} ± {std:.4f}   (runs: {[f'{v:.4f}' for v in vals]})")
        results[model] = m

    # Save aggregated json
    out_path = os.path.join(CKPT_DIR, "aggregated_metrics.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved aggregated metrics to {out_path}")


if __name__ == "__main__":
    main()
