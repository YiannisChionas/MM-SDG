"""Evaluate a trained image->expression regressor (train_regressor.py) as a
judge of RNA-conditioning fidelity:

1. Real baseline: run the regressor on real held-out-WSI test patches
   (upper bound -- how well expression can be predicted from a real image
   at all, independent of any generative model).
2. Paired real/synthetic: given a --samples-dir produced by
   hpc_experiment_rna/sample_rna.py or hpc_experiment_rna_sr/sample_rna_sr.py
   (real/ and synthetic/ subfolders, matching filenames = same held-out
   spot), recompute the exact same deterministic index (same fold/seed/
   max-samples) to recover the ground-truth expression vector each
   synthetic image was conditioned on, then report:
   - real_pairs_mae: regressor on the real/ copies (sanity check, should be
     close to the real baseline -- confirms the regressor + resize pipeline
     is behaving, independent of any generative model)
   - synthetic_mae: regressor on the synthetic/ images vs the SAME target
     expression used to condition them -- the actual fidelity measurement.

All MAE values are in z-scored units (per the regressor's own training
normalizer), averaged over genes and samples, so they're directly
comparable between the 64x64 and 224x224 regressors despite different
absolute pixel content.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_utils_regressor import (
    CANCER_TYPE, load_gene_panel, load_split_sample_ids, build_patch_expr_index,
    GeneExpressionNormalizer, HestPatchExpressionDataset, ImageToExpressionRegressor,
)


def load_regressor(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ImageToExpressionRegressor(output_dim=len(ckpt["genes"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    normalizer = GeneExpressionNormalizer.from_state_dict(ckpt["normalizer"])
    return model, normalizer, ckpt["genes"], ckpt["image_size"]


def mae_z(pred_z, target_z):
    return float(np.mean(np.abs(pred_z - target_z)))


def eval_real_baseline(model, normalizer, genes, image_size, data_dir, fold, device, max_samples, batch_size):
    test_ids = load_split_sample_ids(data_dir, CANCER_TYPE, fold, "test")
    index = build_patch_expr_index(data_dir, test_ids, genes, CANCER_TYPE)
    print(f"Real baseline: {len(index)} held-out (image, expression) pairs", flush=True)
    if len(index) > max_samples:
        rng = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(index), generator=rng)[:max_samples].tolist()
        index = [index[i] for i in perm]

    dataset = HestPatchExpressionDataset(index, normalizer, image_size=image_size)
    errors = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            x = torch.stack([b[0] for b in batch]).to(device)
            z_true = torch.stack([b[1] for b in batch]).numpy()
            z_pred = model(x).cpu().numpy()
            errors.append(np.abs(z_pred - z_true))
    return float(np.mean(np.concatenate(errors, axis=0)))


def recompute_paired_index(data_dir, fold, genes, max_samples, seed=0):
    """Must mirror exactly the index-building + subsampling logic in
    sample_rna.py / sample_rna_sr.py (same fold, same seed=0 default, same
    max-samples) so that index[i] lines up with real/{i:05d}.png and
    synthetic/{i:05d}.png."""
    test_ids = load_split_sample_ids(data_dir, CANCER_TYPE, fold, "test")
    index = build_patch_expr_index(data_dir, test_ids, genes, CANCER_TYPE)
    if len(index) > max_samples:
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(index), generator=rng)[:max_samples].tolist()
        index = [index[i] for i in perm]
    return index


def eval_png_folder(folder, targets_z, model, normalizer, image_size, device, batch_size):
    n = len(targets_z)
    errors = []
    paths = [folder / f"{i:05d}.png" for i in range(n)]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            imgs = []
            for p in paths[start:end]:
                img = Image.open(p).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
                t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
                imgs.append(t)
            x = torch.stack(imgs).to(device)
            z_pred = model(x).cpu().numpy()
            z_true = targets_z[start:end]
            errors.append(np.abs(z_pred - z_true))
    return float(np.mean(np.concatenate(errors, axis=0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--regressor-checkpoint", type=str, required=True)
    parser.add_argument("--samples-dir", type=str, required=True,
                         help="Output dir of sample_rna.py / sample_rna_sr.py (contains real/ and synthetic/)")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=2000,
                         help="Must match the --max-samples used when generating --samples-dir")
    parser.add_argument("--baseline-max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    model, normalizer, genes, image_size = load_regressor(args.regressor_checkpoint, device)
    print(f"Loaded regressor (image_size={image_size}, {len(genes)} genes)", flush=True)

    real_baseline_mae = eval_real_baseline(
        model, normalizer, genes, image_size, args.data_dir, args.fold, device,
        args.baseline_max_samples, args.batch_size,
    )
    print(f"Real baseline MAE (z-score units): {real_baseline_mae:.4f}", flush=True)

    index = recompute_paired_index(args.data_dir, args.fold, genes, args.max_samples)
    targets_z = np.stack([normalizer.transform(e) for _, _, e in index], axis=0).astype(np.float32)
    print(f"Recomputed paired index: {len(index)} entries", flush=True)

    samples_dir = Path(args.samples_dir)
    real_pairs_mae = eval_png_folder(
        samples_dir / "real", targets_z, model, normalizer, image_size, device, args.batch_size,
    )
    print(f"Real-pairs MAE (sanity check): {real_pairs_mae:.4f}", flush=True)

    synthetic_mae = eval_png_folder(
        samples_dir / "synthetic", targets_z, model, normalizer, image_size, device, args.batch_size,
    )
    print(f"Synthetic MAE (RNA-conditioning fidelity): {synthetic_mae:.4f}", flush=True)

    results = {
        "image_size": image_size,
        "n_genes": len(genes),
        "n_paired_samples": len(index),
        "real_baseline_mae": real_baseline_mae,
        "real_pairs_mae": real_pairs_mae,
        "synthetic_mae": synthetic_mae,
    }
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
