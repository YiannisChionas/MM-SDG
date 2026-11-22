"""Generate synthetic low-res (64x64) patches conditioned on REAL gene
expression from held-out (test-split) WSIs -- same paired real/synthetic
layout as hpc_experiment_rna/sample_rna.py, but:
  - pooled across the 5 multi-cancer types, pointed at the multi-cancer VAE
  - samples with the EMA weights (matches the paper's Methods), not the raw
    training network
  - samples with the actual Karras et al. (2022) stochastic sampler
    (Algorithm 2), using the exact hyperparameters the RNA-CDM paper
    reports (rho=7, S_churn=80, S_tmin=0.05, S_tmax=50, S_noise=1.003,
    32 steps) -- same already-trained DDPM checkpoint, no retraining.

Resumable: counts existing .png files and only generates the remainder.
"""

import argparse
from pathlib import Path

import h5py
import pandas as pd
import torch
from diffusers import DDPMScheduler
from PIL import Image

from data_utils_multicancer import (
    GeneExpressionNormalizer, BetaVAEPaper, RnaConditionedUnetMC, karras_sample,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hest-dir", type=str, required=True)
    parser.add_argument("--selected-wsis-csv", type=str, required=True)
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--diffusion-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True,
                         help="Will contain real/ and synthetic/ subfolders with matching filenames")
    parser.add_argument("--max-samples", type=int, default=2000,
                         help="Cap on held-out spots to use (test folds can have thousands)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=32, help="Paper's reported step count")
    parser.add_argument("--rho", type=float, default=7)
    parser.add_argument("--s-churn", type=float, default=80)
    parser.add_argument("--s-tmin", type=float, default=0.05)
    parser.add_argument("--s-tmax", type=float, default=50)
    parser.add_argument("--s-noise", type=float, default=1.003)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    vae = BetaVAEPaper(
        input_dim=vae_ckpt["input_dim"], hidden1=vae_ckpt["hidden1"], hidden2=vae_ckpt["hidden2"],
        latent_dim=vae_ckpt["latent_dim"],
    )
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.to(device).eval()
    normalizer = GeneExpressionNormalizer.from_state_dict(vae_ckpt["normalizer"])

    model = RnaConditionedUnetMC(vae, image_size=args.image_size).to(device)
    diff_ckpt = torch.load(args.diffusion_checkpoint, map_location=device)
    # Sample with the EMA weights, not the raw training network -- matches
    # the paper's Methods (EMA network used for sampling).
    model.model.load_state_dict(diff_ckpt["ema_state_dict"])
    model.eval()
    print(f"Loaded diffusion checkpoint from epoch {diff_ckpt['epoch']} (EMA weights)", flush=True)

    wsis_df = pd.read_csv(args.selected_wsis_csv)
    wsi_to_code = dict(zip(wsis_df["id"], wsis_df["oncotree_code"]))
    test_ids = vae_ckpt["test_sample_ids"]
    print(f"Held-out test WSIs: {test_ids}", flush=True)

    from data_utils_multicancer import build_patch_expr_index
    index = build_patch_expr_index(args.hest_dir, test_ids, wsi_to_code, vae_ckpt["genes"])
    print(f"Total held-out (image, expression) pairs: {len(index)}", flush=True)
    if len(index) > args.max_samples:
        rng = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(index), generator=rng)[: args.max_samples].tolist()
        index = [index[i] for i in perm]
    print(f"Using {len(index)} held-out spots for paired generation", flush=True)

    out_dir = Path(args.output_dir)
    real_dir = out_dir / "real"
    synth_dir = out_dir / "synthetic"
    real_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

    ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    print(f"Karras et al. stochastic sampler: {args.num_steps} steps, rho={args.rho}, "
          f"S_churn={args.s_churn}, S_tmin={args.s_tmin}, S_tmax={args.s_tmax}, "
          f"S_noise={args.s_noise} (paper's reported settings)", flush=True)

    n_done = len(list(synth_dir.glob("*.png")))
    if n_done >= len(index):
        print(f"Already generated {n_done} >= {len(index)} samples. Nothing to do.", flush=True)
        return
    print(f"Resuming from sample {n_done}", flush=True)

    _h5_cache = {}

    def read_real_image(fp, local_idx):
        f = _h5_cache.get(fp)
        if f is None:
            f = h5py.File(fp, "r")
            _h5_cache[fp] = f
        return f["img"][local_idx]

    remaining = index[n_done:]
    pos = 0
    while pos < len(remaining):
        batch = remaining[pos: pos + args.batch_size]
        bs = len(batch)

        exprs = torch.stack([
            torch.from_numpy(normalizer.transform(e).astype("float32")) for _, _, e, _code in batch
        ]).to(device)

        x = karras_sample(
            model, ddpm_scheduler, exprs, args.image_size, device,
            num_steps=args.num_steps, rho=args.rho, s_churn=args.s_churn,
            s_tmin=args.s_tmin, s_tmax=args.s_tmax, s_noise=args.s_noise,
        )
        synth_imgs = ((x.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        for i, (fp, local_idx, _expr, _code) in enumerate(batch):
            idx = n_done + pos + i
            real_img = read_real_image(fp, local_idx)
            Image.fromarray(real_img).resize((args.image_size, args.image_size), Image.BILINEAR).save(
                real_dir / f"{idx:05d}.png"
            )
            Image.fromarray(synth_imgs[i]).save(synth_dir / f"{idx:05d}.png")

        pos += bs
        print(f"  {n_done + pos}/{len(index)}", flush=True)

    for f in _h5_cache.values():
        f.close()

    print("Paired sampling complete.", flush=True)


if __name__ == "__main__":
    main()
