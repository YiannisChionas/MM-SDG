"""Generate synthetic 256x256 patches via the FULL cascade: stage 1 (RNA ->
64x64) -> Gaussian blur -> stage 2 (64x64 + RNA -> 256x256), conditioned on
REAL gene expression from held-out (test-split) WSIs. Paired real/synthetic
layout, same pattern as sample_rna_mc.py, but:
  - runs BOTH diffusion stages back to back (needs both checkpoints)
  - the stage-1 output is blurred before being fed into stage 2, exactly as
    it would be used for actual end-to-end generation (NOT ground-truth-
    downsampled like during stage-2 training) -- this is also what tells us
    whether stage 2 generalizes from training-on-ground-truth to sampling-
    on-real-stage-1-output
  - both stages sample with their own EMA weights and the Karras et al.
    (2022) stochastic sampler (same hyperparameters for both, per the
    paper: rho=7, S_churn=80, S_tmin=0.05, S_tmax=50, S_noise=1.003,
    32 steps)

Resumable: counts existing .png files and only generates the remainder.
"""

import argparse
from pathlib import Path

import h5py
import pandas as pd
import torch
from diffusers import DDPMScheduler
from PIL import Image

import torch.nn.functional as F
import torchvision.transforms.functional as TF

from data_utils_multicancer import (
    GeneExpressionNormalizer, BetaVAEPaper, RnaConditionedUnetMC, RnaSRConditionedUnetMC,
    karras_sample, karras_sample_sr, build_patch_expr_index,
)


def upsample_and_blur(low_res, image_size, blur_kernel_size, blur_sigma):
    """low_res is already at stage 1's native resolution (e.g. 64x64, the
    actual DDPM output) -- just upsample to the SR stage's resolution and
    blur, unlike data_utils_multicancer.make_blurred_lowres_condition
    (training-time helper that starts from a HIGH-res ground-truth image
    and downsamples first)."""
    up = F.interpolate(low_res, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return TF.gaussian_blur(up, kernel_size=blur_kernel_size, sigma=blur_sigma)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hest-dir", type=str, required=True)
    parser.add_argument("--selected-wsis-csv", type=str, required=True)
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--stage1-checkpoint", type=str, required=True)
    parser.add_argument("--stage2-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True,
                         help="Will contain real/ and synthetic/ subfolders with matching filenames")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32,
                         help="Smaller than stage-1-only sampling -- running both stages back to "
                              "back at 256x256 is heavier")
    parser.add_argument("--low-res-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--blur-kernel-size", type=int, default=9)
    parser.add_argument("--blur-sigma", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--rho", type=float, default=7)
    parser.add_argument("--s-churn", type=float, default=80)
    parser.add_argument("--s-tmin", type=float, default=0.05)
    parser.add_argument("--s-tmax", type=float, default=50)
    parser.add_argument("--s-noise", type=float, default=1.003)
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
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

    state_key = "ema_state_dict" if args.weights == "ema" else "model_state_dict"

    model1 = RnaConditionedUnetMC(vae, image_size=args.low_res_size).to(device)
    ckpt1 = torch.load(args.stage1_checkpoint, map_location=device)
    model1.net.load_state_dict(ckpt1[state_key])
    model1.eval()
    print(f"Loaded stage-1 checkpoint from step {ckpt1['step']} ({args.weights} weights)", flush=True)

    model2 = RnaSRConditionedUnetMC(vae, image_size=args.image_size).to(device)
    ckpt2 = torch.load(args.stage2_checkpoint, map_location=device)
    model2.net.load_state_dict(ckpt2[state_key])
    model2.eval()
    print(f"Loaded stage-2 checkpoint from step {ckpt2['step']} ({args.weights} weights)", flush=True)

    wsis_df = pd.read_csv(args.selected_wsis_csv)
    wsi_to_code = dict(zip(wsis_df["id"], wsis_df["oncotree_code"]))
    test_ids = vae_ckpt["test_sample_ids"]
    print(f"Held-out test WSIs: {test_ids}", flush=True)

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

    ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    print(f"Karras et al. stochastic sampler: {args.num_steps} steps, rho={args.rho}, "
          f"S_churn={args.s_churn}, S_tmin={args.s_tmin}, S_tmax={args.s_tmax}, "
          f"S_noise={args.s_noise} (both stages)", flush=True)

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

        low_res = karras_sample(
            model1, ddpm_scheduler, exprs, args.low_res_size, device,
            num_steps=args.num_steps, rho=args.rho, s_churn=args.s_churn,
            s_tmin=args.s_tmin, s_tmax=args.s_tmax, s_noise=args.s_noise,
        )
        low_res_cond = upsample_and_blur(low_res, args.image_size, args.blur_kernel_size, args.blur_sigma)

        high_res = karras_sample_sr(
            model2, ddpm_scheduler, low_res_cond, exprs, args.image_size, device,
            num_steps=args.num_steps, rho=args.rho, s_churn=args.s_churn,
            s_tmin=args.s_tmin, s_tmax=args.s_tmax, s_noise=args.s_noise,
        )
        synth_imgs = ((high_res.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

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
