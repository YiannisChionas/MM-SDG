"""Cascaded paired sampling on held-out (test-split) WSIs: stage 1
(hpc_experiment_rna's trained 64x64 checkpoint) generates a low-res image
conditioned on each held-out spot's REAL gene expression; the upsampled
result + the same expression embedding condition stage 2 (this
experiment's 224x224 model) for the final high-res output.

Same paired real/synthetic evaluation protocol as hpc_experiment_rna's
sample_rna.py, but real patches are now saved at NATIVE 224x224 (no
downsampling) -- same FID-resolution-fix motivation as hpc_experiment_sr.
Resumable: counts existing .png files and only generates the remainder.
"""

import argparse
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMScheduler

from data_utils_rna_sr import (
    CANCER_TYPE, load_gene_panel, load_split_sample_ids, build_patch_expr_index,
    GeneExpressionNormalizer, BetaVAE, Stage1RnaConditionedUnet, RnaSRConditionedUnet,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--stage1-checkpoint", type=str, required=True,
                         help="hpc_experiment_rna's trained checkpoint.pt (64x64 RNA-conditioned model)")
    parser.add_argument("--stage2-checkpoint", type=str, required=True,
                         help="This experiment's trained checkpoint.pt (224x224 SR model)")
    parser.add_argument("--output-dir", type=str, required=True,
                         help="Will contain real/ and synthetic/ subfolders with matching filenames")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--low-res-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    vae = BetaVAE(input_dim=vae_ckpt["input_dim"], hidden_dim=vae_ckpt["hidden_dim"], latent_dim=vae_ckpt["latent_dim"])
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.to(device).eval()
    normalizer = GeneExpressionNormalizer.from_state_dict(vae_ckpt["normalizer"])

    stage1 = Stage1RnaConditionedUnet(vae, image_size=args.low_res_size).to(device)
    ckpt1 = torch.load(args.stage1_checkpoint, map_location=device)
    stage1.model.load_state_dict(ckpt1["model_state_dict"])
    stage1.eval()
    print(f"Loaded stage-1 checkpoint from epoch {ckpt1['epoch']}", flush=True)

    stage2 = RnaSRConditionedUnet(vae, image_size=args.image_size).to(device)
    ckpt2 = torch.load(args.stage2_checkpoint, map_location=device)
    stage2.model.load_state_dict(ckpt2["model_state_dict"])
    stage2.eval()
    print(f"Loaded stage-2 checkpoint from epoch {ckpt2['epoch']}", flush=True)

    scheduler1 = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    scheduler2 = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    genes = load_gene_panel(args.data_dir, CANCER_TYPE)
    test_ids = load_split_sample_ids(args.data_dir, CANCER_TYPE, args.fold, "test")
    print(f"Fold {args.fold} test (held-out) WSIs: {test_ids}", flush=True)

    index = build_patch_expr_index(args.data_dir, test_ids, genes, CANCER_TYPE)
    print(f"Total held-out (image, expression) pairs: {len(index)}", flush=True)
    if len(index) > args.max_samples:
        rng = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(index), generator=rng)[: args.max_samples].tolist()
        index = [index[i] for i in perm]
    print(f"Using {len(index)} held-out spots for paired cascaded generation", flush=True)

    out_dir = Path(args.output_dir)
    real_dir = out_dir / "real"
    synth_dir = out_dir / "synthetic"
    real_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

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

        expr_z = torch.stack([
            torch.from_numpy(normalizer.transform(e).astype("float32")) for _, _, e in batch
        ]).to(device)

        with torch.no_grad():
            # Stage 1: generate low-res (64x64) conditioned on real expression
            x_low = torch.randn(bs, 3, args.low_res_size, args.low_res_size, device=device)
            for t in scheduler1.timesteps:
                residual = stage1(x_low, t, expr_z)
                x_low = scheduler1.step(residual, t, x_low).prev_sample

            low_res_cond = F.interpolate(
                x_low, size=(args.image_size, args.image_size),
                mode="bilinear", align_corners=False,
            )

            # Stage 2: refine to high-res (224x224), conditioned on upsampled
            # low-res + the SAME real expression embedding
            x_high = torch.randn(bs, 3, args.image_size, args.image_size, device=device)
            for t in scheduler2.timesteps:
                residual = stage2(x_high, low_res_cond, t, expr_z)
                x_high = scheduler2.step(residual, t, x_high).prev_sample

        synth_imgs = ((x_high.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        for i, (fp, local_idx, _) in enumerate(batch):
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

    print("Cascaded paired sampling complete.", flush=True)


if __name__ == "__main__":
    main()
