"""Generate synthetic patches conditioned on REAL gene expression from
held-out (test-split) WSIs -- i.e. for each held-out spot, encode its true
expression through the frozen VAE and generate the corresponding image.

This gives a paired real/synthetic evaluation set (same filename index in
both folders = same underlying spot), stronger than an unpaired pooled
comparison: compute_fid.py's FID still works as before on the two flat
folders, and the pairing additionally allows e.g. per-pair similarity
metrics later if wanted.

Resumable: counts existing .png files and only generates the remainder.
"""

import argparse
from pathlib import Path

import h5py
import torch
from PIL import Image
from diffusers import DDPMScheduler

from data_utils_rna import (
    CANCER_TYPE, load_gene_panel, load_split_sample_ids, build_patch_expr_index,
    GeneExpressionNormalizer, BetaVAE, RnaConditionedUnet,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--diffusion-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True,
                         help="Will contain real/ and synthetic/ subfolders with matching filenames")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=2000,
                         help="Cap on held-out spots to use (test folds can have thousands)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    vae = BetaVAE(
        input_dim=vae_ckpt["input_dim"], hidden_dim=vae_ckpt["hidden_dim"], latent_dim=vae_ckpt["latent_dim"],
    )
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.to(device).eval()
    normalizer = GeneExpressionNormalizer.from_state_dict(vae_ckpt["normalizer"])

    model = RnaConditionedUnet(vae, image_size=args.image_size).to(device)
    diff_ckpt = torch.load(args.diffusion_checkpoint, map_location=device)
    model.model.load_state_dict(diff_ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded diffusion checkpoint from epoch {diff_ckpt['epoch']}", flush=True)

    genes = load_gene_panel(args.data_dir, CANCER_TYPE)
    test_ids = load_split_sample_ids(args.data_dir, CANCER_TYPE, args.fold, "test")
    print(f"Fold {args.fold} test (held-out) WSIs: {test_ids}", flush=True)

    index = build_patch_expr_index(args.data_dir, test_ids, genes, CANCER_TYPE)
    print(f"Total held-out (image, expression) pairs: {len(index)}", flush=True)
    if len(index) > args.max_samples:
        # Deterministic subsample (seeded) rather than just truncating by WSI order
        rng = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(index), generator=rng)[: args.max_samples].tolist()
        index = [index[i] for i in perm]
    print(f"Using {len(index)} held-out spots for paired generation", flush=True)

    out_dir = Path(args.output_dir)
    real_dir = out_dir / "real"
    synth_dir = out_dir / "synthetic"
    real_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

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
            torch.from_numpy(normalizer.transform(e).astype("float32")) for _, _, e in batch
        ]).to(device)

        x = torch.randn(bs, 3, args.image_size, args.image_size, device=device)
        with torch.no_grad():
            for t in noise_scheduler.timesteps:
                residual = model(x, t, exprs)
                x = noise_scheduler.step(residual, t, x).prev_sample

        synth_imgs = ((x.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

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

    print("Paired sampling complete.", flush=True)


if __name__ == "__main__":
    main()
