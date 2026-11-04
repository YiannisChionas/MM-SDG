"""Stage 2: train the diffusion UNet conditioned on the frozen beta-VAE's
expression embeddings (train-split WSIs only, same HEST-bench fold as the
VAE). Auto-resumes from checkpoint.pt, same pattern as hpc_experiment/train.py,
to fit the cluster's 5h job limit.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from data_utils_rna import (
    CANCER_TYPE, load_gene_panel, load_split_sample_ids, build_patch_expr_index,
    GeneExpressionNormalizer, HestPatchExpressionDataset, BetaVAE, RnaConditionedUnet,
)


def load_frozen_vae(vae_ckpt_path, device):
    ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae = BetaVAE(
        input_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"], latent_dim=ckpt["latent_dim"],
    )
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.to(device)
    vae.eval()
    normalizer = GeneExpressionNormalizer.from_state_dict(ckpt["normalizer"])
    return vae, normalizer, ckpt


def save_checkpoint(path, model, optimizer, epoch, losses):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10, help="Cumulative epoch target")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"

    vae, normalizer, vae_ckpt = load_frozen_vae(args.vae_checkpoint, device)
    print(f"Loaded frozen VAE (latent_dim={vae.latent_dim}, fold={vae_ckpt['fold']})", flush=True)

    genes = load_gene_panel(args.data_dir, CANCER_TYPE)
    assert genes == vae_ckpt["genes"], "gene panel mismatch between VAE checkpoint and current data-dir"

    train_ids = load_split_sample_ids(args.data_dir, CANCER_TYPE, args.fold, "train")
    assert train_ids == vae_ckpt["train_sample_ids"], (
        "train split WSIs differ from the ones the VAE was fit on -- "
        "use the same --fold for both stages"
    )

    index = build_patch_expr_index(args.data_dir, train_ids, genes, CANCER_TYPE)
    print(f"Total matched (image, expression) pairs: {len(index)}", flush=True)

    dataset = HestPatchExpressionDataset(index, normalizer, image_size=args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = RnaConditionedUnet(vae, image_size=args.image_size).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    loss_fn = nn.MSELoss()
    # Only the UNet has trainable params -- the VAE is frozen inside RnaConditionedUnet.
    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.lr)

    start_epoch = 0
    losses = []
    resume_from = args.resume or (ckpt_path if ckpt_path.exists() else None)
    if resume_from:
        print(f"Resuming from {resume_from}", flush=True)
        ckpt = torch.load(resume_from, map_location=device)
        model.model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        losses = ckpt.get("losses", [])

    if start_epoch >= args.epochs:
        print(f"Already trained to epoch {start_epoch} >= target {args.epochs}. Nothing to do.", flush=True)
        return

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.model.train()
        epoch_losses = []
        for step, (x, expr_z) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            expr_z = expr_z.to(device, non_blocking=True)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],), device=device).long()
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            pred = model(noisy_x, timesteps, expr_z)
            loss = loss_fn(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {loss.item():.5f}", flush=True)

        losses.extend(epoch_losses)
        avg = sum(epoch_losses) / len(epoch_losses)
        dt = time.time() - t0
        print(f"Finished epoch {epoch}. Avg loss: {avg:.5f}. Time: {dt/60:.1f} min", flush=True)

        # Only save the UNet (model.model) -- the VAE is a fixed, separately
        # versioned checkpoint (--vae-checkpoint), no need to duplicate it here.
        save_checkpoint(ckpt_path, model.model, optimizer, epoch, losses)
        with open(out_dir / "losses.json", "w") as f:
            json.dump(losses, f)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
