"""Stage 1: train the RNA-conditioned low-res (64x64) diffusion UNet, pooled
across the 5 multi-cancer types, conditioned on the frozen multi-cancer
beta-VAE's expression embeddings. The multi-cancer analogue of
hpc_experiment_rna/train_rna.py.

Trains on exactly the VAE's own `fit_sample_ids` (never val/test -- those
stay held out for the VAE's early stopping and the eventual evaluation,
same leak-avoidance as the CCRCC-only version). Tracks an EMA copy of the
UNet alongside the training network, matching the paper's Methods ("Two
versions of the model were maintained during training... the EMA network"
used for sampling) -- see data_utils_multicancer.EMA. Auto-resumes from
checkpoint.pt to fit the cluster's job time limit.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from data_utils_multicancer import (
    build_patch_expr_index, GeneExpressionNormalizer,
    HestMultiCancerPatchExpressionDataset, BetaVAEPaper, RnaConditionedUnetMC, EMA,
)


def load_frozen_vae(vae_ckpt_path, device):
    ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae = BetaVAEPaper(
        input_dim=ckpt["input_dim"], hidden1=ckpt["hidden1"], hidden2=ckpt["hidden2"],
        latent_dim=ckpt["latent_dim"],
    )
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.to(device)
    vae.eval()
    normalizer = GeneExpressionNormalizer.from_state_dict(ckpt["normalizer"])
    return vae, normalizer, ckpt


def save_checkpoint(path, model, ema, optimizer, epoch, losses):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hest-dir", type=str, required=True)
    parser.add_argument("--selected-wsis-csv", type=str, required=True)
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10, help="Cumulative epoch target")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
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
    print(f"Loaded frozen VAE (latent_dim={vae.latent_dim}, "
          f"best epoch={vae_ckpt['epoch']})", flush=True)

    import pandas as pd
    wsis_df = pd.read_csv(args.selected_wsis_csv)
    wsi_to_code = dict(zip(wsis_df["id"], wsis_df["oncotree_code"]))

    fit_ids = vae_ckpt["fit_sample_ids"]
    print(f"Training the diffusion UNet on the VAE's {len(fit_ids)} fit WSIs "
          f"(val/test stay held out)", flush=True)

    index = build_patch_expr_index(args.hest_dir, fit_ids, wsi_to_code, vae_ckpt["genes"])
    print(f"Total matched (image, expression) pairs: {len(index)}", flush=True)

    dataset = HestMultiCancerPatchExpressionDataset(index, normalizer, image_size=args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = RnaConditionedUnetMC(vae, image_size=args.image_size).to(device)
    ema = EMA(model, decay=args.ema_decay)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    loss_fn = nn.MSELoss()
    # Only the UNet has trainable params -- the VAE is frozen inside RnaConditionedUnetMC.
    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.lr)

    start_epoch = 0
    losses = []
    resume_from = args.resume or (ckpt_path if ckpt_path.exists() else None)
    if resume_from:
        print(f"Resuming from {resume_from}", flush=True)
        ckpt = torch.load(resume_from, map_location=device)
        model.model.load_state_dict(ckpt["model_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
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
        for step, (x, expr_z, _codes) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            expr_z = expr_z.to(device, non_blocking=True)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 1000, (x.shape[0],), device=device).long()
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            pred = model(noisy_x, timesteps, expr_z)
            loss = loss_fn(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)

            epoch_losses.append(loss.item())
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {loss.item():.5f}", flush=True)

        losses.extend(epoch_losses)
        avg = sum(epoch_losses) / len(epoch_losses)
        dt = time.time() - t0
        print(f"Finished epoch {epoch}. Avg loss: {avg:.5f}. Time: {dt/60:.1f} min", flush=True)

        save_checkpoint(ckpt_path, model.model, ema, optimizer, epoch, losses)
        with open(out_dir / "losses.json", "w") as f:
            json.dump(losses, f)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
