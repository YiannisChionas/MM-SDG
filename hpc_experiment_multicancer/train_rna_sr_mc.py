"""Stage 2: train the super-resolution (64x64 -> 256x256) diffusion UNet,
pooled across the 5 multi-cancer types, conditioned on BOTH a blurred
low-res image (ground-truth-downsampled during training -- see
data_utils_multicancer.HestMultiCancerPatchPairDataset) and the frozen
multi-cancer beta-VAE's expression embeddings. The stage-2 analogue of
train_rna_mc.py, matched to the same paper specs confirmed for stage 1
(plus the paper's stage-2-specific bits):
  - noise schedule: "linear gaussian diffusion process", 1000 timesteps,
    same as stage 1 ("For both Unets we used timesteps = 1000...").
  - architecture: Imagen-style cross-attention + FiLM-projected pooled
    conditioning (RnaSRConditionedUnetMC), cross_attention_dim=128
    (paper: "a dimension of 256 for the low-resolution and 128 for the
    super-resolution diffusion models").
  - low-res conditioning: blurred low-res image (paper: "a gaussian blur
    is applied to that image and it serves as input to the super-
    resolution diffusion model").
  - training budget: step-based (--total-steps, default 79000) -- same
    reasoning as stage 1 (the paper's 79k-step figure isn't tied to a
    specific stage in the text; our best reading is it applies per stage,
    see chat/DECISIONS.md).
  - batch size: default 64, NOT the paper's effective 256 -- at 256x256
    resolution this is far more memory-hungry per-sample than stage 1's
    64x64, and we only have 1 GPU. Unvalidated guess, lower if it OOMs,
    raise if there's headroom.

Trains on exactly the VAE's own fit_sample_ids (same leak-avoidance as
stage 1). Tracks an EMA copy with the same warmup schedule as stage 1
(data_utils_multicancer.EMA). Auto-resumes from checkpoint.pt, with the
same signal-based graceful shutdown as train_rna_mc.py.
"""

import argparse
import json
import signal
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from data_utils_multicancer import (
    build_patch_expr_index_sr, GeneExpressionNormalizer,
    HestMultiCancerPatchPairDataset, BetaVAEPaper, RnaSRConditionedUnetMC, EMA,
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


def save_checkpoint(path, model, ema, optimizer, step, losses):
    torch.save({
        "step": step,
        "model_state_dict": model.net.state_dict(),
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
    parser.add_argument("--total-steps", type=int, default=79000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64,
                         help="Unvalidated at 256x256 resolution -- lower if it OOMs on anakin")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--low-res-size", type=int, default=64)
    parser.add_argument("--blur-kernel-size", type=int, default=9)
    parser.add_argument("--blur-sigma", type=float, default=3.0)
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

    wsis_df = pd.read_csv(args.selected_wsis_csv)
    wsi_to_code = dict(zip(wsis_df["id"], wsis_df["oncotree_code"]))

    fit_ids = vae_ckpt["fit_sample_ids"]
    print(f"Training the SR diffusion UNet on the VAE's {len(fit_ids)} fit WSIs "
          f"(val/test stay held out)", flush=True)

    index = build_patch_expr_index_sr(args.hest_dir, fit_ids, wsi_to_code, vae_ckpt["genes"])
    print(f"Total matched (image, expression) pairs: {len(index)}", flush=True)

    def _worker_ignore_shutdown_signals(_worker_id):
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    dataset = HestMultiCancerPatchPairDataset(
        index, normalizer, image_size=args.image_size, low_res_size=args.low_res_size,
        blur_kernel_size=args.blur_kernel_size, blur_sigma=args.blur_sigma,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_ignore_shutdown_signals if args.num_workers > 0 else None,
    )
    steps_per_epoch = len(loader)
    print(f"{steps_per_epoch} steps/epoch at batch size {args.batch_size} "
          f"-- {args.total_steps} total steps ~= {args.total_steps / steps_per_epoch:.1f} epochs", flush=True)

    model = RnaSRConditionedUnetMC(vae, image_size=args.image_size).to(device)
    ema = EMA(model, decay=args.ema_decay)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.net.parameters(), lr=args.lr)

    start_step = 0
    losses = []
    resume_from = args.resume or (ckpt_path if ckpt_path.exists() else None)
    if resume_from:
        print(f"Resuming from {resume_from}", flush=True)
        ckpt = torch.load(resume_from, map_location=device)
        model.net.load_state_dict(ckpt["model_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        losses = ckpt.get("losses", [])

    if start_step >= args.total_steps:
        print(f"Already trained to step {start_step} >= target {args.total_steps}. Nothing to do.", flush=True)
        return

    stop_requested = {"flag": False}

    def handle_stop_signal(signum, frame):
        print(f"  received signal {signum} -- will checkpoint and exit "
              f"after the current step", flush=True)
        stop_requested["flag"] = True

    signal.signal(signal.SIGUSR1, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)

    def infinite_batches():
        while True:
            for batch in loader:
                yield batch

    data_iter = infinite_batches()
    model.net.train()
    t0 = time.time()
    for step in range(start_step, args.total_steps):
        low_res_cond, high_res, expr_z, _codes = next(data_iter)
        low_res_cond = low_res_cond.to(device, non_blocking=True)
        high_res = high_res.to(device, non_blocking=True)
        expr_z = expr_z.to(device, non_blocking=True)
        noise = torch.randn_like(high_res)
        timesteps = torch.randint(0, 1000, (high_res.shape[0],), device=device).long()
        noisy_high_res = noise_scheduler.add_noise(high_res, noise, timesteps)

        pred = model(noisy_high_res, low_res_cond, timesteps, expr_z)
        loss = loss_fn(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model, step=step)

        losses.append(loss.item())
        if step % 50 == 0:
            dt = time.time() - t0
            print(f"  step {step}/{args.total_steps} loss {loss.item():.5f} "
                  f"({dt/60:.1f} min elapsed this run)", flush=True)

        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.total_steps or stop_requested["flag"]:
            save_checkpoint(ckpt_path, model, ema, optimizer, step, losses)
            with open(out_dir / "losses.json", "w") as f:
                json.dump(losses, f)
            print(f"  checkpointed at step {step}", flush=True)

        if stop_requested["flag"]:
            print(f"Stopping early at step {step} due to signal -- resubmit to resume.", flush=True)
            return

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
