"""Stage 1: train the RNA-conditioned low-res (64x64) diffusion UNet, pooled
across the 5 multi-cancer types, conditioned on the frozen multi-cancer
beta-VAE's expression embeddings. The multi-cancer analogue of
hpc_experiment_rna/train_rna.py, but matched to the paper's stated specs
for this stage rather than inheriting the CCRCC-only script's choices
unchecked:
  - noise schedule: "linear Gaussian diffusion process" (paper's wording)
    -- NOT squaredcos_cap_v2, which the CCRCC version used.
  - batch size 256 (paper: 64/GPU x 4 GPUs = effective 256; we only have
    1 GPU here, so we use 256 directly rather than the paper's per-GPU 64).
  - training budget: the paper reports 79,000 steps (not an epoch count)
    at batch 256 -- this script is step-based (--total-steps), not
    epoch-based, specifically to match that unit.
  - architecture: Imagen-style cross-attention + FiLM-projected pooled
    conditioning (RnaConditionedUnetMC in data_utils_multicancer.py), not
    the earlier spatial-broadcast-concat version.

Trains on exactly the VAE's own `fit_sample_ids` (never val/test -- those
stay held out for the VAE's early stopping and the eventual evaluation,
same leak-avoidance as the CCRCC-only version). Tracks an EMA copy of the
trainable weights alongside the training network, matching the paper's
Methods ("Two versions of the model were maintained during training... the
EMA network" used for sampling) -- see data_utils_multicancer.EMA.
Auto-resumes from checkpoint.pt to fit the cluster's job time limit.
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
    parser.add_argument("--total-steps", type=int, default=79000,
                         help="Cumulative step target -- the paper reports 79k steps "
                              "(at their effective batch size of 256), not an epoch count")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256,
                         help="Paper: 64/GPU x 4 GPUs = effective 256. We have 1 GPU, "
                              "so use 256 directly to match the paper's effective batch.")
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
    steps_per_epoch = len(loader)
    print(f"{steps_per_epoch} steps/epoch at batch size {args.batch_size} "
          f"-- {args.total_steps} total steps ~= {args.total_steps / steps_per_epoch:.1f} epochs", flush=True)

    model = RnaConditionedUnetMC(vae, image_size=args.image_size).to(device)
    ema = EMA(model, decay=args.ema_decay)
    # Paper's Methods: "linear gaussian diffusion process", 1000 timesteps --
    # NOT squaredcos_cap_v2 (the CCRCC-only script's unchecked default).
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    loss_fn = nn.MSELoss()
    # model.net bundles the UNet + cond_proj (both trainable) -- the VAE
    # inside RnaConditionedUnetMC is frozen and excluded.
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

    def infinite_batches():
        # NOT itertools.cycle(loader) -- that caches and replays the exact
        # same first-epoch batch sequence forever with no reshuffling.
        # Calling iter(loader) fresh each pass re-triggers DataLoader's
        # shuffle=True reshuffling for every "epoch" through the data.
        while True:
            for batch in loader:
                yield batch

    # SLURM's --signal=B:USR1@<seconds> (see job_train_rna_mc.sh) sends
    # SIGUSR1 some time before the hard time-limit kill -- catching it here
    # lets us save a final checkpoint at whatever step we're on, instead of
    # losing everything back to the last periodic --checkpoint-every save
    # (which, at ~10 sec/step, can be most of an hour of work).
    stop_requested = {"flag": False}

    def handle_stop_signal(signum, frame):
        print(f"  received signal {signum} -- will checkpoint and exit "
              f"after the current step", flush=True)
        stop_requested["flag"] = True

    signal.signal(signal.SIGUSR1, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)

    data_iter = infinite_batches()
    model.net.train()
    t0 = time.time()
    for step in range(start_step, args.total_steps):
        x, expr_z, _codes = next(data_iter)
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
