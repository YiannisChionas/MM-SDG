"""Train the cascaded super-resolution (stage 2) model: conditions on a
degraded low-res version of each real patch + its cancer-type class label,
learns to recover/generate the native 224x224 image.

Same auto-resume pattern as hpc_experiment/train.py -- the 224x224
resolution makes a full 10-12 epoch run take ~14-17h (measured locally:
~87min/epoch on the pooled dataset), well over the 5h job limit, so this
MUST be split across several sbatch submissions via slurm/submit_train.sh.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from data_utils_sr import CANCER_TYPES, build_patch_index, HestPatchPairDataset, SRConditionedUnet


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
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=12,
                         help="Cumulative epoch target for this invocation (not epochs-per-call)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--low-res-size", type=int, default=64)
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

    print("Building patch index ...", flush=True)
    index = build_patch_index(args.data_dir, CANCER_TYPES)
    print(f"Total patches in pool: {len(index)}", flush=True)
    if not index:
        raise RuntimeError(f"No patches found under {args.data_dir} -- check that the dataset was staged correctly.")

    dataset = HestPatchPairDataset(index, image_size=args.image_size, low_res_size=args.low_res_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = SRConditionedUnet(num_classes=len(CANCER_TYPES), image_size=args.image_size).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    losses = []
    resume_from = args.resume or (ckpt_path if ckpt_path.exists() else None)
    if resume_from:
        print(f"Resuming from {resume_from}", flush=True)
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        losses = ckpt.get("losses", [])

    if start_epoch >= args.epochs:
        print(f"Already trained to epoch {start_epoch} >= target {args.epochs}. Nothing to do.", flush=True)
        return

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        epoch_losses = []
        for step, (low_res, high_res, y) in enumerate(loader):
            low_res = low_res.to(device, non_blocking=True)
            high_res = high_res.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            noise = torch.randn_like(high_res)
            timesteps = torch.randint(0, 999, (high_res.shape[0],), device=device).long()
            noisy_high_res = noise_scheduler.add_noise(high_res, noise, timesteps)

            pred = model(noisy_high_res, low_res, timesteps, y)
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

        save_checkpoint(ckpt_path, model, optimizer, epoch, losses)
        with open(out_dir / "losses.json", "w") as f:
            json.dump(losses, f)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
