"""Train the cancer-type-conditioned HEST diffusion model.

Designed for the HUA HPC cluster's 5-hour job time limit: auto-resumes from
`checkpoint.pt` in --output-dir on every invocation, so a long run can be
split across several sbatch submissions (see slurm/submit_train.sh) without
any extra bookkeeping -- just call this script again with a higher
cumulative --epochs target.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from data_utils import CANCER_TYPES, build_patch_index, HestPatchDataset, ClassConditionedUnet


def save_checkpoint(path, model, optimizer, epoch, losses):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                         help="Path to the staged hest-bench directory (expects <CANCER_TYPE>/patches/*.h5)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10,
                         help="Cumulative epoch target for this invocation (not epochs-per-call)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None,
                         help="Explicit checkpoint path to resume from (default: <output-dir>/checkpoint.pt if present)")
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

    dataset = HestPatchDataset(index, image_size=args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = ClassConditionedUnet(num_classes=len(CANCER_TYPES), image_size=args.image_size).to(device)
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
        for step, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],), device=device).long()
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            pred = model(noisy_x, timesteps, y)
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

        # Checkpoint after every epoch so a job that hits the 5h wall-clock
        # limit never loses more than one epoch of progress.
        save_checkpoint(ckpt_path, model, optimizer, epoch, losses)
        with open(out_dir / "losses.json", "w") as f:
            json.dump(losses, f)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
