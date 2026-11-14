"""Train the cascaded super-resolution (stage 2) model: conditions on a
degraded low-res version of each real patch + its cancer-type class label,
learns to recover/generate the native 224x224 image.

Uses HuggingFace `accelerate` for multi-GPU support (e.g. 2 MIG slices on
obiwan), following the same Accelerator pattern as the official diffusers
train_unconditional.py example (accelerator.prepare / accelerator.backward).
Launch via `accelerate launch --multi_gpu --num_processes=N train_sr.py ...`
for N>1 GPUs, or plain `python3 train_sr.py ...` for single-GPU/CPU (works
either way -- Accelerator() with 1 process behaves like plain PyTorch).

Same auto-resume pattern as hpc_experiment/train.py -- the 224x224
resolution makes a full 10-12 epoch run take ~14-17h (measured locally:
~87min/epoch single-GPU on the pooled dataset), well over the 5h job
limit, so this MUST be split across several sbatch submissions via
slurm/submit_train.sh even with multi-GPU speedup.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from accelerate import Accelerator

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
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Per-GPU batch size (effective batch = this x num processes)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--low-res-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4,
                         help="Per-process worker count -- total workers = this x num processes")
    parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--persist-dir", type=str, default=None,
                         help="If set, copy checkpoint.pt + losses.json here after EVERY epoch "
                              "(not just at job exit) -- protects progress against a hard "
                              "SLURM timeout kill, which skips the job script's trailing cp -a")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    def log(*a, **kw):
        if accelerator.is_main_process:
            print(*a, **kw, flush=True)

    log("Using device:", device, "| num processes:", accelerator.num_processes)

    out_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    ckpt_path = out_dir / "checkpoint.pt"

    log("Building patch index ...")
    index = build_patch_index(args.data_dir, CANCER_TYPES)
    log(f"Total patches in pool: {len(index)}")
    if not index:
        raise RuntimeError(f"No patches found under {args.data_dir} -- check that the dataset was staged correctly.")

    dataset = HestPatchPairDataset(index, image_size=args.image_size, low_res_size=args.low_res_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = SRConditionedUnet(num_classes=len(CANCER_TYPES), image_size=args.image_size)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    start_epoch = 0
    losses = []
    resume_from = args.resume or (ckpt_path if ckpt_path.exists() else None)
    if resume_from:
        log(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        losses = ckpt.get("losses", [])

    if start_epoch >= args.epochs:
        log(f"Already trained to epoch {start_epoch} >= target {args.epochs}. Nothing to do.")
        return

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        epoch_losses = []
        for step, (low_res, high_res, y) in enumerate(loader):
            noise = torch.randn_like(high_res)
            timesteps = torch.randint(0, 999, (high_res.shape[0],), device=device).long()
            noisy_high_res = noise_scheduler.add_noise(high_res, noise, timesteps)

            pred = model(noisy_high_res, low_res, timesteps, y)
            loss = loss_fn(pred, noise)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            # Average the loss across all processes so logging reflects the
            # true batch loss, not just the main process's local shard.
            gathered_loss = accelerator.gather(loss.detach()).mean().item()
            epoch_losses.append(gathered_loss)
            if step % 50 == 0:
                log(f"  epoch {epoch} step {step}/{len(loader)} loss {gathered_loss:.5f}")

        losses.extend(epoch_losses)
        avg = sum(epoch_losses) / len(epoch_losses)
        dt = time.time() - t0
        log(f"Finished epoch {epoch}. Avg loss: {avg:.5f}. Time: {dt/60:.1f} min")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(ckpt_path, accelerator.unwrap_model(model), optimizer, epoch, losses)
            losses_path = out_dir / "losses.json"
            with open(losses_path, "w") as f:
                json.dump(losses, f)

            if args.persist_dir:
                persist_dir = Path(args.persist_dir)
                persist_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(ckpt_path, persist_dir / "checkpoint.pt")
                shutil.copy(losses_path, persist_dir / "losses.json")
                log(f"  Synced checkpoint to {persist_dir} (epoch {epoch})")
        accelerator.wait_for_everyone()

    log("Training complete.")


if __name__ == "__main__":
    main()
