"""Train an independent image -> cancer-type classifier on REAL patches
only (80% of WSIs per cancer type, deterministic split -- see
data_utils_classifier.py). Used by eval_classifier.py as a judge of
whether class-conditioned synthetic images actually depict the cancer type
they were conditioned on.

Auto-resumes from checkpoint.pt. --image-size must match the resolution of
whatever synthetic samples this classifier will later judge (train a
separate checkpoint per resolution: 64 for hpc_experiment, 224 for
hpc_experiment_sr).
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils_classifier import (
    CANCER_TYPES, build_patch_index_split, HestPatchDataset, ImageToClassClassifier,
)


def save_checkpoint(path, model, optimizer, epoch, losses, image_size):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
        "image_size": image_size,
        "cancer_types": CANCER_TYPES,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15, help="Cumulative epoch target")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, required=True,
                         help="Must match the resolution of the samples this classifier will judge")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"classifier_{args.image_size}.pt"

    print("Building train/test patch index (80/20 per-WSI split)...", flush=True)
    train_index, _ = build_patch_index_split(args.data_dir, CANCER_TYPES, test_fraction=0.2, seed=args.seed)
    print(f"Train patches: {len(train_index)}", flush=True)

    dataset = HestPatchDataset(train_index, image_size=args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = ImageToClassClassifier(num_classes=len(CANCER_TYPES)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    losses = []
    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
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
        correct, total = 0, 0
        for step, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.shape[0]
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {loss.item():.5f}", flush=True)

        losses.extend(epoch_losses)
        avg = sum(epoch_losses) / len(epoch_losses)
        dt = time.time() - t0
        print(f"Finished epoch {epoch}. Avg loss: {avg:.5f}. Train acc: {correct/total:.4f}. Time: {dt/60:.1f} min",
              flush=True)

        save_checkpoint(ckpt_path, model, optimizer, epoch, losses, args.image_size)
        with open(out_dir / f"losses_{args.image_size}.json", "w") as f:
            json.dump(losses, f)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
