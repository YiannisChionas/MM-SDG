"""Train an independent image -> gene-expression regressor on REAL
(image, expression) pairs only (CCRCC train-split WSIs, official HEST-bench
fold). Used by eval_regressor.py as a judge of whether RNA-conditioned
synthetic images actually encode the expression they were conditioned on.

Auto-resumes from checkpoint.pt, same pattern as the other experiments, to
fit the cluster's 5h job limit. --image-size must match the resolution of
whatever synthetic samples this regressor will later judge (train a separate
checkpoint per resolution: 64 for hpc_experiment_rna, 224 for
hpc_experiment_rna_sr).
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils_regressor import (
    CANCER_TYPE, load_gene_panel, load_split_sample_ids, build_patch_expr_index,
    GeneExpressionNormalizer, HestPatchExpressionDataset, ImageToExpressionRegressor,
)


def save_checkpoint(path, model, optimizer, epoch, losses, normalizer, genes, image_size):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
        "normalizer": normalizer.state_dict(),
        "genes": genes,
        "image_size": image_size,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15, help="Cumulative epoch target")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, required=True,
                         help="Must match the resolution of the samples this regressor will judge")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"regressor_{args.image_size}.pt"

    genes = load_gene_panel(args.data_dir, CANCER_TYPE)
    train_ids = load_split_sample_ids(args.data_dir, CANCER_TYPE, args.fold, "train")
    print(f"Fold {args.fold} train WSIs ({len(train_ids)}): {train_ids}", flush=True)

    index = build_patch_expr_index(args.data_dir, train_ids, genes, CANCER_TYPE)
    print(f"Total matched (image, expression) pairs: {len(index)}", flush=True)

    normalizer = GeneExpressionNormalizer.fit(index)

    dataset = HestPatchExpressionDataset(index, normalizer, image_size=args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = ImageToExpressionRegressor(output_dim=len(genes)).to(device)
    loss_fn = nn.MSELoss()
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
        normalizer = GeneExpressionNormalizer.from_state_dict(ckpt["normalizer"])

    if start_epoch >= args.epochs:
        print(f"Already trained to epoch {start_epoch} >= target {args.epochs}. Nothing to do.", flush=True)
        return

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        epoch_losses = []
        for step, (x, expr_z) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            expr_z = expr_z.to(device, non_blocking=True)

            pred = model(x)
            loss = loss_fn(pred, expr_z)

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

        save_checkpoint(ckpt_path, model, optimizer, epoch, losses, normalizer, genes, args.image_size)
        with open(out_dir / f"losses_{args.image_size}.json", "w") as f:
            json.dump(losses, f)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
