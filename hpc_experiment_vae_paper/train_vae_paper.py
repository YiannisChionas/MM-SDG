"""Train the paper-scale beta-VAE on the gene panel selected by
select_genes.py (CCRCC train-split WSIs, official HEST-bench fold).
Same role as hpc_experiment_rna/train_vae.py, scaled up to match the
RNA-CDM paper's reported architecture (6000/4000 hidden, 200 latent,
BatchNorm+LeakyReLU) -- ~220.4M params at our 14,166-gene panel vs. our
existing small VAE's ~19.4K.

Settings matched to the paper's own Methods (confirmed directly from the
paper, not assumed): beta=0.005, batch size 128 (NOT changed here -- kept
at our own default to use cluster resources fully, a deliberate
divergence), 250 epochs with early stopping on a held-out validation
split carved out of the train WSIs (separate from the official test
split). See README.md's "Settings to decide together" for what's still
open (overfitting risk, actual batch-size/partition benchmark).
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_utils_vae_paper import (
    CANCER_TYPE, load_split_sample_ids, split_train_val_ids, load_gene_panel, build_expr_index,
    GeneExpressionNormalizer, HestExpressionDataset, BetaVAEPaper, vae_loss,
)


def warmup_cosine_lambda(epoch, warmup_epochs, total_epochs):
    """Linear warmup from 0 to lr over warmup_epochs, then cosine decay to
    0 over the remaining epochs -- matches the paper's "warm-up and a
    cosine learning rate scheduler" (exact warmup length isn't stated in
    the paper, so --warmup-epochs is our own choice, default 10)."""
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(1.0, progress)
    return 0.5 * (1 + math.cos(math.pi * progress))


def evaluate(model, loader, device, beta):
    model.eval()
    losses = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar, beta)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--gene-panel", type=str, required=True,
                         help="Path to the JSON panel produced by select_genes.py")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=250, help="Max epochs; early stopping may stop sooner")
    parser.add_argument("--patience", type=int, default=20,
                         help="Stop if val loss hasn't improved for this many epochs")
    parser.add_argument("--val-frac", type=float, default=0.2,
                         help="Fraction of train-split WSIs held out for early-stopping validation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup-epochs", type=int, default=10,
                         help="Linear LR warmup length; followed by cosine decay to 0 over the "
                              "remaining epochs (paper uses 'a warm-up and a cosine learning rate "
                              "scheduler' but doesn't state the warmup length)")
    parser.add_argument("--hidden1", type=int, default=6000)
    parser.add_argument("--hidden2", type=int, default=4000)
    parser.add_argument("--latent-dim", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "vae_paper_checkpoint.pt"
    best_path = out_dir / "vae_paper_best.pt"

    genes = load_gene_panel(args.gene_panel)
    print(f"Gene panel: {len(genes)} genes (from {args.gene_panel})", flush=True)

    train_ids = load_split_sample_ids(args.data_dir, CANCER_TYPE, args.fold, "train")
    fit_ids, val_ids = split_train_val_ids(train_ids, val_frac=args.val_frac, seed=args.seed)
    print(f"Fold {args.fold}: {len(fit_ids)} fit WSIs, {len(val_ids)} early-stopping val WSIs "
          f"(out of {len(train_ids)} official train-split WSIs)", flush=True)

    fit_index = build_expr_index(args.data_dir, fit_ids, genes, CANCER_TYPE)
    val_index = build_expr_index(args.data_dir, val_ids, genes, CANCER_TYPE)
    print(f"Fit spots: {len(fit_index)}, val spots: {len(val_index)}", flush=True)

    model = BetaVAEPaper(
        input_dim=len(genes), hidden1=args.hidden1, hidden2=args.hidden2, latent_dim=args.latent_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: warmup_cosine_lambda(e, args.warmup_epochs, args.epochs),
    )

    start_epoch = 0
    losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    early_stopped = False

    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        losses = ckpt.get("losses", [])
        val_losses = ckpt.get("val_losses", [])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        epochs_since_improvement = ckpt.get("epochs_since_improvement", 0)
        early_stopped = ckpt.get("early_stopped", False)
        normalizer = GeneExpressionNormalizer.from_state_dict(ckpt["normalizer"])
    else:
        normalizer = GeneExpressionNormalizer.fit(fit_index)  # fit ONLY on the fit subset, never val/test

    fit_dataset = HestExpressionDataset(fit_index, normalizer)
    fit_loader = DataLoader(fit_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataset = HestExpressionDataset(val_index, normalizer)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if early_stopped:
        print(f"Already early-stopped at epoch {start_epoch - 1} "
              f"(best val loss {best_val_loss:.4f}). Nothing to do.", flush=True)
        return
    if start_epoch >= args.epochs:
        print(f"Already trained to epoch {start_epoch} >= target {args.epochs}. Nothing to do.", flush=True)
        return

    def save_checkpoint(path):
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "losses": losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_since_improvement": epochs_since_improvement,
            "early_stopped": early_stopped,
            "input_dim": len(genes),
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "latent_dim": args.latent_dim,
            "beta": args.beta,
            "genes": genes,
            "normalizer": normalizer.state_dict(),
            "fold": args.fold,
            "fit_sample_ids": fit_ids,
            "val_sample_ids": val_ids,
        }, path)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        epoch_losses, epoch_recon, epoch_kl = [], [], []
        for x in fit_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, recon_l, kl_l = vae_loss(recon, x, mu, logvar, args.beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_recon.append(recon_l.item())
            epoch_kl.append(kl_l.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_recon = sum(epoch_recon) / len(epoch_recon)
        avg_kl = sum(epoch_kl) / len(epoch_kl)
        losses.append(avg_loss)

        val_loss = evaluate(model, val_loader, device, args.beta)
        val_losses.append(val_loss)

        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: loss={avg_loss:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f} "
              f"val_loss={val_loss:.4f} lr={cur_lr:.2e} time={dt/60:.1f}min", flush=True)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            save_checkpoint(best_path)
            print(f"  New best val loss: {best_val_loss:.4f} -- saved {best_path}", flush=True)
        else:
            epochs_since_improvement += 1

        with open(out_dir / "vae_paper_losses.json", "w") as f:
            json.dump({"train": losses, "val": val_losses}, f)

        if epochs_since_improvement >= args.patience:
            early_stopped = True
            print(f"Early stopping at epoch {epoch}: no val improvement for {args.patience} epochs. "
                  f"Best val loss: {best_val_loss:.4f} (see {best_path}).", flush=True)
            save_checkpoint(ckpt_path)
            break

        save_checkpoint(ckpt_path)

    if not early_stopped:
        print("Training complete (reached max epochs without early stopping).", flush=True)
    print(f"Best model (val loss {best_val_loss:.4f}) saved at {best_path}", flush=True)


if __name__ == "__main__":
    main()
