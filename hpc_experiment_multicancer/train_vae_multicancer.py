"""Train the paper-scale beta-VAE pooled across the 5 multi-cancer types
(COAD, SCCRCC, EPM, PRAD, IDC) -- the multi-cancer analogue of
hpc_experiment_vae_paper's CCRCC-only train_vae_paper.py. Same recipe
matched to the paper's confirmed Methods (BatchNorm+LeakyReLU, beta=0.005,
lr=3e-3 with warmup+cosine, up to 250 epochs with early stopping).

Train/test split is per-patient where possible, WSI-level fallback
otherwise (checked per-type at runtime, see
data_utils_multicancer.split_train_test_wsis and DECISIONS.md #9). The
*fit* split (for training, as opposed to the held-out early-stopping val
split) uses a WeightedRandomSampler (inverse cancer-type frequency) so
each batch sees roughly equal representation across types despite the
post-4000-cap imbalance -- DECISIONS.md #7.
"""

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_utils_multicancer import (
    split_train_test_wsis, build_expr_index, make_weighted_sampler,
    GeneExpressionNormalizer, HestMultiCancerExpressionDataset,
    load_gene_panel, BetaVAEPaper, vae_loss,
)


def warmup_cosine_lambda(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(1.0, progress)
    return 0.5 * (1 + math.cos(math.pi * progress))


def evaluate(model, loader, device, beta):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, _codes in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar, beta)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hest-dir", type=str, required=True, help="Dir containing st/*.h5ad")
    parser.add_argument("--selected-wsis-csv", type=str, required=True)
    parser.add_argument("--gene-panel", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--test-frac", type=float, default=0.2,
                         help="Fraction held out as the final test split (never touched here)")
    parser.add_argument("--val-frac", type=float, default=0.25,
                         help="Fraction of the TRAIN split further held out for early-stopping "
                              "validation. 0.25 (not 0.2) is deliberate: test_frac=0.2 removes 20%% "
                              "up front, so val_frac=0.25 of the remaining 80%% works out to exactly "
                              "60/20/20 train/val/test overall, matching the paper's split ratio.")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup-epochs", type=int, default=10)
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
    ckpt_path = out_dir / "vae_multicancer_checkpoint.pt"
    best_path = out_dir / "vae_multicancer_best.pt"

    genes = load_gene_panel(args.gene_panel)
    print(f"Gene panel: {len(genes)} genes (from {args.gene_panel})", flush=True)

    wsis_df = pd.read_csv(args.selected_wsis_csv)
    wsi_to_code = dict(zip(wsis_df["id"], wsis_df["oncotree_code"]))

    train_ids, test_ids = split_train_test_wsis(wsis_df, test_frac=args.test_frac, seed=args.seed)
    print(f"Train WSIs: {len(train_ids)}, held-out test WSIs: {len(test_ids)} (test never used here)", flush=True)

    # Further split the train WSIs into fit/val for early stopping, same
    # patient-aware logic, applied only within the train subset.
    train_df = wsis_df[wsis_df["id"].isin(train_ids)]
    fit_ids, val_ids = split_train_test_wsis(train_df, test_frac=args.val_frac, seed=args.seed)
    print(f"Fit WSIs: {len(fit_ids)}, early-stopping val WSIs: {len(val_ids)}", flush=True)

    fit_index = build_expr_index(args.hest_dir, fit_ids, wsi_to_code, genes)
    val_index = build_expr_index(args.hest_dir, val_ids, wsi_to_code, genes)
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
        normalizer = GeneExpressionNormalizer.fit(fit_index)

    fit_dataset = HestMultiCancerExpressionDataset(fit_index, normalizer)
    fit_codes = [code for _, code in fit_index]
    sampler = make_weighted_sampler(fit_codes)
    fit_loader = DataLoader(fit_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)

    val_dataset = HestMultiCancerExpressionDataset(val_index, normalizer)
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
            "fit_sample_ids": fit_ids,
            "val_sample_ids": val_ids,
            "test_sample_ids": test_ids,
        }, path)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        epoch_losses, epoch_recon, epoch_kl = [], [], []
        for x, _codes in fit_loader:
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

        with open(out_dir / "vae_multicancer_losses.json", "w") as f:
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
