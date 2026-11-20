"""Train the paper-scale beta-VAE on the gene panel selected by
select_genes.py (CCRCC train-split WSIs, official HEST-bench fold).
Same role as hpc_experiment_rna/train_vae.py, scaled up to match the
RNA-CDM paper's reported architecture (6000/4000 hidden, 200 latent) --
~282M params vs. our existing VAE's ~19.4K.

Deliberately NOT auto-launched at full scale: see README.md's "Settings
to decide together" section before submitting a real training job. This
script just needs to run correctly; the actual --beta/--lr/--epochs
values are a separate discussion given how much bigger this model is
relative to our ~56K-spot training set.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_utils_vae_paper import (
    CANCER_TYPE, load_split_sample_ids, load_gene_panel, build_expr_index,
    GeneExpressionNormalizer, HestExpressionDataset, BetaVAEPaper, vae_loss,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--gene-panel", type=str, required=True,
                         help="Path to the JSON panel produced by select_genes.py")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden1", type=int, default=6000)
    parser.add_argument("--hidden2", type=int, default=4000)
    parser.add_argument("--latent-dim", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "vae_paper_checkpoint.pt"

    genes = load_gene_panel(args.gene_panel)
    print(f"Gene panel: {len(genes)} genes (from {args.gene_panel})", flush=True)

    train_ids = load_split_sample_ids(args.data_dir, CANCER_TYPE, args.fold, "train")
    print(f"Fold {args.fold} train WSIs ({len(train_ids)}): {train_ids}", flush=True)

    index = build_expr_index(args.data_dir, train_ids, genes, CANCER_TYPE)
    print(f"Total matched spots: {len(index)}", flush=True)

    start_epoch = 0
    losses = []
    model = BetaVAEPaper(
        input_dim=len(genes), hidden1=args.hidden1, hidden2=args.hidden2, latent_dim=args.latent_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        losses = ckpt.get("losses", [])
        normalizer = GeneExpressionNormalizer.from_state_dict(ckpt["normalizer"])
    else:
        normalizer = GeneExpressionNormalizer.fit(index)

    dataset = HestExpressionDataset(index, normalizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if start_epoch >= args.epochs:
        print(f"Already trained to epoch {start_epoch} >= target {args.epochs}. Nothing to do.", flush=True)
        return

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        epoch_losses, epoch_recon, epoch_kl = [], [], []
        for x in loader:
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
        dt = time.time() - t0
        print(f"Epoch {epoch}: loss={avg_loss:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f} time={dt/60:.1f}min",
              flush=True)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "input_dim": len(genes),
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "latent_dim": args.latent_dim,
            "beta": args.beta,
            "genes": genes,
            "normalizer": normalizer.state_dict(),
            "fold": args.fold,
            "train_sample_ids": train_ids,
        }, ckpt_path)
        with open(out_dir / "vae_paper_losses.json", "w") as f:
            json.dump(losses, f)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
