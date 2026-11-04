"""Stage 1: train the beta-VAE on real gene expression (var_50genes panel),
CCRCC train-split WSIs only (official HEST-bench fold). Its latent space is
used in stage 2 (train_rna.py) as the diffusion model's conditioning signal.

Small/fast -- the whole point is to compress 50 genes into a low-dim latent,
not to model images, so this comfortably fits on CPU too if no GPU is free.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_utils_rna import (
    CANCER_TYPE, load_gene_panel, load_split_sample_ids, build_patch_expr_index,
    GeneExpressionNormalizer, HestExpressionDataset, BetaVAE, vae_loss,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genes = load_gene_panel(args.data_dir, CANCER_TYPE)
    print(f"Gene panel ({len(genes)} genes): {genes}", flush=True)

    train_ids = load_split_sample_ids(args.data_dir, CANCER_TYPE, args.fold, "train")
    print(f"Fold {args.fold} train WSIs ({len(train_ids)}): {train_ids}", flush=True)

    index = build_patch_expr_index(args.data_dir, train_ids, genes, CANCER_TYPE)
    print(f"Total matched spots: {len(index)}", flush=True)

    normalizer = GeneExpressionNormalizer.fit(index)

    dataset = HestExpressionDataset(index, normalizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = BetaVAE(input_dim=len(genes), hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    for epoch in range(args.epochs):
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
        print(f"Epoch {epoch}: loss={avg_loss:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f}", flush=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": len(genes),
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "genes": genes,
        "normalizer": normalizer.state_dict(),
        "fold": args.fold,
        "train_sample_ids": train_ids,
    }, out_dir / "vae_checkpoint.pt")

    with open(out_dir / "vae_losses.json", "w") as f:
        json.dump(losses, f)

    print(f"Saved VAE checkpoint to {out_dir / 'vae_checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
