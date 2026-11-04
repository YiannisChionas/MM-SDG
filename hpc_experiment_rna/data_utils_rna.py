"""Shared dataset + model definitions for the RNA-conditioned HEST diffusion
experiment (CCRCC only).

Conditioning signal: a beta-VAE latent embedding of each spot's real gene
expression (var_50genes panel), instead of a discrete class label. Matches
the paper's approach of training a beta-VAE on expression and using its
embeddings to condition the image generator.

Uses the official HEST-bench WSI-level train/test splits (splits/*.csv) so
evaluation happens on held-out WSIs, not held-out patches from WSIs the
model already saw.
"""

import glob
import json
import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

CANCER_TYPE = "CCRCC"
GENE_PANEL_FILE = "var_50genes.json"
NORMALIZE_TARGET_SUM = 1e4


def load_gene_panel(hest_dir, cancer_type=CANCER_TYPE, panel_file=GENE_PANEL_FILE):
    with open(os.path.join(hest_dir, cancer_type, panel_file), encoding="utf-8") as f:
        data = json.load(f)
    return data["genes"]


def load_split_sample_ids(hest_dir, cancer_type, fold, split):
    """split: 'train' or 'test'."""
    csv_path = os.path.join(hest_dir, cancer_type, "splits", f"{split}_{fold}.csv")
    df = pd.read_csv(csv_path)
    return df["sample_id"].tolist()


def _extract_normalized_expression(adata_obj, gene_names):
    """Library-size normalize (target_sum=1e4) + log1p, for the given gene subset only.
    Mirrors scanpy's normalize_total + log1p, using the adata's own precomputed
    total_counts (matches HEST's own QC fields) rather than recomputing from X."""
    gene_idx = [adata_obj.var_names.get_loc(g) for g in gene_names]
    raw = adata_obj.X[:, gene_idx]
    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    raw = np.asarray(raw, dtype=np.float32)

    total_counts = adata_obj.obs["total_counts"].to_numpy(dtype=np.float32)
    total_counts = np.clip(total_counts, a_min=1.0, a_max=None)  # guard div-by-zero

    normalized = raw / total_counts[:, None] * NORMALIZE_TARGET_SUM
    return np.log1p(normalized)


def build_patch_expr_index(hest_dir, sample_ids, gene_names, cancer_type=CANCER_TYPE):
    """For each WSI in sample_ids, match patches (.h5) to their real gene
    expression (.h5ad) via barcode, and return a flat list of
    (h5_path, local_patch_idx, expr_vector) tuples -- expr_vector already
    library-size-normalized + log1p'd (z-scoring with global stats happens
    later, via GeneExpressionNormalizer, computed only from the train split).
    """
    index = []
    for sample_id in sample_ids:
        patches_path = os.path.join(hest_dir, cancer_type, "patches", f"{sample_id}.h5")
        adata_path = os.path.join(hest_dir, cancer_type, "adata", f"{sample_id}.h5ad")
        if not os.path.exists(patches_path) or not os.path.exists(adata_path):
            print(f"[warn] missing files for {sample_id}, skipping")
            continue

        a = ad.read_h5ad(adata_path)
        expr = _extract_normalized_expression(a, gene_names)  # (n_spots, n_genes)
        obs_name_to_row = {name: i for i, name in enumerate(a.obs_names)}

        with h5py.File(patches_path, "r") as f:
            barcodes = f["barcode"][:]
            n_patches = len(barcodes)

        matched = 0
        for local_idx in range(n_patches):
            bc = barcodes[local_idx][0]
            bc = bc.decode() if isinstance(bc, bytes) else str(bc)
            row = obs_name_to_row.get(bc)
            if row is None:
                continue
            index.append((patches_path, local_idx, expr[row].copy()))
            matched += 1

        print(f"{sample_id}: matched {matched}/{n_patches} patches to expression", flush=True)

    return index


class GeneExpressionNormalizer:
    """Z-score using mean/std computed once, from the TRAIN split only, to avoid
    leaking test-set statistics into preprocessing."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, index):
        exprs = np.stack([e for _, _, e in index], axis=0)
        mean = exprs.mean(axis=0)
        std = exprs.std(axis=0)
        std = np.clip(std, a_min=1e-6, a_max=None)
        return cls(mean, std)

    def transform(self, expr_vec):
        return (expr_vec - self.mean) / self.std

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["mean"], state["std"])


class HestExpressionDataset(Dataset):
    """For beta-VAE training: just the normalized expression vectors, no images."""

    def __init__(self, index, normalizer):
        self.index = index
        self.normalizer = normalizer

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        _, _, expr = self.index[idx]
        z = self.normalizer.transform(expr)
        return torch.from_numpy(z.astype(np.float32))


class HestPatchExpressionDataset(Dataset):
    """For diffusion training/sampling: image + normalized expression vector.
    Lazy .h5 loading (per-worker file handle cache), same pattern as
    hpc_experiment/data_utils.py."""

    def __init__(self, index, normalizer, image_size=64):
        self.index = index
        self.normalizer = normalizer
        self.image_size = image_size
        self._handles = {}

    def __len__(self):
        return len(self.index)

    def _get_handle(self, path):
        h = self._handles.get(path)
        if h is None:
            h = h5py.File(path, "r")
            self._handles[path] = h
        return h

    def __getitem__(self, idx):
        fp, local_idx, expr = self.index[idx]
        f = self._get_handle(fp)
        img = f["img"][local_idx]
        t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        t = F.interpolate(
            t.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        z = self.normalizer.transform(expr)
        return t, torch.from_numpy(z.astype(np.float32))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state


class BetaVAE(nn.Module):
    """Small MLP beta-VAE over the (z-scored) gene expression vector."""

    def __init__(self, input_dim=50, hidden_dim=128, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


class RnaConditionedUnet(nn.Module):
    """UNet2DModel conditioned on a frozen beta-VAE's latent embedding of the
    patch's real gene expression, broadcast as extra input channels (same
    broadcast-and-concat pattern as ClassConditionedUnet, but the
    conditioning vector now comes from a frozen pretrained encoder instead
    of an nn.Embedding lookup)."""

    def __init__(self, vae: BetaVAE, image_size=64):
        super().__init__()
        from diffusers import UNet2DModel

        self.vae = vae
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

        latent_dim = vae.latent_dim
        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=3 + latent_dim,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D",
                "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                "UpBlock2D", "UpBlock2D", "UpBlock2D",
            ),
        )

    def encode_condition(self, expr_z):
        with torch.no_grad():
            mu, _ = self.vae.encode(expr_z)  # posterior mean -- deterministic conditioning
        return mu

    def forward(self, x, t, expr_z):
        bs, ch, w, h = x.shape
        cond = self.encode_condition(expr_z).view(bs, -1, 1, 1).expand(bs, -1, w, h)
        return self.model(torch.cat((x, cond), dim=1), t).sample
