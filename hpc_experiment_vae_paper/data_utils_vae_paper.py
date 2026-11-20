"""Shared dataset + model definitions for the paper-scale beta-VAE
experiment (CCRCC only). Matches the architecture scale reported in the
RNA-CDM paper (Carrillo-Perez et al.): two hidden layers of 6,000 and
4,000, latent dim 200 -- versus our existing hpc_experiment_rna's 50
genes / 128 hidden / 16 latent. Gene panel size is NOT forced to the
paper's 17,655 -- see select_highly_variable_genes below.

HEST-bench's own var_50genes.json panel is a small curated benchmark
panel, not a stand-in for "highly variable genes" at paper scale, so this
module selects its OWN larger gene panel directly from the data (see
select_genes.py) rather than reusing that file. Self-contained (not
imported across experiment folders), matching this repo's convention.
"""

import json
import os
import random

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

CANCER_TYPE = "CCRCC"
NORMALIZE_TARGET_SUM = 1e4


def load_split_sample_ids(hest_dir, cancer_type, fold, split):
    csv_path = os.path.join(hest_dir, cancer_type, "splits", f"{split}_{fold}.csv")
    df = pd.read_csv(csv_path)
    return df["sample_id"].tolist()


def split_train_val_ids(train_ids, val_frac=0.2, seed=0):
    """Carve a validation subset out of the official train-split WSIs, by
    WSI (not by spot), for early stopping -- separate from, and not to be
    confused with, the official held-out TEST split used for final
    evaluation. Deterministic given seed."""
    ids = sorted(train_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = ids[:n_val]
    fit_ids = ids[n_val:]
    return fit_ids, val_ids


def _extract_normalized_expression(adata_obj, gene_names):
    """Library-size normalize (target_sum=1e4) + log1p, for the given gene
    subset. Same preprocessing as hpc_experiment_rna, generalized to an
    arbitrary (and much larger) gene list."""
    gene_idx = [adata_obj.var_names.get_loc(g) for g in gene_names]
    raw = adata_obj.X[:, gene_idx]
    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    raw = np.asarray(raw, dtype=np.float32)

    total_counts = adata_obj.obs["total_counts"].to_numpy(dtype=np.float32)
    total_counts = np.clip(total_counts, a_min=1.0, a_max=None)

    normalized = raw / total_counts[:, None] * NORMALIZE_TARGET_SUM
    return np.log1p(normalized)


def select_qc_passed_genes(hest_dir, sample_ids, cancer_type=CANCER_TYPE, min_cells_frac=0.01):
    """Standard scanpy QC filtering only (sc.pp.filter_genes, min_cells as a
    fraction of total spots) over all spots in the given (train-split)
    WSIs' full transcriptome -- NO highly-variable-gene cutoff on top.

    Measured on our data: an HVG dispersion cutoff (flavor='seurat',
    default thresholds) kept only 2,913/14,166 QC-passed genes, but those
    2,913 captured just 22% of total expression variance -- throwing away
    too much real signal for a generative model (as opposed to a
    clustering/visualization use case, where HVG selection is standard).
    So this function applies only the (near-lossless: QC filtering alone
    keeps 99.9% of total raw counts) QC step, and leaves the VAE's own
    bottleneck (the latent_dim) to do the actual compression -- same
    philosophy as the paper feeding its VAE a large, only lightly
    filtered gene set.
    """
    import scanpy as sc

    adatas = []
    for sample_id in sample_ids:
        adata_path = os.path.join(hest_dir, cancer_type, "adata", f"{sample_id}.h5ad")
        if not os.path.exists(adata_path):
            print(f"[warn] missing adata for {sample_id}, skipping")
            continue
        a = ad.read_h5ad(adata_path)
        a.obs_names = [f"{sample_id}_{bc}" for bc in a.obs_names]  # avoid barcode collisions across WSIs
        adatas.append(a)

    # join="inner": not every WSI's adata necessarily carries the exact same
    # gene set (seen in practice -- e.g. one WSI missing 'MIR1302-2HG' that
    # others have), so pool over the intersection.
    pooled = ad.concat(adatas, join="inner")
    print(f"Pooled {pooled.n_obs} spots x {pooled.n_vars} common genes across {len(adatas)} train WSIs", flush=True)

    min_cells = max(1, int(min_cells_frac * pooled.n_obs))
    sc.pp.filter_genes(pooled, min_cells=min_cells)
    print(f"After filter_genes(min_cells={min_cells}): {pooled.n_vars} genes kept", flush=True)

    return pooled.var_names.tolist()


def load_gene_panel(panel_path):
    with open(panel_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["genes"]


def save_gene_panel(panel_path, genes):
    with open(panel_path, "w", encoding="utf-8") as f:
        json.dump({"genes": genes}, f)


def build_expr_index(hest_dir, sample_ids, gene_names, cancer_type=CANCER_TYPE):
    """Spot-level (no image) expression index: list of expr_vector arrays,
    one per spot, pooled across all given WSIs. No barcode-matching to
    patches needed here -- the VAE only ever sees expression vectors."""
    index = []
    for sample_id in sample_ids:
        adata_path = os.path.join(hest_dir, cancer_type, "adata", f"{sample_id}.h5ad")
        if not os.path.exists(adata_path):
            print(f"[warn] missing adata for {sample_id}, skipping")
            continue
        a = ad.read_h5ad(adata_path)
        expr = _extract_normalized_expression(a, gene_names)
        index.extend(expr[i].copy() for i in range(expr.shape[0]))
        print(f"{sample_id}: {expr.shape[0]} spots", flush=True)
    return index


class GeneExpressionNormalizer:
    """Z-score using mean/std computed once, from the TRAIN split only."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, index):
        exprs = np.stack(index, axis=0)
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
    def __init__(self, index, normalizer):
        self.index = index
        self.normalizer = normalizer

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        z = self.normalizer.transform(self.index[idx])
        return torch.from_numpy(z.astype(np.float32))


class BetaVAEPaper(nn.Module):
    """Paper-scale beta-VAE: two hidden layers (default 6000, 4000) instead
    of hpc_experiment_rna's single hidden//2-halving MLP, latent dim 200
    (vs 16) -- matching the RNA-CDM paper's reported architecture
    (confirmed directly from the paper's Methods: BatchNorm between
    layers, LeakyReLU activation), applied to our own (larger, QC-only)
    selected gene panel."""

    def __init__(self, input_dim, hidden1=6000, hidden2=4000, latent_dim=200):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.BatchNorm1d(hidden1), nn.LeakyReLU(),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(hidden2, latent_dim)
        self.fc_logvar = nn.Linear(hidden2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2), nn.BatchNorm1d(hidden2), nn.LeakyReLU(),
            nn.Linear(hidden2, hidden1), nn.BatchNorm1d(hidden1), nn.LeakyReLU(),
            nn.Linear(hidden1, input_dim),
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
