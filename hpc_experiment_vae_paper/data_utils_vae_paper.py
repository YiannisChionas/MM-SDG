"""Shared dataset + model definitions for the paper-scale beta-VAE
experiment (CCRCC only). Matches the architecture scale reported in the
RNA-CDM paper (Carrillo-Perez et al.): 17,655 genes in, two hidden layers
of 6,000 and 4,000, latent dim 200 -- versus our existing
hpc_experiment_rna's 50 genes / 128 hidden / 16 latent.

HEST-bench's own var_50genes.json panel is a small curated benchmark
panel, not a stand-in for "most variable genes" at paper scale, so this
module selects its OWN larger gene panel directly from the data (see
select_genes.py) rather than reusing that file. Self-contained (not
imported across experiment folders), matching this repo's convention.
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
NORMALIZE_TARGET_SUM = 1e4


def load_split_sample_ids(hest_dir, cancer_type, fold, split):
    csv_path = os.path.join(hest_dir, cancer_type, "splits", f"{split}_{fold}.csv")
    df = pd.read_csv(csv_path)
    return df["sample_id"].tolist()


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


def select_highly_variable_genes(hest_dir, sample_ids, cancer_type=CANCER_TYPE,
                                  top_k=17655, min_detection_frac=0.01):
    """Rank ALL genes (not just the var_50genes.json panel) by variance of
    their library-size-normalized + log1p expression across every spot in
    the given (train-split) WSIs, after dropping genes barely detected in
    this cohort. Returns the top_k gene names, sorted by descending
    variance. This is OUR OWN gene-selection criterion (there is no public
    "which 17,655 genes" list from the paper to reproduce exactly) --
    chosen to match the paper's panel SIZE, not its exact identity."""
    # Pass 1: not every WSI's adata necessarily carries the exact same gene
    # set (seen in practice -- e.g. one WSI missing 'MIR1302-2HG' that others
    # have), so take the intersection across all WSIs first rather than
    # assuming the first WSI's gene list is universal.
    common_genes = None
    adatas = {}
    for sample_id in sample_ids:
        adata_path = os.path.join(hest_dir, cancer_type, "adata", f"{sample_id}.h5ad")
        if not os.path.exists(adata_path):
            print(f"[warn] missing adata for {sample_id}, skipping")
            continue
        a = ad.read_h5ad(adata_path)
        adatas[sample_id] = a
        this_genes = set(a.var_names)
        common_genes = this_genes if common_genes is None else (common_genes & this_genes)
    gene_names = np.array(sorted(common_genes))
    print(f"Common genes across all {len(adatas)} train WSIs: {len(gene_names)}", flush=True)

    # Pass 2: accumulate mean/variance/detection stats over that common set.
    sum_x = np.zeros(len(gene_names), dtype=np.float64)
    sum_x2 = np.zeros(len(gene_names), dtype=np.float64)
    n_detected = np.zeros(len(gene_names), dtype=np.int64)
    n_total = 0
    for sample_id, a in adatas.items():
        expr = _extract_normalized_expression(a, gene_names)  # (n_spots, n_genes)
        sum_x += expr.sum(axis=0)
        sum_x2 += (expr ** 2).sum(axis=0)
        n_detected += (expr > 0).sum(axis=0)
        n_total += expr.shape[0]
        print(f"{sample_id}: {expr.shape[0]} spots", flush=True)

    mean = sum_x / n_total
    var = sum_x2 / n_total - mean ** 2
    detection_frac = n_detected / n_total

    keep = detection_frac >= min_detection_frac
    print(f"Genes passing min_detection_frac={min_detection_frac}: {keep.sum()}/{len(gene_names)}", flush=True)

    kept_genes = gene_names[keep]
    kept_var = var[keep]
    order = np.argsort(-kept_var)
    top_k = min(top_k, len(kept_genes))
    selected = kept_genes[order[:top_k]].tolist()
    print(f"Selected top {len(selected)} genes by variance", flush=True)
    return selected


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
    (vs 16) -- matching the RNA-CDM paper's reported architecture, applied
    to our own (smaller) selected gene panel."""

    def __init__(self, input_dim, hidden1=6000, hidden2=4000, latent_dim=200):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.SiLU(),
            nn.Linear(hidden1, hidden2), nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden2, latent_dim)
        self.fc_logvar = nn.Linear(hidden2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2), nn.SiLU(),
            nn.Linear(hidden2, hidden1), nn.SiLU(),
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
