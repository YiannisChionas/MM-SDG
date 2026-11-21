"""Shared dataset + split + sampling utilities for the multi-cancer
paper-scale beta-VAE (DECISIONS.md). Pools spots across the 5 selected
cancer types (COAD, SCCRCC, EPM, LIHB, IDC), using the gene panel from
select_genes_multicancer.py.

Covers two open items from DECISIONS.md:
- #7 (weighted DataLoader sampling): make_weighted_sampler builds a
  WeightedRandomSampler with inverse cancer-type-frequency weights, so
  each batch sees roughly equal representation across types without
  discarding any patches.
- Train/val/test split: split_train_test_wsis splits by PATIENT where
  the metadata's `patient` field is populated for every WSI of a given
  cancer type, falling back to a WSI-level split for types where it
  isn't (LIHB, confirmed empty for all its rows in the HEST catalog).
"""

import json
import os
import random
from collections import Counter

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

NORMALIZE_TARGET_SUM = 1e4


def split_train_test_wsis(selected_wsis_df, test_frac=0.2, seed=0):
    """Per-cancer-type split of WSI ids into (train_ids, test_ids).

    If the `patient` column is populated for every WSI of that type,
    splits by unique patient (so all WSIs/sections from the same patient
    land in the same split -- avoids leakage, matters most for COAD,
    whose 50 selected samples come from only ~5-10 unique patients).
    Otherwise (confirmed: LIHB has an empty `patient` field for all its
    rows) falls back to a WSI-level split for that type only.
    """
    rng = random.Random(seed)
    train_ids, test_ids = [], []

    for code, sub in selected_wsis_df.groupby("oncotree_code"):
        has_patient = "patient" in sub.columns and sub["patient"].notna().all()
        if has_patient:
            patients = sorted(sub["patient"].unique().tolist())
            rng.shuffle(patients)
            n_test = max(1, round(len(patients) * test_frac))
            test_patients = set(patients[:n_test])
            test_sub = sub[sub["patient"].isin(test_patients)]
            train_sub = sub[~sub["patient"].isin(test_patients)]
            print(f"{code}: patient-level split ({len(patients)} patients, "
                  f"{len(test_patients)} held out)", flush=True)
        else:
            ids = sorted(sub["id"].tolist())
            rng.shuffle(ids)
            n_test = max(1, round(len(ids) * test_frac))
            test_wsi_ids = set(ids[:n_test])
            test_sub = sub[sub["id"].isin(test_wsi_ids)]
            train_sub = sub[~sub["id"].isin(test_wsi_ids)]
            print(f"{code}: WSI-level split fallback (no usable `patient` field, "
                  f"{len(ids)} WSIs, {len(test_wsi_ids)} held out)", flush=True)

        train_ids.extend(train_sub["id"].tolist())
        test_ids.extend(test_sub["id"].tolist())

    return train_ids, test_ids


def _extract_normalized_expression(adata_obj, gene_names):
    gene_idx = [adata_obj.var_names.get_loc(g) for g in gene_names]
    raw = adata_obj.X[:, gene_idx]
    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    raw = np.asarray(raw, dtype=np.float32)

    total_counts = adata_obj.obs["total_counts"].to_numpy(dtype=np.float32)
    total_counts = np.clip(total_counts, a_min=1.0, a_max=None)

    normalized = raw / total_counts[:, None] * NORMALIZE_TARGET_SUM
    return np.log1p(normalized)


def build_expr_index(adata_dir, wsi_ids, wsi_to_code, gene_names):
    """Returns a list of (expr_vector, cancer_type_code) tuples, pooled
    across the given WSIs. The code is kept alongside each example so
    make_weighted_sampler can compute per-example sampling weights."""
    index = []
    for wsi_id in wsi_ids:
        adata_path = os.path.join(adata_dir, "st", f"{wsi_id}.h5ad")
        if not os.path.exists(adata_path):
            print(f"[warn] missing adata for {wsi_id}, skipping", flush=True)
            continue
        a = ad.read_h5ad(adata_path)
        expr = _extract_normalized_expression(a, gene_names)
        code = wsi_to_code[wsi_id]
        index.extend((expr[i].copy(), code) for i in range(expr.shape[0]))
        print(f"{wsi_id} ({code}): {expr.shape[0]} spots", flush=True)
    return index


def make_weighted_sampler(codes, num_samples=None):
    """WeightedRandomSampler with inverse cancer-type-frequency weights --
    each training batch ends up with roughly equal representation across
    types, without discarding any real data (DECISIONS.md #7)."""
    counts = Counter(codes)
    print(f"Cancer-type counts for weighting: {dict(counts)}", flush=True)
    weights = [1.0 / counts[c] for c in codes]
    return WeightedRandomSampler(
        weights, num_samples=num_samples or len(weights), replacement=True,
    )


class GeneExpressionNormalizer:
    """Z-score using mean/std computed once, from the TRAIN split only."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, index):
        exprs = np.stack([e for e, _ in index], axis=0)
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


class HestMultiCancerExpressionDataset(Dataset):
    """Returns (expr_z, cancer_type_code) for each pooled spot."""

    def __init__(self, index, normalizer):
        self.index = index
        self.normalizer = normalizer

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        expr, code = self.index[idx]
        z = self.normalizer.transform(expr)
        return torch.from_numpy(z.astype(np.float32)), code


def load_gene_panel(panel_path):
    with open(panel_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["genes"]


class BetaVAEPaper(nn.Module):
    """Identical architecture to hpc_experiment_vae_paper's BetaVAEPaper
    (BatchNorm1d + LeakyReLU, matching the paper's confirmed Methods) --
    duplicated here, not imported across experiment folders, per this
    repo's convention."""

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
