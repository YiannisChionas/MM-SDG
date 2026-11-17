"""Shared dataset + model definitions for the image-to-expression regressor
(CCRCC only). Self-contained duplicate of the relevant pieces of
hpc_experiment_rna/data_utils_rna.py (barcode matching, normalizer, dataset),
plus a small CNN regressor model -- not imported across experiment folders,
matching this repo's existing convention.

Purpose: an independent "judge" of whether RNA-conditioned synthetic images
actually encode the expression vector they were conditioned on, trained only
on REAL (image, expression) pairs, then applied to synthetic images (see
eval_regressor.py).
"""

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
    """Library-size normalize (target_sum=1e4) + log1p, for the given gene subset only."""
    gene_idx = [adata_obj.var_names.get_loc(g) for g in gene_names]
    raw = adata_obj.X[:, gene_idx]
    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    raw = np.asarray(raw, dtype=np.float32)

    total_counts = adata_obj.obs["total_counts"].to_numpy(dtype=np.float32)
    total_counts = np.clip(total_counts, a_min=1.0, a_max=None)

    normalized = raw / total_counts[:, None] * NORMALIZE_TARGET_SUM
    return np.log1p(normalized)


def build_patch_expr_index(hest_dir, sample_ids, gene_names, cancer_type=CANCER_TYPE):
    """For each WSI in sample_ids, match patches (.h5) to their real gene
    expression (.h5ad) via barcode. Returns a flat list of
    (h5_path, local_patch_idx, expr_vector) tuples, in exactly the same
    order/logic as hpc_experiment_rna's build_patch_expr_index -- this
    determinism is required so that eval_regressor.py can recompute the same
    index (same fold/seed/max-samples) that sample_rna.py / sample_rna_sr.py
    used, to recover the ground-truth expression vector behind each
    real/synthetic PNG pair by filename index."""
    index = []
    for sample_id in sample_ids:
        patches_path = os.path.join(hest_dir, cancer_type, "patches", f"{sample_id}.h5")
        adata_path = os.path.join(hest_dir, cancer_type, "adata", f"{sample_id}.h5ad")
        if not os.path.exists(patches_path) or not os.path.exists(adata_path):
            print(f"[warn] missing files for {sample_id}, skipping")
            continue

        a = ad.read_h5ad(adata_path)
        expr = _extract_normalized_expression(a, gene_names)
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
    """Z-score using mean/std computed once, from the TRAIN split only."""

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


class HestPatchExpressionDataset(Dataset):
    """Image + normalized expression vector, at a configurable resolution.
    Lazy .h5 loading (per-worker file handle cache)."""

    def __init__(self, index, normalizer, image_size=224):
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


class ImageToExpressionRegressor(nn.Module):
    """Small CNN: image -> predicted (z-scored) gene expression vector.
    Works at any input resolution (adaptive avg pool before the head), so
    the same class trains the 64x64 and 224x224 regressors."""

    def __init__(self, output_dim=50):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.head(self.features(x))
