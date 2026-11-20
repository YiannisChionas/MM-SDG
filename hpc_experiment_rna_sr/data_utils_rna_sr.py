"""Shared dataset + model definitions for the cascaded super-resolution
(stage 2) experiment on top of the RNA-expression-conditioned model: same
idea as hpc_experiment_sr, but conditioned on the frozen beta-VAE's gene
expression embedding (hpc_experiment_rna's mechanism) instead of a discrete
cancer-type label.

Produces native 224x224 (HEST's patch size) -- the same FID-resolution-fix
motivation as hpc_experiment_sr applies here too, and lets us test whether
the RNA-conditioned model's FID advantage over class-conditioning (167.5 vs
184.4 at 64x64) holds, strengthens, or breaks once evaluated at native
resolution through a cascade (hpc_experiment_sr's own cascade made the
class-conditioned FID *worse*, 137.3 -> 155.05 -- this experiment checks
if the same happens here).

CCRCC only, official HEST-bench fold-0 split (same as hpc_experiment_rna):
train-split WSIs for training both stages, held-out test-split WSIs for
paired evaluation.
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
    csv_path = os.path.join(hest_dir, cancer_type, "splits", f"{split}_{fold}.csv")
    df = pd.read_csv(csv_path)
    return df["sample_id"].tolist()


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


def build_patch_expr_index(hest_dir, sample_ids, gene_names, cancer_type=CANCER_TYPE):
    """Identical to hpc_experiment_rna/data_utils_rna.py's version."""
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


class HestPatchExprPairDataset(Dataset):
    """Returns (low_res_condition, high_res_target, expr_z) -- the SR-stage
    analogue of hpc_experiment_rna's HestPatchExpressionDataset, with the
    same low-res degradation (+ conditioning-augmentation noise) as
    hpc_experiment_sr's HestPatchPairDataset -- see that class's docstring
    for why: bridges the train (clean degraded real image) vs sampling
    (stage-1's actual imperfect DDPM output) distribution gap."""

    def __init__(self, index, normalizer, image_size=224, low_res_size=64, cond_aug_max_std=0.2):
        self.index = index
        self.normalizer = normalizer
        self.image_size = image_size
        self.low_res_size = low_res_size
        self.cond_aug_max_std = cond_aug_max_std
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
        high_res = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        if high_res.shape[-1] != self.image_size:
            high_res = F.interpolate(
                high_res.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        low_res = F.interpolate(
            high_res.unsqueeze(0), size=(self.low_res_size, self.low_res_size),
            mode="bilinear", align_corners=False,
        )
        low_res_upsampled = F.interpolate(
            low_res, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        if self.cond_aug_max_std > 0:
            noise_std = torch.rand(1).item() * self.cond_aug_max_std
            low_res_upsampled = low_res_upsampled + torch.randn_like(low_res_upsampled) * noise_std

        z = self.normalizer.transform(expr)
        return low_res_upsampled, high_res, torch.from_numpy(z.astype(np.float32))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state


class BetaVAE(nn.Module):
    """Exact copy of hpc_experiment_rna's BetaVAE -- duplicated so this
    folder can load that experiment's vae_checkpoint.pt without a
    cross-folder import."""

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


class Stage1RnaConditionedUnet(nn.Module):
    """Exact copy of hpc_experiment_rna's RnaConditionedUnet (64x64) --
    duplicated here so sample_rna_sr.py can load that experiment's already-
    trained checkpoint.pt directly for cascaded sampling."""

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
            mu, _ = self.vae.encode(expr_z)
        return mu

    def forward(self, x, t, expr_z):
        bs, ch, w, h = x.shape
        cond = self.encode_condition(expr_z).view(bs, -1, 1, 1).expand(bs, -1, w, h)
        return self.model(torch.cat((x, cond), dim=1), t).sample


class RnaSRConditionedUnet(nn.Module):
    """UNet2DModel conditioned on BOTH a low-res image (3 channels,
    concatenated) AND the frozen beta-VAE's expression embedding (broadcast,
    same mechanism as Stage1RnaConditionedUnet) -- the RNA-conditioning
    analogue of hpc_experiment_sr's SRConditionedUnet.

    Same deliberately-small channel width (64 base) as hpc_experiment_sr,
    for the same time-budget reasons."""

    def __init__(self, vae: BetaVAE, image_size=224):
        super().__init__()
        from diffusers import UNet2DModel

        self.vae = vae
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

        latent_dim = vae.latent_dim
        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=3 + 3 + latent_dim,  # noisy high-res + low-res cond + expr embedding
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 64, 128, 128, 256, 256),
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
            mu, _ = self.vae.encode(expr_z)
        return mu

    def forward(self, noisy_high_res, low_res_cond, t, expr_z):
        bs, ch, w, h = noisy_high_res.shape
        cond = self.encode_condition(expr_z).view(bs, -1, 1, 1).expand(bs, -1, w, h)
        net_input = torch.cat((noisy_high_res, low_res_cond, cond), dim=1)
        return self.model(net_input, t).sample
