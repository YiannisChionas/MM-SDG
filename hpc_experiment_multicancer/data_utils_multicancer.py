"""Shared dataset + split + sampling utilities for the multi-cancer
paper-scale beta-VAE (DECISIONS.md). Pools spots across the 5 selected
cancer types (COAD, SCCRCC, EPM, PRAD, IDC), using the gene panel from
select_genes_multicancer.py.

Covers two open items from DECISIONS.md:
- #7 (weighted DataLoader sampling): make_weighted_sampler builds a
  WeightedRandomSampler with inverse cancer-type-frequency weights, so
  each batch sees roughly equal representation across types without
  discarding any patches.
- Train/val/test split: split_train_test_wsis splits by PATIENT where
  the metadata's `patient` field is populated for every WSI of a given
  cancer type, falling back to a WSI-level split for types where it
  isn't (checked per-type at runtime -- see DECISIONS.md #9 for which
  types that applied to for the previous LIHB-based selection; PRAD
  needs re-checking since it replaced LIHB).
"""

import copy
import json
import os
import random
from collections import Counter

import anndata as ad
import h5py
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
    land in the same split -- avoids leakage). Otherwise falls back to a
    WSI-level split for that type only. Checked once on our actual
    selected 50 WSIs (DECISIONS.md #9): SCCRCC and EPM have usable
    patient info; COAD, LIHB (since replaced by PRAD) and IDC did not,
    for the specific WSIs selected -- re-check whenever the selection
    changes, the broader candidate pool's patient coverage doesn't
    necessarily carry over to whichever WSIs end up selected.
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


# ---------------------------------------------------------------------------
# Stage-1 (RNA -> low-res image) diffusion: patch/expression pairing, the
# conditioned UNet, EMA, and a Karras et al. (2022) sampler.
# ---------------------------------------------------------------------------

def build_patch_expr_index(hest_dir, wsi_ids, wsi_to_code, gene_names):
    """For each WSI, match patches_multicancer/{id}.h5 patches to their real
    gene expression (st/{id}.h5ad) via barcode. Returns a flat list of
    (h5_path, local_patch_idx, expr_vector, cancer_type_code) tuples --
    same barcode-matching pattern as hpc_experiment_rna's
    build_patch_expr_index, generalized to pool across cancer types."""
    index = []
    for wsi_id in wsi_ids:
        patches_path = os.path.join(hest_dir, "patches_multicancer", f"{wsi_id}.h5")
        adata_path = os.path.join(hest_dir, "st", f"{wsi_id}.h5ad")
        if not os.path.exists(patches_path) or not os.path.exists(adata_path):
            print(f"[warn] missing files for {wsi_id}, skipping", flush=True)
            continue

        a = ad.read_h5ad(adata_path)
        expr = _extract_normalized_expression(a, gene_names)
        obs_name_to_row = {name: i for i, name in enumerate(a.obs_names)}
        code = wsi_to_code[wsi_id]

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
            index.append((patches_path, local_idx, expr[row].copy(), code))
            matched += 1

        print(f"{wsi_id} ({code}): matched {matched}/{n_patches} patches to expression", flush=True)

    return index


class HestMultiCancerPatchExpressionDataset(Dataset):
    """For diffusion training/sampling: image + normalized expression vector
    + cancer type code (code only used for logging/weighting, not fed to
    the model -- matches the paper's un-conditioned-on-cancer-type design,
    DECISIONS.md #7). Lazy .h5 loading, same pattern as data_utils_rna.py."""

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
        fp, local_idx, expr, code = self.index[idx]
        f = self._get_handle(fp)
        img = f["img"][local_idx]
        t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        t = F.interpolate(
            t.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        z = self.normalizer.transform(expr)
        return t, torch.from_numpy(z.astype(np.float32)), code

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state


class RnaConditionedUnetMC(nn.Module):
    """UNet2DModel conditioned on the frozen multi-cancer beta-VAE's latent
    embedding of the patch's real gene expression, broadcast as extra input
    channels -- same pattern as hpc_experiment_rna's RnaConditionedUnet,
    pointed at BetaVAEPaper (latent_dim=200) instead of the CCRCC-only
    50-gene BetaVAE."""

    def __init__(self, vae: BetaVAEPaper, image_size=64):
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


class EMA:
    """Exponential moving average of a model's weights, tracked alongside
    the training network -- matches the paper's Methods ("Two versions of
    the model were maintained during training: i) the training network...
    ii) the EMA network", with the EMA network used for sampling).

    Only wraps model.model (the UNet2DModel's own parameters) since that's
    the only trainable submodule of RnaConditionedUnetMC -- the VAE inside
    it is frozen and never updated."""

    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = copy.deepcopy(model.model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for shadow_p, p in zip(self.shadow.parameters(), model.model.parameters()):
            shadow_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for shadow_b, b in zip(self.shadow.buffers(), model.model.buffers()):
            shadow_b.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state):
        self.shadow.load_state_dict(state)


def karras_sigma_schedule(num_steps, sigma_min, sigma_max, rho=7):
    """The Karras et al. noise schedule (their Eq. 5): num_steps+1 sigmas,
    descending from sigma_max to 0."""
    i = torch.arange(num_steps, dtype=torch.float64)
    sigmas = (
        sigma_max ** (1 / rho) + i / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)])


@torch.no_grad()
def karras_sample(
    model, ddpm_scheduler, expr_z, image_size, device,
    num_steps=32, rho=7, s_churn=80, s_tmin=0.05, s_tmax=50, s_noise=1.003,
):
    """Karras et al. (2022) Algorithm 2 (stochastic sampler), with the
    exact hyperparameters the RNA-CDM paper reports ("we used the extension
    presented by Karras et al. [51]... ρ=7, S_churn=80, S_tmin=0.05,
    S_tmax=50, S_noise=1.003 and 32 steps"), run against our standard
    DDPM-trained epsilon-prediction UNet (the paper's own training stays
    plain DDPM: linear Gaussian, 1000 timesteps, L2/epsilon-prediction --
    only the sampler is Karras et al.'s, applied after the fact).

    Conversion used: the standard VP (DDPM) <-> VE (EDM) correspondence --
    for the variance-preserving forward process x_t = sqrt(ab_t) x_0 +
    sqrt(1-ab_t) eps, dividing through by sqrt(ab_t) gives the equivalent
    variance-exploding sample x_sigma = x_0 + sigma * eps with
    sigma(t) = sqrt((1-ab_t)/ab_t). sigma_min/sigma_max for the Karras
    schedule are taken from the DDPMScheduler's own alphas_cumprod (the
    model's actual trained noise range), since the paper doesn't state
    these two values explicitly. No retraining needed -- inference only,
    same checkpoint as plain DDPM ancestral sampling would use."""
    ab = ddpm_scheduler.alphas_cumprod.to(device)
    sigmas_by_t = torch.sqrt((1 - ab) / ab)
    sigma_min, sigma_max = sigmas_by_t[0].item(), sigmas_by_t[-1].item()

    def nearest_t(sigma):
        return torch.argmin(torch.abs(sigmas_by_t - sigma)).item()

    def denoise(x_sigma, sigma, expr_z):
        t = nearest_t(sigma)
        ab_t = ab[t]
        x_t_scaled = x_sigma * torch.sqrt(ab_t)
        t_batch = torch.full((x_sigma.shape[0],), t, device=device, dtype=torch.long)
        eps_pred = model(x_t_scaled, t_batch, expr_z)
        return x_sigma - sigma * eps_pred

    sigmas = karras_sigma_schedule(num_steps, sigma_min, sigma_max, rho=rho).to(device)
    bs = expr_z.shape[0]
    x = torch.randn(bs, 3, image_size, image_size, device=device) * sigmas[0]

    for i in range(num_steps):
        sigma_i, sigma_next = sigmas[i], sigmas[i + 1]
        gamma = min(s_churn / num_steps, 2 ** 0.5 - 1) if s_tmin <= sigma_i <= s_tmax else 0.0
        sigma_hat = sigma_i * (1 + gamma)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x_hat = x + (sigma_hat ** 2 - sigma_i ** 2).clamp(min=0).sqrt() * eps
        else:
            x_hat = x

        d_i = (x_hat - denoise(x_hat, sigma_hat, expr_z)) / sigma_hat
        x_next = x_hat + (sigma_next - sigma_hat) * d_i

        if sigma_next > 0:
            d_next = (x_next - denoise(x_next, sigma_next, expr_z)) / sigma_next
            x_next = x_hat + (sigma_next - sigma_hat) * 0.5 * (d_i + d_next)

        x = x_next

    return x
